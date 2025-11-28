from __future__ import annotations

import json
import os
from typing import List, Sequence

import chromadb

from models import RecommendationRequest, Resume


class ResumeVectorStore:
    """Manages all access to the ChromaDB resume collection."""

    def __init__(self, persist_directory: str, collection_name: str = "resumes") -> None:
        os.makedirs(persist_directory, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _metadata_from_resume(self, resume: Resume) -> dict:
        metadata = resume.model_dump()
        metadata["skills"] = json.dumps(resume.skills)
        return metadata

    def _resume_from_metadata(self, metadata: dict) -> Resume:
        skills_raw = metadata.get("skills", "[]")
        skills = json.loads(skills_raw) if isinstance(skills_raw, str) else skills_raw
        return Resume(
            id=str(metadata["id"]),
            name=metadata["name"],
            email=metadata["email"],
            role=metadata["role"],
            skills=skills,
            experience=metadata["experience"],
            summary=metadata["summary"],
        )

    def add_resume(self, resume: Resume, embedding: Sequence[float], document: str) -> None:
        self._collection.upsert(
            ids=[resume.id],
            embeddings=[list(embedding)],
            documents=[document],
            metadatas=[self._metadata_from_resume(resume)],
        )

    def get_all_resumes(self) -> List[Resume]:
        result = self._collection.get(include=["metadatas"])
        if not result or not result.get("metadatas"):
            return []
        return [self._resume_from_metadata(metadata) for metadata in result["metadatas"]]

    def query(self, embedding: Sequence[float], top_k: int) -> List[tuple[Resume, float]]:
        query_result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        if not query_result.get("metadatas"):
            return []
        candidates: List[tuple[Resume, float]] = []
        metadatas = query_result["metadatas"][0]
        distances = query_result["distances"][0]
        for metadata, distance in zip(metadatas, distances):
            resume = self._resume_from_metadata(metadata)
            similarity = 1 - distance  # cosine metric returns smaller distance for closer vectors
            candidates.append((resume, similarity))
        return candidates

    @staticmethod
    def build_document(resume: Resume) -> str:
        return "\n".join(
            [
                resume.role,
                ", ".join(resume.skills),
                resume.experience,
                resume.summary,
            ]
        )

    @staticmethod
    def build_query_text(request: RecommendationRequest) -> str:
        return "\n".join(
            [
                request.role,
                ", ".join(request.skills),
                request.summary,
            ]
        )
