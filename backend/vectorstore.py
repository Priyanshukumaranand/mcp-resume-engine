from __future__ import annotations

import json
import os
from typing import List, Sequence, Tuple

import chromadb

try:
    from .models import Resume
except ImportError:
    from models import Resume  # type: ignore


class ResumeVectorStore:
    """Manages all access to the ChromaDB resume collection."""

    def __init__(self, persist_directory: str, collection_name: str = "resumes") -> None:
        os.makedirs(persist_directory, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _metadata_from_resume(self, resume: Resume, chunk_index: int) -> dict:
        metadata = resume.model_dump()
        metadata["skills"] = json.dumps(resume.skills)
        metadata["role"] = resume.role or ""
        metadata["chunk_index"] = chunk_index
        return metadata

    def _resume_from_metadata(self, metadata: dict) -> Resume:
        skills_raw = metadata.get("skills", "[]")
        skills = json.loads(skills_raw) if isinstance(skills_raw, str) else skills_raw
        return Resume(
            id=str(metadata["id"]),
            name=metadata["name"],
            email=metadata["email"],
            role=metadata.get("role") or None,
            skills=skills,
            experience=metadata["experience"],
            summary=metadata["summary"],
        )

    def add_resume_chunks(
        self,
        resume: Resume,
        embeddings: Sequence[Sequence[float]],
        chunks: Sequence[str],
    ) -> None:
        ids: List[str] = []
        metadatas: List[dict] = []
        documents: List[str] = []
        for index, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            ids.append(f"{resume.id}::chunk-{index}")
            metadatas.append(self._metadata_from_resume(resume, index))
            documents.append(chunk)

        if not ids:
            return

        self._collection.upsert(
            ids=ids,
            embeddings=[list(e) for e in embeddings],
            documents=documents,
            metadatas=metadatas,
        )

    def get_all_resumes(self) -> List[Resume]:
        result = self._collection.get(include=["metadatas"], limit=1000)
        if not result or not result.get("metadatas"):
            return []
        seen = {}
        for metadata in result["metadatas"]:
            resume_id = metadata.get("id")
            if resume_id in seen:
                continue
            seen[resume_id] = self._resume_from_metadata(metadata)
        return list(seen.values())

    def has_resumes(self) -> bool:
        try:
            return self._collection.count() > 0
        except AttributeError:
            result = self._collection.get(limit=1)
            return bool(result.get("ids")) if result else False

    def query(self, embedding: Sequence[float], top_k: int) -> List[Tuple[Resume, str, float]]:
        query_result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            include=["metadatas", "distances", "documents"],
        )
        if not query_result.get("metadatas"):
            return []
        candidates: List[Tuple[Resume, str, float]] = []
        metadatas = query_result["metadatas"][0]
        distances = query_result["distances"][0]
        documents = query_result.get("documents", [[]])[0]
        for metadata, distance, document in zip(metadatas, distances, documents):
            resume = self._resume_from_metadata(metadata)
            similarity = 1 - distance
            candidates.append((resume, document, similarity))
        return candidates

    def delete(self, resume_id: str) -> bool:
        if not resume_id:
            return False
        existing = self._collection.get(where={"id": resume_id})
        ids = existing.get("ids") if existing else None
        if not ids:
            return False
        self._collection.delete(where={"id": resume_id})
        return True

    @staticmethod
    def build_document(resume: Resume) -> str:
        parts = [
            resume.role or "",
            ", ".join(resume.skills) if resume.skills else "",
            resume.experience,
            resume.summary,
        ]
        return "\n".join(part for part in parts if part).strip()
