"""Vector store for resume chunks using ChromaDB.

Supports section-aware storage and weighted retrieval.
"""
from __future__ import annotations

import json
import os
from typing import List, Sequence, Tuple, Optional, Any

import chromadb

# Import section weights
try:
    from ..core.section_detector import SECTION_WEIGHTS, SECTION_TYPES
except ImportError:
    try:
        from backend.core.section_detector import SECTION_WEIGHTS, SECTION_TYPES
    except ImportError:
        SECTION_WEIGHTS = {
            "skills": 1.3, "projects": 1.2, "experience": 1.15,
            "achievements": 1.1, "education": 1.0, "header": 0.9, "unknown": 1.0
        }
        SECTION_TYPES = frozenset(SECTION_WEIGHTS.keys())


# Confidence thresholds for retrieval
MIN_SIMILARITY_THRESHOLD = 0.35
CONFIDENT_SIMILARITY = 0.65


class ResumeVectorStore:
    """ChromaDB-based vector store for resume chunks.
    
    Features:
    - Section metadata stored per chunk
    - Weighted retrieval based on section type
    - Parent document context preservation
    - Skill-based filtering
    
    Attributes:
        persist_directory: Path to ChromaDB storage
        collection_name: Name of the collection
    """
    
    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "resumes_v2",
    ) -> None:
        """Initialize vector store.
        
        Args:
            persist_directory: Path for persistent storage
            collection_name: Name of ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def add_resume_chunks(
        self,
        resume: Any,  # Resume model
        embeddings: Sequence[Sequence[float]],
        chunks: Sequence[str],
        section_types: Optional[Sequence[str]] = None,
    ) -> None:
        """Add resume chunks with embeddings to the store.
        
        Args:
            resume: Resume model with metadata
            embeddings: Embedding vectors for each chunk
            chunks: Text chunks
            section_types: Section type for each chunk
        """
        if not chunks:
            return
        
        # Default section types if not provided
        if section_types is None:
            section_types = ["unknown"] * len(chunks)
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embedding_list = []
        
        for i, (chunk, embedding, section_type) in enumerate(
            zip(chunks, embeddings, section_types)
        ):
            chunk_id = f"{resume.id}__chunk_{i}"
            
            # Build metadata
            metadata = self._build_metadata(resume, i, section_type)
            
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(metadata)
            embedding_list.append(list(embedding))
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embedding_list,
            documents=documents,
            metadatas=metadatas,
        )
    
    def _build_metadata(
        self,
        resume: Any,
        chunk_index: int,
        section_type: str,
    ) -> dict:
        """Build metadata dict for a chunk."""
        # Normalize section type
        if section_type not in SECTION_TYPES:
            section_type = "unknown"
        
        # Build skills text for keyword matching
        skills_text = ", ".join(resume.skills[:15]) if resume.skills else ""
        
        return {
            "resume_id": resume.id,
            "anon_id": resume.anon_id or "",
            "name": resume.name or "Unknown",
            "email": resume.email or "",
            "role": resume.role or "",
            "skills_text": skills_text,
            "skills_json": json.dumps(resume.skills[:15] if resume.skills else []),
            "projects_json": json.dumps(resume.projects[:5] if resume.projects else []),
            "education": resume.education or "",
            "experience": (resume.experience or "")[:500],
            "summary": (resume.summary or "")[:500],
            "chunk_index": chunk_index,
            "section_type": section_type,
            "section_weight": SECTION_WEIGHTS.get(section_type, 1.0),
        }
    
    def query(
        self,
        embedding: Sequence[float],
        top_k: int = 10,
        apply_section_weights: bool = True,
        min_similarity: float = MIN_SIMILARITY_THRESHOLD,
    ) -> List[Tuple[Any, str, float, str]]:
        """Query for similar resume chunks.
        
        Args:
            embedding: Query embedding vector
            top_k: Number of results to return
            apply_section_weights: Whether to apply section-based weighting
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (Resume, chunk_text, similarity, section_type) tuples
        """
        # Fetch more candidates for re-ranking
        n_candidates = min(top_k * 5, 50)
        
        results = self.collection.query(
            query_embeddings=[list(embedding)],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )
        
        if not results or not results.get("ids") or not results["ids"][0]:
            return []
        
        # Process results
        processed = []
        
        for i, chunk_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            document = results["documents"][0][i]
            distance = results["distances"][0][i]
            
            # Convert distance to similarity (ChromaDB uses L2 distance for cosine)
            similarity = 1 - distance
            
            # Apply section weight if requested
            section_type = metadata.get("section_type", "unknown")
            if apply_section_weights:
                weight = SECTION_WEIGHTS.get(section_type, 1.0)
                weighted_similarity = similarity * weight
            else:
                weighted_similarity = similarity
            
            # Filter by minimum similarity
            if similarity < min_similarity:
                continue
            
            # Reconstruct Resume from metadata
            resume = self._resume_from_metadata(metadata)
            
            processed.append((resume, document, weighted_similarity, section_type))
        
        # Sort by weighted similarity and return top_k
        processed.sort(key=lambda x: x[2], reverse=True)
        
        # Deduplicate by resume ID (keep best chunk per resume)
        seen_resumes = set()
        deduplicated = []
        
        for item in processed:
            resume_id = item[0].id
            if resume_id not in seen_resumes:
                seen_resumes.add(resume_id)
                deduplicated.append(item)
            
            if len(deduplicated) >= top_k:
                break
        
        return deduplicated
    
    def _resume_from_metadata(self, metadata: dict) -> Any:
        """Reconstruct Resume model from chunk metadata."""
        # Import here to avoid circular dependency
        try:
            from ..models import Resume
        except ImportError:
            from backend.models import Resume
        
        # Parse JSON fields
        skills = json.loads(metadata.get("skills_json", "[]"))
        projects = json.loads(metadata.get("projects_json", "[]"))
        
        return Resume(
            id=metadata.get("resume_id", ""),
            anon_id=metadata.get("anon_id", ""),
            name=metadata.get("name", "Unknown"),
            email=metadata.get("email", ""),
            role=metadata.get("role"),
            skills=skills,
            projects=projects,
            education=metadata.get("education"),
            experience=metadata.get("experience", ""),
            summary=metadata.get("summary", ""),
        )
    
    def query_by_skills(
        self,
        skills: List[str],
        embedding: Sequence[float],
        top_k: int = 10,
    ) -> List[Tuple[Any, List[str], float]]:
        """Query for resumes matching specific skills.
        
        Uses both semantic search and skill keyword matching.
        
        Returns:
            List of (Resume, matched_skills, similarity) tuples
        """
        # Get candidates from semantic search
        candidates = self.query(
            embedding,
            top_k=top_k * 3,
            apply_section_weights=True,
        )
        
        # Filter and score by skill matches
        skills_lower = {s.lower() for s in skills}
        results = []
        
        for resume, chunk, similarity, section_type in candidates:
            resume_skills_lower = {s.lower() for s in resume.skills}
            matched = skills_lower.intersection(resume_skills_lower)
            
            if matched:
                # Boost similarity based on skill match ratio
                match_ratio = len(matched) / len(skills)
                boosted_similarity = similarity * (1 + match_ratio * 0.5)
                
                # Return original case skills
                original_case_matched = [
                    s for s in resume.skills
                    if s.lower() in matched
                ]
                
                results.append((resume, original_case_matched, boosted_similarity))
        
        # Sort and return top results
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
    
    def get_all_resumes(self) -> List[Any]:
        """Get all unique resumes (deduplicated by ID)."""
        results = self.collection.get(include=["metadatas"])
        
        if not results or not results.get("ids"):
            return []
        
        seen_ids = set()
        resumes = []
        
        for metadata in results["metadatas"]:
            resume_id = metadata.get("resume_id")
            if resume_id and resume_id not in seen_ids:
                seen_ids.add(resume_id)
                resumes.append(self._resume_from_metadata(metadata))
        
        return resumes
    
    def has_resumes(self) -> bool:
        """Check if store contains any resumes."""
        try:
            count = self.collection.count()
            return count > 0
        except Exception:
            return False
    
    def delete(self, resume_id: str) -> None:
        """Delete all chunks for a resume."""
        try:
            # Find all chunk IDs for this resume
            results = self.collection.get(
                where={"resume_id": resume_id},
                include=["metadatas"],
            )
            
            if results and results.get("ids"):
                self.collection.delete(ids=results["ids"])
        except Exception:
            pass
    
    @staticmethod
    def build_document(resume: Any) -> str:
        """Build a searchable document from resume fields."""
        parts = [
            f"Name: {resume.name}",
            f"Role: {resume.role}" if resume.role else "",
            f"Skills: {', '.join(resume.skills)}" if resume.skills else "",
            f"Projects: {', '.join(resume.projects)}" if resume.projects else "",
            f"Education: {resume.education}" if resume.education else "",
            resume.experience or "",
            resume.summary or "",
        ]
        return "\n".join(p for p in parts if p).strip()
