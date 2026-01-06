"""Reranker for improving retrieval quality.

Uses similarity scoring and skill matching to rerank candidates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

try:
    from ..core.section_detector import SECTION_WEIGHTS
except ImportError:
    SECTION_WEIGHTS = {
        "skills": 1.3, "projects": 1.2, "experience": 1.15,
        "achievements": 1.1, "education": 1.0, "header": 0.9, "unknown": 1.0
    }


@dataclass
class RankedResult:
    """A reranked search result."""
    resume: Any
    chunk: str
    original_score: float
    reranked_score: float
    section_type: str
    matched_terms: List[str]
    
    def __post_init__(self):
        if not self.matched_terms:
            self.matched_terms = []


class Reranker:
    """Reranks search results using multiple signals.
    
    Combines:
    - Original similarity score
    - Section type weight
    - Query term matching
    - Skill overlap
    """
    
    def __init__(
        self,
        term_weight: float = 0.2,
        section_weight: float = 0.15,
        skill_weight: float = 0.25,
    ):
        """Initialize reranker with weight parameters.
        
        Args:
            term_weight: Weight for query term matches
            section_weight: Weight for section type boost
            skill_weight: Weight for skill overlap
        """
        self.term_weight = term_weight
        self.section_weight = section_weight
        self.skill_weight = skill_weight
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Any, str, float, str]],
        query_skills: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[RankedResult]:
        """Rerank candidates based on multiple signals.
        
        Args:
            query: Original search query
            candidates: List of (resume, chunk, similarity, section_type)
            query_skills: Skills extracted from query
            top_k: Number of results to return
            
        Returns:
            List of RankedResult objects
        """
        if not candidates:
            return []
        
        # Extract query terms
        query_terms = self._extract_terms(query)
        query_skills = query_skills or []
        query_skills_lower = {s.lower() for s in query_skills}
        
        results = []
        
        for resume, chunk, similarity, section_type in candidates:
            # Calculate term match score
            chunk_lower = chunk.lower()
            matched_terms = [
                term for term in query_terms
                if term.lower() in chunk_lower
            ]
            term_score = len(matched_terms) / max(len(query_terms), 1)
            
            # Calculate section weight
            section_score = SECTION_WEIGHTS.get(section_type, 1.0) - 1.0
            
            # Calculate skill overlap
            resume_skills_lower = {s.lower() for s in resume.skills}
            skill_overlap = len(query_skills_lower.intersection(resume_skills_lower))
            skill_score = skill_overlap / max(len(query_skills), 1) if query_skills else 0
            
            # Combine scores
            reranked_score = (
                similarity +
                term_score * self.term_weight +
                section_score * self.section_weight +
                skill_score * self.skill_weight
            )
            
            results.append(RankedResult(
                resume=resume,
                chunk=chunk,
                original_score=similarity,
                reranked_score=reranked_score,
                section_type=section_type,
                matched_terms=matched_terms,
            ))
        
        # Sort by reranked score
        results.sort(key=lambda x: x.reranked_score, reverse=True)
        
        return results[:top_k]
    
    def _extract_terms(self, query: str) -> List[str]:
        """Extract significant terms from query."""
        # Remove stop words
        stop_words = {
            "who", "what", "which", "how", "can", "does", "has", "have",
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "do", "did", "doing", "and", "or", "but", "not",
            "with", "for", "to", "from", "in", "on", "at", "by", "of",
            "about", "any", "all", "some", "most", "other", "than", "that",
            "this", "these", "those", "i", "you", "we", "they", "it",
            "me", "know", "find", "list", "show", "tell",
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter stop words and short words
        terms = [w for w in words if w not in stop_words and len(w) > 2]
        
        return terms
    
    def extract_skills_from_query(self, query: str) -> List[str]:
        """Extract skill names from a query.
        
        Uses a simple heuristic to identify technology/skill names.
        """
        # Common skills to look for
        common_skills = {
            "python", "java", "javascript", "typescript", "react", "angular",
            "vue", "node", "nodejs", "express", "django", "flask", "fastapi",
            "spring", "sql", "mysql", "postgresql", "mongodb", "redis",
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
            "git", "linux", "machine learning", "deep learning", "tensorflow",
            "pytorch", "scikit", "pandas", "numpy", "spark", "hadoop",
            "kafka", "rabbitmq", "graphql", "rest", "api", "microservices",
            "html", "css", "sass", "webpack", "jest", "cypress", "selenium",
            "agile", "scrum", "jira", "confluence", "ci/cd", "devops",
        }
        
        query_lower = query.lower()
        found_skills = []
        
        for skill in common_skills:
            if skill in query_lower:
                found_skills.append(skill)
        
        return found_skills
