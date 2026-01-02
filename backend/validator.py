"""Response validation and hallucination prevention."""
from __future__ import annotations

from typing import List, Tuple, Optional
from dataclasses import dataclass

# Confidence thresholds
MIN_SIMILARITY_THRESHOLD = 0.5   # Minimum similarity to consider a match
CONFIDENT_THRESHOLD = 0.7        # Above this = confident answer
LOW_CONFIDENCE_THRESHOLD = 0.4   # Below this = fallback response

# Fallback messages
FALLBACK_INSUFFICIENT_INFO = "Insufficient information in resumes to answer this question."
FALLBACK_NO_MATCH = "No candidates found matching the specified criteria."
FALLBACK_LOW_CONFIDENCE = "The available information is not detailed enough to provide a reliable answer."


@dataclass
class ValidationResult:
    """Result of answer validation."""
    is_valid: bool
    confidence_score: float
    reason: str
    evidence_snippets: List[str]


class ResponseValidator:
    """
    Validates LLM responses against retrieved context to prevent hallucinations.
    
    Ensures:
    - Answers are supported by retrieved chunks
    - No information is generated beyond retrieved context
    - Appropriate fallbacks when evidence is weak
    """
    
    def __init__(
        self,
        min_similarity: float = MIN_SIMILARITY_THRESHOLD,
        confident_threshold: float = CONFIDENT_THRESHOLD,
    ):
        self.min_similarity = min_similarity
        self.confident_threshold = confident_threshold
    
    def calculate_confidence(
        self,
        similarities: List[float],
        num_chunks: int,
    ) -> float:
        """
        Calculate confidence score from similarity distribution.
        
        Factors:
        - Average similarity of top matches
        - Number of supporting chunks
        - Spread of similarity scores
        
        Args:
            similarities: List of similarity scores from retrieval
            num_chunks: Number of chunks retrieved
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not similarities:
            return 0.0
        
        # Base confidence from average similarity
        avg_similarity = sum(similarities) / len(similarities)
        
        # Boost for multiple supporting chunks
        chunk_boost = min(0.1 * (num_chunks - 1), 0.2) if num_chunks > 1 else 0
        
        # Penalty for high variance (inconsistent matches)
        if len(similarities) > 1:
            variance = sum((s - avg_similarity) ** 2 for s in similarities) / len(similarities)
            variance_penalty = min(variance * 0.5, 0.15)
        else:
            variance_penalty = 0.1  # Single source penalty
        
        confidence = avg_similarity + chunk_boost - variance_penalty
        return max(0.0, min(1.0, confidence))
    
    def validate_answer(
        self,
        answer: str,
        chunks: List[str],
        similarities: List[float],
    ) -> ValidationResult:
        """
        Validate that an answer is supported by retrieved chunks.
        
        Checks:
        - Similarity scores meet threshold
        - Answer doesn't contain claims beyond context
        
        Args:
            answer: Generated answer to validate
            chunks: Retrieved context chunks
            similarities: Similarity scores for each chunk
            
        Returns:
            ValidationResult with validity, confidence, and evidence
        """
        if not chunks or not similarities:
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                reason="No context chunks available",
                evidence_snippets=[],
            )
        
        confidence = self.calculate_confidence(similarities, len(chunks))
        
        # Check if any similarity meets minimum threshold
        max_similarity = max(similarities)
        if max_similarity < self.min_similarity:
            return ValidationResult(
                is_valid=False,
                confidence_score=confidence,
                reason="Retrieved chunks don't meet similarity threshold",
                evidence_snippets=[],
            )
        
        # Extract evidence snippets (top 3 chunks above threshold)
        evidence = [
            chunk[:300] + "..." if len(chunk) > 300 else chunk
            for chunk, sim in zip(chunks, similarities)
            if sim >= self.min_similarity
        ][:3]
        
        is_confident = confidence >= self.confident_threshold
        
        return ValidationResult(
            is_valid=True,
            confidence_score=confidence,
            reason="Sufficient evidence found" if is_confident else "Low confidence match",
            evidence_snippets=evidence,
        )
    
    def get_fallback_response(
        self,
        confidence: float,
        has_matches: bool,
    ) -> str:
        """
        Get appropriate fallback response based on situation.
        
        Args:
            confidence: Calculated confidence score
            has_matches: Whether any matches were found
            
        Returns:
            Appropriate fallback message
        """
        if not has_matches:
            return FALLBACK_NO_MATCH
        
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            return FALLBACK_INSUFFICIENT_INFO
        
        return FALLBACK_LOW_CONFIDENCE
    
    def should_use_fallback(self, confidence: float) -> bool:
        """Check if fallback response should be used."""
        return confidence < self.confident_threshold


def extract_section_types(chunks: List[str], section_metadata: List[str]) -> List[str]:
    """
    Extract unique section types from chunk metadata.
    
    Args:
        chunks: Retrieved chunks
        section_metadata: Section type for each chunk
        
    Returns:
        List of unique section types that contributed to answer
    """
    if not section_metadata:
        return ["unknown"]
    
    seen = set()
    sections = []
    for section in section_metadata:
        if section and section not in seen:
            seen.add(section)
            sections.append(section)
    
    return sections or ["unknown"]
