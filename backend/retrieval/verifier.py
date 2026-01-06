"""Response verification and confidence scoring.

Validates retrieved results and calculates confidence scores.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any


# Confidence thresholds
HIGH_CONFIDENCE = 0.75
MEDIUM_CONFIDENCE = 0.5
LOW_CONFIDENCE = 0.35

# Fallback response for insufficient information
FALLBACK_INSUFFICIENT_INFO = "I don't have enough information in the resumes to answer this question."


@dataclass
class VerificationResult:
    """Result of response verification."""
    is_valid: bool
    confidence: float
    reason: str
    evidence_snippets: List[str]


class ResponseVerifier:
    """Verifies and scores QA responses.
    
    Provides:
    - Confidence scoring based on retrieval quality
    - Evidence extraction from chunks
    - Fallback decision logic
    """
    
    def __init__(
        self,
        confidence_threshold: float = LOW_CONFIDENCE,
    ):
        """Initialize verifier.
        
        Args:
            confidence_threshold: Minimum confidence for valid response
        """
        self.confidence_threshold = confidence_threshold
    
    def calculate_confidence(
        self,
        similarities: List[float],
        match_count: int,
        section_types: Optional[List[str]] = None,
    ) -> float:
        """Calculate confidence score from retrieval signals.
        
        Args:
            similarities: List of similarity scores
            match_count: Number of matching candidates
            section_types: Section types of matches (for weighting)
            
        Returns:
            Confidence score between 0 and 1
        """
        if not similarities or match_count == 0:
            return 0.0
        
        # Base confidence from similarity scores
        avg_similarity = sum(similarities) / len(similarities)
        max_similarity = max(similarities)
        
        # Weight towards best match
        base_confidence = 0.7 * max_similarity + 0.3 * avg_similarity
        
        # Boost for multiple matches
        match_boost = min(0.15, 0.05 * (match_count - 1))
        
        # Section type bonus (skills/experience sections are more reliable)
        section_bonus = 0.0
        if section_types:
            high_value_sections = {"skills", "experience", "projects"}
            high_value_count = sum(1 for s in section_types if s in high_value_sections)
            section_bonus = min(0.1, 0.03 * high_value_count)
        
        confidence = base_confidence + match_boost + section_bonus
        
        return min(1.0, max(0.0, confidence))
    
    def should_use_fallback(self, confidence: float) -> bool:
        """Determine if fallback response should be used.
        
        Args:
            confidence: Calculated confidence score
            
        Returns:
            True if fallback should be used
        """
        return confidence < self.confidence_threshold
    
    def get_fallback_response(self) -> str:
        """Get the fallback response for low confidence."""
        return FALLBACK_INSUFFICIENT_INFO
    
    def verify_answer(
        self,
        question: str,
        answer: str,
        chunks: List[str],
        similarities: List[float],
    ) -> VerificationResult:
        """Verify an answer against source chunks.
        
        Args:
            question: Original question
            answer: Generated answer
            chunks: Source chunks used for generation
            similarities: Similarity scores for chunks
            
        Returns:
            VerificationResult with confidence and evidence
        """
        # Extract key terms from question
        question_terms = self._extract_key_terms(question)
        
        # Check if answer addresses the question
        answer_terms = self._extract_key_terms(answer)
        term_overlap = len(question_terms.intersection(answer_terms))
        
        # Calculate base confidence
        confidence = self.calculate_confidence(
            similarities,
            len(chunks),
        )
        
        # Adjust confidence based on term overlap
        if term_overlap == 0 and question_terms:
            confidence *= 0.7  # Reduce if answer doesn't use question terms
        
        # Extract evidence snippets
        evidence = self._extract_evidence(question, chunks)
        
        # Determine validity
        is_valid = confidence >= self.confidence_threshold and len(evidence) > 0
        
        # Build reason
        if is_valid:
            reason = f"Answer supported by {len(evidence)} evidence snippets"
        elif confidence < self.confidence_threshold:
            reason = "Low confidence in retrieved information"
        else:
            reason = "Unable to find supporting evidence in resumes"
        
        return VerificationResult(
            is_valid=is_valid,
            confidence=confidence,
            reason=reason,
            evidence_snippets=evidence[:3],  # Top 3 evidence snippets
        )
    
    def _extract_key_terms(self, text: str) -> set:
        """Extract key terms from text."""
        stop_words = {
            "who", "what", "which", "how", "can", "does", "has", "have",
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "and", "or", "but", "with", "for", "to", "from", "in", "on",
            "i", "you", "we", "they", "it", "this", "that",
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        return {w for w in words if w not in stop_words and len(w) > 2}
    
    def _extract_evidence(
        self,
        question: str,
        chunks: List[str],
        max_snippets: int = 5,
    ) -> List[str]:
        """Extract relevant evidence snippets from chunks."""
        question_terms = self._extract_key_terms(question)
        evidence = []
        
        for chunk in chunks:
            # Find sentences containing question terms
            sentences = re.split(r'[.!?\n]+', chunk)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence or len(sentence) < 20:
                    continue
                
                sentence_lower = sentence.lower()
                matching_terms = sum(
                    1 for term in question_terms
                    if term in sentence_lower
                )
                
                if matching_terms > 0:
                    # Truncate long sentences
                    if len(sentence) > 200:
                        sentence = sentence[:200] + "..."
                    evidence.append(sentence)
                    
                    if len(evidence) >= max_snippets:
                        break
            
            if len(evidence) >= max_snippets:
                break
        
        return evidence


class ConfidenceLevel:
    """Confidence level constants."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"
    
    @classmethod
    def from_score(cls, score: float) -> str:
        """Get confidence level from numeric score."""
        if score >= HIGH_CONFIDENCE:
            return cls.HIGH
        elif score >= MEDIUM_CONFIDENCE:
            return cls.MEDIUM
        elif score >= LOW_CONFIDENCE:
            return cls.LOW
        else:
            return cls.NONE
