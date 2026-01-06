"""Semantic-aware text chunking for resume sections.

Creates coherent chunks that preserve sentence boundaries
and semantic groupings within sections.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .section_detector import SectionDetector, Section, SECTION_WEIGHTS
from .text_processor import preprocess_resume_text, split_into_paragraphs


@dataclass
class SectionChunk:
    """A text chunk with section metadata."""
    text: str
    section_type: str = "unknown"
    chunk_index: int = 0
    parent_section_index: int = 0
    
    # Metadata for retrieval
    word_count: int = field(default=0)
    has_bullet_points: bool = field(default=False)
    
    def __post_init__(self):
        if self.word_count == 0:
            self.word_count = len(self.text.split())
        if not self.has_bullet_points:
            self.has_bullet_points = bool(re.search(r'[•\-\*]\s', self.text))


class SemanticChunker:
    """Semantic-aware resume chunker.
    
    Creates chunks that:
    1. Respect section boundaries
    2. Preserve sentence integrity
    3. Group semantically related content
    4. Maintain context with overlap
    """
    
    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
    ):
        """Initialize chunker with size parameters.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks for context
            min_chunk_size: Minimum chunk size (avoid tiny chunks)
            max_chunk_size: Maximum chunk size (hard limit)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.section_detector = SectionDetector()
    
    def chunk_resume(
        self,
        text: str,
        preprocess: bool = True,
    ) -> Tuple[List[SectionChunk], List[Section]]:
        """Chunk a resume into semantic sections.
        
        Args:
            text: Raw resume text
            preprocess: Whether to preprocess text first
            
        Returns:
            Tuple of (chunks, sections)
        """
        if not text:
            return [], []
        
        # Preprocess if requested
        if preprocess:
            text = preprocess_resume_text(text)
        
        # Detect sections
        sections = self.section_detector.detect_sections(text)
        
        # Chunk each section
        all_chunks: List[SectionChunk] = []
        global_chunk_idx = 0
        
        for section_idx, section in enumerate(sections):
            section_chunks = self._chunk_section(section, global_chunk_idx, section_idx)
            all_chunks.extend(section_chunks)
            global_chunk_idx += len(section_chunks)
        
        return all_chunks, sections
    
    def _chunk_section(
        self,
        section: Section,
        start_index: int,
        section_index: int,
    ) -> List[SectionChunk]:
        """Chunk a single section with sentence awareness.
        
        Args:
            section: Section to chunk
            start_index: Starting chunk index
            section_index: Index of parent section
            
        Returns:
            List of SectionChunks
        """
        text = section.text
        if not text:
            return []
        
        # For short sections, return as single chunk
        if len(text) <= self.max_chunk_size:
            return [SectionChunk(
                text=text.strip(),
                section_type=section.section_type,
                chunk_index=start_index,
                parent_section_index=section_index,
            )]
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        # Group sentences into chunks
        chunks = self._group_sentences(
            sentences,
            section.section_type,
            start_index,
            section_index,
        )
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving structure.
        
        Handles:
        - Standard sentence endings (. ! ?)
        - Bullet points as separate items
        - Newline-separated items
        """
        sentences = []
        
        # Split by bullet points first
        bullet_pattern = r'(?:^|\n)\s*[•\-\*]\s*'
        bullet_parts = re.split(bullet_pattern, text)
        
        for part in bullet_parts:
            if not part.strip():
                continue
            
            # Split each part by sentence endings
            sentence_parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', part)
            
            for sentence in sentence_parts:
                cleaned = sentence.strip()
                if cleaned:
                    sentences.append(cleaned)
        
        # If no sentences found, split by newlines
        if not sentences:
            sentences = [s.strip() for s in text.split('\n') if s.strip()]
        
        return sentences
    
    def _group_sentences(
        self,
        sentences: List[str],
        section_type: str,
        start_index: int,
        section_index: int,
    ) -> List[SectionChunk]:
        """Group sentences into appropriately sized chunks."""
        chunks = []
        current_chunk_text = []
        current_length = 0
        chunk_idx = start_index
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds max, flush current chunk
            if current_length + sentence_length > self.max_chunk_size and current_chunk_text:
                chunk_text = self._join_chunk_sentences(current_chunk_text)
                chunks.append(SectionChunk(
                    text=chunk_text,
                    section_type=section_type,
                    chunk_index=chunk_idx,
                    parent_section_index=section_index,
                ))
                chunk_idx += 1
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk_text)
                current_chunk_text = overlap_sentences
                current_length = sum(len(s) for s in overlap_sentences)
            
            # Add sentence to current chunk
            current_chunk_text.append(sentence)
            current_length += sentence_length
            
            # If we've reached target size, consider flushing
            if current_length >= self.chunk_size:
                chunk_text = self._join_chunk_sentences(current_chunk_text)
                chunks.append(SectionChunk(
                    text=chunk_text,
                    section_type=section_type,
                    chunk_index=chunk_idx,
                    parent_section_index=section_index,
                ))
                chunk_idx += 1
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk_text)
                current_chunk_text = overlap_sentences
                current_length = sum(len(s) for s in overlap_sentences)
        
        # Flush remaining content
        if current_chunk_text:
            chunk_text = self._join_chunk_sentences(current_chunk_text)
            
            # If too small, try to merge with previous chunk
            if len(chunk_text) < self.min_chunk_size and chunks:
                last_chunk = chunks[-1]
                merged_text = last_chunk.text + "\n" + chunk_text
                
                if len(merged_text) <= self.max_chunk_size:
                    last_chunk.text = merged_text
                    last_chunk.word_count = len(merged_text.split())
                else:
                    chunks.append(SectionChunk(
                        text=chunk_text,
                        section_type=section_type,
                        chunk_index=chunk_idx,
                        parent_section_index=section_index,
                    ))
            else:
                chunks.append(SectionChunk(
                    text=chunk_text,
                    section_type=section_type,
                    chunk_index=chunk_idx,
                    parent_section_index=section_index,
                ))
        
        return chunks
    
    def _join_chunk_sentences(self, sentences: List[str]) -> str:
        """Join sentences into a chunk, preserving formatting."""
        # Check if sentences are bullet points
        if all(len(s) < 200 for s in sentences):
            # Short items - likely bullet points, use newlines
            return "\n• ".join(sentences)
        else:
            # Longer passages - use space
            return " ".join(sentences)
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap into next chunk."""
        if not sentences:
            return []
        
        # Take last sentence(s) up to overlap size
        overlap_text = ""
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            if len(overlap_text) + len(sentence) <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_text = sentence + " " + overlap_text
            else:
                break
        
        return overlap_sentences


def get_section_weight(section_type: str) -> float:
    """Get retrieval weight for a section type.
    
    Higher weights boost relevance for that section type.
    """
    return SECTION_WEIGHTS.get(section_type, 1.0)
