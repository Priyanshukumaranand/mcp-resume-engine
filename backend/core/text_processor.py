"""Text preprocessing utilities for resume processing.

Handles text normalization, encoding fixes, and PDF artifact cleanup.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional


def normalize_text(text: str) -> str:
    """Normalize unicode and whitespace in text.
    
    Args:
        text: Raw input text
        
    Returns:
        Normalized text with consistent whitespace and encoding
    """
    if not text:
        return ""
    
    # Normalize unicode (NFKC normalizes compatibility characters)
    text = unicodedata.normalize("NFKC", text)
    
    # Replace various dash types with standard hyphen
    text = re.sub(r"[–—−]", "-", text)
    
    # Replace various quote types with standard quotes
    text = re.sub(r"[""„‟]", '"', text)
    text = re.sub(r"[''‚‛]", "'", text)
    
    # Normalize bullet points to standard format
    text = re.sub(r"[•◦▪▸►●○]", "•", text)
    
    # Replace multiple spaces with single space
    text = re.sub(r"[ \t]+", " ", text)
    
    # Normalize line endings and reduce multiple newlines
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


def clean_pdf_artifacts(text: str) -> str:
    """Remove common PDF extraction artifacts.
    
    Args:
        text: Text extracted from PDF
        
    Returns:
        Cleaned text without common artifacts
    """
    if not text:
        return ""
    
    # Remove page numbers (common patterns)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"\n\s*Page\s+\d+\s*(?:of\s+\d+)?\s*\n", "\n", text, flags=re.IGNORECASE)
    
    # Remove common header/footer artifacts
    text = re.sub(r"\n\s*(?:Confidential|Private)\s*\n", "\n", text, flags=re.IGNORECASE)
    
    # Remove excessive whitespace around section headers
    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
    
    # Remove form feed characters
    text = text.replace("\f", "\n")
    
    return text.strip()


def preprocess_resume_text(text: str, preserve_structure: bool = True) -> str:
    """Full preprocessing pipeline for resume text.
    
    Args:
        text: Raw resume text from PDF extraction
        preserve_structure: If True, maintain paragraph structure
        
    Returns:
        Clean, normalized text ready for processing
    """
    if not text:
        return ""
    
    # Step 1: Clean PDF artifacts
    text = clean_pdf_artifacts(text)
    
    # Step 2: Normalize unicode and whitespace 
    text = normalize_text(text)
    
    # Step 3: Fix common OCR/extraction issues
    # Replace broken hyphenation (word-\nword -> word)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    
    # Step 4: Ensure consistent section header formatting
    # Add blank line before common section headers if missing
    section_headers = [
        "education", "experience", "skills", "projects", 
        "work history", "employment", "qualifications",
        "achievements", "awards", "certifications", "summary"
    ]
    for header in section_headers:
        # Match header at start of line, optionally with colons
        pattern = rf"(?<!\n)\n({header}s?)\s*:?"
        text = re.sub(pattern, rf"\n\n\1", text, flags=re.IGNORECASE)
    
    if not preserve_structure:
        # Collapse all whitespace to single spaces
        text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def extract_sentences(text: str) -> list[str]:
    """Split text into sentences using simple heuristics.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    # Split on sentence-ending punctuation followed by space and capital
    # This is a simple heuristic - could use nltk.sent_tokenize for better results
    sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
    sentences = re.split(sentence_pattern, text)
    
    # Clean up each sentence
    return [s.strip() for s in sentences if s.strip()]


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs.
    
    Args:
        text: Input text
        
    Returns:
        List of non-empty paragraphs
    """
    if not text:
        return []
    
    # Split on double newlines (paragraph breaks)
    paragraphs = re.split(r"\n\s*\n", text)
    
    # Clean and filter empty paragraphs
    return [p.strip() for p in paragraphs if p.strip()]
