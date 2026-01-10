"""High-quality embeddings for resume search and retrieval.

Uses HuggingFace Inference API for embeddings - lightweight, no local model downloads.
"""
from __future__ import annotations

import os
import re
from typing import Iterable, List, Optional

from langchain_huggingface import HuggingFaceEndpointEmbeddings


class ResumeEmbedder:
    """Resume embedder using HuggingFace Inference API.
    
    Uses MiniLM (all-MiniLM-L6-v2) via HuggingFace API which provides:
    - No large model downloads
    - Fast API-based inference
    - 384 dimensions
    
    Requires HUGGINGFACE_API_TOKEN environment variable.
    
    Attributes:
        model_name: HuggingFace model identifier
        embedding_dimension: Vector dimension size
    """
    
    # Default model - fast and effective for semantic search
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Model dimensions lookup
    MODEL_DIMS = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-small-en-v1.5": 384,
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_instruction_prefix: bool = True,
    ) -> None:
        """Initialize embedder with HuggingFace Inference API.
        
        Args:
            model_name: Model to use for embeddings
            use_instruction_prefix: Kept for API compatibility
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.use_instruction_prefix = use_instruction_prefix
        
        # Get API token
        api_token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
        
        # Initialize with HuggingFace Inference API
        self._model = HuggingFaceEndpointEmbeddings(
            model=self.model_name,
            huggingfacehub_api_token=api_token,
        )
        
        # Get embedding dimension
        self._embedding_dim = self.MODEL_DIMS.get(self.model_name, 384)
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding vector dimension."""
        return self._embedding_dim
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Normalized embedding vector
        """
        if not query:
            return [0.0] * self._embedding_dim
        
        clean_query = self._prepare_text(query)
        
        # Retry logic for API failures
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self._model.embed_query(clean_query)
                if result and len(result) > 0:
                    return result
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    continue
                raise RuntimeError(f"Embedding API failed after {max_retries} attempts: {e}")
        
        return [0.0] * self._embedding_dim
    
    def embed_document(self, text: str) -> List[float]:
        """Embed a document chunk.
        
        Args:
            text: Document text to embed
            
        Returns:
            Normalized embedding vector
        """
        if not text:
            return [0.0] * self._embedding_dim
        
        clean_text = self._prepare_text(text)
        
        # Retry logic for API failures
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self._model.embed_query(clean_text)
                if result and len(result) > 0:
                    return result
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise RuntimeError(f"Embedding API failed after {max_retries} attempts: {e}")
        
        return [0.0] * self._embedding_dim
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Normalized embedding vector
        """
        return self.embed_document(text)
    
    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed multiple texts in batch.
        
        Args:
            texts: Iterable of text strings
            
        Returns:
            List of normalized embedding vectors
        """
        items = [self._prepare_text(t) for t in texts]
        if not items:
            return []
        
        return self._model.embed_documents(items)
    
    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed multiple documents.
        
        Convenience method for embedding multiple document chunks.
        
        Args:
            texts: Document texts to embed
            
        Returns:
            List of normalized embedding vectors
        """
        items = list(texts)
        if not items:
            return []
        
        prepared = [self._prepare_text(t) for t in items]
        return self._model.embed_documents(prepared)
    
    def _prepare_text(self, text: str) -> str:
        """Prepare text for embedding.
        
        Cleans and normalizes text for better embedding quality.
        """
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove very long sequences of special characters
        text = re.sub(r'[^\w\s]{10,}', '', text)
        
        # Truncate if too long (most models have 512 token limit)
        max_chars = 2000
        if len(text) > max_chars:
            text = text[:max_chars]
        
        return text


class QueryExpander:
    """Expands queries for improved retrieval.
    
    Uses simple heuristics to add related terms.
    """
    
    # Common skill synonyms for expansion
    SKILL_SYNONYMS = {
        "python": ["python3", "py"],
        "javascript": ["js", "ecmascript"],
        "typescript": ["ts"],
        "react": ["reactjs", "react.js"],
        "node": ["nodejs", "node.js"],
        "machine learning": ["ml", "deep learning", "ai"],
        "aws": ["amazon web services", "cloud"],
        "docker": ["containers", "containerization"],
        "kubernetes": ["k8s", "container orchestration"],
        "sql": ["database", "rdbms"],
        "api": ["rest", "restful", "web services"],
    }
    
    @classmethod
    def expand_query(cls, query: str) -> str:
        """Expand query with related terms.
        
        Args:
            query: Original search query
            
        Returns:
            Expanded query with synonyms
        """
        query_lower = query.lower()
        expansions = []
        
        for term, synonyms in cls.SKILL_SYNONYMS.items():
            if term in query_lower:
                expansions.extend(synonyms)
        
        if expansions:
            return query + " " + " ".join(expansions)
        
        return query
