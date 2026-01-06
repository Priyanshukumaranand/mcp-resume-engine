"""High-quality embeddings for resume search and retrieval.

Uses BAAI/bge-base-en-v1.5 for improved semantic understanding.
Supports instruction-based embedding for query vs document differentiation.
"""
from __future__ import annotations

import math
import os
import re
from typing import Iterable, List, Optional

from huggingface_hub import InferenceClient


class ResumeEmbedder:
    """Resume embedder with instruction-based embedding support.
    
    Uses BGE (BAAI General Embedding) model which supports:
    - Asymmetric retrieval (query vs document differentiation)
    - Better performance on MTEB benchmarks
    - 768 dimensions for richer semantic representation
    
    Attributes:
        model_name: HuggingFace model identifier
        query_prefix: Instruction prefix for query embeddings
        document_prefix: Instruction prefix for document embeddings
    """
    
    # Default model with excellent retrieval performance
    DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
    
    # Instruction prefixes for asymmetric retrieval
    QUERY_PREFIX = "Represent this sentence for retrieving relevant resume passages: "
    DOCUMENT_PREFIX = ""  # BGE documents don't need prefix
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_instruction_prefix: bool = True,
    ) -> None:
        """Initialize embedder with HuggingFace Inference API.
        
        Args:
            model_name: Model to use for embeddings
            use_instruction_prefix: Whether to use instruction prefix for queries
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.use_instruction_prefix = use_instruction_prefix
        
        token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not token:
            raise RuntimeError("HUGGINGFACE_API_TOKEN is required for embeddings")
        
        self.client = InferenceClient(token=token)
        
        # Determine embedding dimension based on model
        self._embedding_dim = self._get_embedding_dimension()
    
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension for the model."""
        model_dims = {
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            "BAAI/bge-small-en-v1.5": 384,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
        }
        return model_dims.get(self.model_name, 768)
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding vector dimension."""
        return self._embedding_dim
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a search query with instruction prefix.
        
        For asymmetric retrieval, queries should be prefixed
        to improve retrieval accuracy.
        
        Args:
            query: Search query text
            
        Returns:
            Normalized embedding vector
        """
        if not query:
            return [0.0] * self._embedding_dim
        
        # Clean and prepare query
        clean_query = self._prepare_text(query)
        
        # Add instruction prefix for query
        if self.use_instruction_prefix:
            clean_query = self.QUERY_PREFIX + clean_query
        
        return self.embed_text(clean_query)
    
    def embed_document(self, text: str) -> List[float]:
        """Embed a document chunk.
        
        Documents don't need instruction prefix for BGE models.
        
        Args:
            text: Document text to embed
            
        Returns:
            Normalized embedding vector
        """
        if not text:
            return [0.0] * self._embedding_dim
        
        # Clean and prepare document text
        clean_text = self._prepare_text(text)
        
        # Add document prefix if using instruction-based model
        if self.use_instruction_prefix and self.DOCUMENT_PREFIX:
            clean_text = self.DOCUMENT_PREFIX + clean_text
        
        return self.embed_text(clean_text)
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Normalized embedding vector
        """
        if not text:
            return [0.0] * self._embedding_dim
        
        return self.embed_many([text])[0]
    
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
        
        try:
            result = self.client.feature_extraction(
                items,
                model=self.model_name,
            )
        except Exception as exc:
            raise RuntimeError(f"Embedding API failed: {exc}") from exc
        
        if result is None:
            raise RuntimeError("Embedding API returned no data")
        
        # Convert to list of lists
        vectors = result.tolist() if hasattr(result, "tolist") else result
        if not isinstance(vectors, list) or not vectors:
            raise RuntimeError("Unexpected embedding response shape")
        
        # Normalize all vectors
        return [self._normalize(vec) for vec in vectors]
    
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
        
        # Prepare each document
        prepared = [self._prepare_text(t) for t in items]
        
        # Add document prefix if needed
        if self.use_instruction_prefix and self.DOCUMENT_PREFIX:
            prepared = [self.DOCUMENT_PREFIX + t for t in prepared]
        
        return self.embed_many(prepared)
    
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
        # Approximate with 4 chars per token
        max_chars = 2000
        if len(text) > max_chars:
            text = text[:max_chars]
        
        return text
    
    @staticmethod
    def _normalize(vector: List[float]) -> List[float]:
        """L2 normalize a vector.
        
        Normalized vectors enable cosine similarity via dot product.
        """
        magnitude = math.sqrt(sum(float(v) ** 2 for v in vector))
        if not magnitude:
            return [float(v) for v in vector]
        return [float(v) / magnitude for v in vector]


class QueryExpander:
    """Expands queries for improved retrieval.
    
    Uses simple heuristics to add related terms.
    Could be extended with LLM-based expansion.
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
                # Add synonyms
                expansions.extend(synonyms)
        
        if expansions:
            return query + " " + " ".join(expansions)
        
        return query
