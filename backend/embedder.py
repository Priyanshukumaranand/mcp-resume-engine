from __future__ import annotations

from typing import Iterable, List

from sentence_transformers import SentenceTransformer


class ResumeEmbedder:
    """Wraps SentenceTransformer to provide normalized embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def _model_instance(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str) -> List[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        model = self._model_instance()
        embeddings = model.encode(list(texts), normalize_embeddings=True)
        return [embedding.tolist() for embedding in embeddings]
