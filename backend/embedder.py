from __future__ import annotations

import math
import os
from typing import Iterable, List

from huggingface_hub import InferenceClient


class ResumeEmbedder:
    """Remote embedder that uses Hugging Face feature extraction API."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not token:
            raise RuntimeError("HUGGINGFACE_API_TOKEN is required for remote embeddings")
        # Use the official HF Inference client (supports serverless embeddings; no local weights)
        self.client = InferenceClient(token=token)

    def embed_text(self, text: str) -> List[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        items = list(texts)
        if not items:
            return []
        try:
            result = self.client.feature_extraction(
                items,
                model=self.model_name,
            )
        except Exception as exc:  # pragma: no cover - network errors
            raise RuntimeError(f"Hugging Face embedding failed: {exc}") from exc

        if result is None:
            raise RuntimeError("Embedding API returned no data")

        # huggingface_hub returns a numpy array when available; convert to list of lists
        vectors = result.tolist() if hasattr(result, "tolist") else result
        if not isinstance(vectors, list) or not vectors:
            raise RuntimeError("Unexpected embedding response shape")

        return [self._normalize(vec) for vec in vectors]

    @staticmethod
    def _normalize(vector: List[float]) -> List[float]:
        magnitude = math.sqrt(sum(float(value) ** 2 for value in vector))
        if not magnitude:
            return [float(value) for value in vector]
        return [float(value) / magnitude for value in vector]
