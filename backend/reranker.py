from __future__ import annotations

import os
from typing import List

import requests


class RecommendationRanker:
	"""Remote reranker that calls Hugging Face cross-encoder pipeline."""

	def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
		self.model_name = model_name
		self.token = os.getenv("HUGGINGFACE_API_TOKEN")
		self.enabled = bool(self.token)
		if self.enabled:
			self.session = requests.Session()
			self.session.headers.update(
				{
					"Authorization": f"Bearer {self.token}",
					"Content-Type": "application/json",
				}
			)
			self.url = f"https://api-inference.huggingface.co/models/{self.model_name}"

	def score(self, query: str, candidate_text: str) -> float:
		if not query or not candidate_text:
			return 0.0
		if not self.enabled:
			return 0.0
		payload = {"inputs": [[query, candidate_text]], "options": {"wait_for_model": True}}
		response = self.session.post(self.url, json=payload, timeout=45)
		response.raise_for_status()
		data = response.json()
		score = self._parse_score(data)
		return max(0.0, min(1.0, score))

	@staticmethod
	def _parse_score(payload: List) -> float:
		if not payload or not isinstance(payload, list):
			return 0.0
		first = payload[0]
		if isinstance(first, dict):
			value = first.get("score")
			if isinstance(value, (int, float)):
				return float(value)
		if isinstance(first, (int, float)):
			return float(first)
		return 0.0


