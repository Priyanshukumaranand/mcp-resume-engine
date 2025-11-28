from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional


class Resume(BaseModel):
    id: str = Field(..., description="Unique identifier for the resume")
    name: str
    email: str
    role: str
    skills: List[str] = Field(default_factory=list)
    experience: str
    summary: str


class ResumeListResponse(BaseModel):
    resumes: List[Resume]


class RecommendationRequest(BaseModel):
    role: str
    skills: List[str] = Field(default_factory=list)
    summary: str = ""
    top_k: int = Field(default=5, ge=1, le=20, description="Number of candidates to evaluate before reranking")


class RecommendationResult(BaseModel):
    candidate: Resume
    match_score: float
    matching_skills: List[str] = Field(default_factory=list)
    explanation: str


class RecommendationResponse(BaseModel):
    best_match: RecommendationResult
    considered: int
