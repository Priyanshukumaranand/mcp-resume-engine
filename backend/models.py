from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Resume(BaseModel):
    id: str = Field(..., description="Unique identifier for the resume")
    name: str
    email: str
    role: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: str
    summary: str

    @field_validator("id", "name", "email", mode="before")
    @classmethod
    def _strip_and_validate(cls, value: str) -> str:
        if value is None:
            raise ValueError("Value cannot be null")
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Value cannot be empty")
        return trimmed

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_role(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None

    @field_validator("skills", mode="before")
    @classmethod
    def _normalize_skills(cls, value):
        if not value:
            return []
        if isinstance(value, str):
            candidates = value.split(",")
        else:
            candidates = value
        cleaned = []
        for item in candidates:
            text = item.strip()
            if text:
                cleaned.append(text)
        return cleaned


class ResumeListResponse(BaseModel):
    resumes: List[Resume]


class QARequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)

    @field_validator("question", mode="before")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        if value is None:
            raise ValueError("question is required")
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("question cannot be empty")
        return trimmed


class QASource(BaseModel):
    id: str
    name: str
    role: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    summary: str


class QAResponse(BaseModel):
    answer: str
    sources: List[QASource]
