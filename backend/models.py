from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ExtractedField(BaseModel):
    """
    A field extracted from resume with source attribution.
    
    Ensures traceability - every extracted value references
    the exact text from which it was derived.
    """
    value: str
    source_section: str = Field(
        default="unknown",
        description="Section where field was found: skills, experience, projects, education, header, unknown"
    )
    source_span: Optional[str] = Field(
        default=None,
        description="Verbatim text from resume that contains this value"
    )

    @field_validator("value", mode="before")
    @classmethod
    def _clean_value(cls, v) -> str:
        if v is None:
            return "not provided"
        return str(v).strip() or "not provided"


class SectionChunk(BaseModel):
    """
    A text chunk with section metadata for weighted retrieval.
    """
    text: str
    section_type: str = Field(
        default="unknown",
        description="Section type: skills, experience, projects, education, header, unknown"
    )
    chunk_index: int = 0

    @field_validator("section_type", mode="before")
    @classmethod
    def _normalize_section(cls, v) -> str:
        if not v:
            return "unknown"
        normalized = str(v).lower().strip()
        valid_sections = {"skills", "experience", "projects", "education", "header", "unknown"}
        return normalized if normalized in valid_sections else "unknown"


class Resume(BaseModel):
    """
    Internal resume storage model.
    
    Privacy design:
    - name: Visible in chatbot responses
    - anon_id: Stable hash of email for identification
    - email: Stored internally but never exposed in API responses
    - Private details (phone, URLs): Stripped during extraction
    """
    id: str = Field(..., description="Unique identifier for the resume")
    anon_id: str = Field(..., description="SHA-256 hash of email for stable anonymous ID")
    name: str = Field(default="Unknown", description="Candidate name (visible in responses)")
    email: str = Field(default="", description="Email for internal use only, never exposed")
    role: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    education: Optional[str] = None
    experience: str = Field(default="")
    summary: str = Field(default="")
    raw_text: str = Field(default="", description="Original resume text for reprocessing")
    
    # Source attribution for strict extraction
    skills_with_source: List[ExtractedField] = Field(default_factory=list)
    projects_with_source: List[ExtractedField] = Field(default_factory=list)
    experience_with_source: List[ExtractedField] = Field(default_factory=list)

    @field_validator("id", mode="before")
    @classmethod
    def _strip_and_validate_id(cls, value: str) -> str:
        if value is None:
            raise ValueError("Value cannot be null")
        trimmed = str(value).strip()
        if not trimmed:
            raise ValueError("Value cannot be empty")
        return trimmed

    @field_validator("anon_id", mode="before")
    @classmethod
    def _strip_anon_id(cls, value: str) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("name", mode="before")
    @classmethod
    def _clean_name(cls, value) -> str:
        if value is None:
            return "Unknown"
        trimmed = str(value).strip()
        return trimmed or "Unknown"

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_role(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        trimmed = str(value).strip()
        return trimmed or None

    @field_validator("skills", "projects", mode="before")
    @classmethod
    def _normalize_list(cls, value):
        if not value:
            return []
        if isinstance(value, str):
            candidates = value.split(",")
        else:
            candidates = value
        cleaned = []
        for item in candidates:
            text = str(item).strip()
            if text:
                cleaned.append(text)
        return cleaned


class ResumeListResponse(BaseModel):
    """API response for listing resumes - email hidden."""
    resumes: List["ResumeResponse"]


class ResumeResponse(BaseModel):
    """
    API response model for resume data.
    
    Privacy: Name visible, email/phone/URLs hidden.
    """
    id: str
    anon_id: str
    name: str
    role: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    education: Optional[str] = None
    summary: str = ""

    @classmethod
    def from_resume(cls, resume: Resume) -> "ResumeResponse":
        """Create API response from internal Resume model."""
        return cls(
            id=resume.id,
            anon_id=resume.anon_id,
            name=resume.name,
            role=resume.role,
            skills=resume.skills,
            projects=resume.projects,
            education=resume.education,
            summary=resume.summary[:500] if resume.summary else "",
        )


class QARequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)

    @field_validator("question", mode="before")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        if value is None:
            raise ValueError("question is required")
        trimmed = str(value).strip()
        if not trimmed:
            raise ValueError("question cannot be empty")
        return trimmed


class QASource(BaseModel):
    """Source candidate in QA response - name visible, email hidden."""
    id: str
    anon_id: str
    name: str
    role: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    summary: str


class QAResponse(BaseModel):
    """
    Evidence-based QA response with confidence scoring.
    
    Includes:
    - confidence_score: Derived from similarity scores
    - source_sections: Which resume sections contributed
    - evidence_snippets: Verbatim text supporting the answer
    - is_fallback: True if answer is a fallback due to low confidence
    """
    answer: str
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_sections: List[str] = Field(default_factory=list)
    evidence_snippets: List[str] = Field(default_factory=list)
    sources: List[QASource]
    is_fallback: bool = False

