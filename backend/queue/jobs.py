"""Job models for queue-based processing.

Defines structured job payloads for ingest and QA tasks.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job execution status."""
    QUEUED = "queued"
    STARTED = "started"
    FINISHED = "finished"
    FAILED = "failed"


class IngestJob(BaseModel):
    """Job payload for PDF ingestion.
    
    Contains all data needed to process a resume
    in a background worker.
    """
    pdf_text: str = Field(..., description="Extracted PDF text content")
    filename: str = Field(default="resume.pdf", description="Original filename")
    
    class Config:
        extra = "forbid"


class QAJob(BaseModel):
    """Job payload for question answering.
    
    Contains the question and retrieval parameters.
    """
    question: str = Field(..., description="User question")
    top_k: int = Field(default=3, ge=1, le=10)
    use_multihop: bool = Field(default=False)
    
    class Config:
        extra = "forbid"


class JobResult(BaseModel):
    """Result wrapper for completed jobs.
    
    Provides consistent structure for both success
    and failure cases.
    """
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(default=JobStatus.QUEUED)
    result: Optional[Dict[str, Any]] = Field(default=None)
    error: Optional[str] = Field(default=None)
    
    @classmethod
    def success(cls, job_id: str, result: Dict[str, Any]) -> "JobResult":
        """Create successful result."""
        return cls(
            job_id=job_id,
            status=JobStatus.FINISHED,
            result=result,
        )
    
    @classmethod
    def failure(cls, job_id: str, error: str) -> "JobResult":
        """Create failed result."""
        return cls(
            job_id=job_id,
            status=JobStatus.FAILED,
            error=error,
        )


class AsyncJobResponse(BaseModel):
    """API response for async job submission."""
    job_id: str
    status: str = "queued"
    message: str = "Job submitted successfully"


class JobStatusResponse(BaseModel):
    """API response for job status check."""
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
