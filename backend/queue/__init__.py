"""Queue module for Redis-based job processing.

Provides a simple queue infrastructure with fixed-size worker pool
to handle concurrent Gemini API requests without rate limiting.
"""
from .connection import get_redis_connection, get_queue, is_redis_available
from .jobs import IngestJob, QAJob, JobStatus, JobResult

__all__ = [
    "get_redis_connection",
    "get_queue",
    "is_redis_available",
    "IngestJob",
    "QAJob",
    "JobStatus",
    "JobResult",
]
