"""Redis connection management for job queue.

Simple connection handling with environment-based configuration.
"""
from __future__ import annotations

import os
from typing import Optional

from redis import Redis
from rq import Queue

# Module-level connection cache
_redis_connection: Optional[Redis] = None
_queue: Optional[Queue] = None


def get_redis_url() -> str:
    """Get Redis URL from environment."""
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")


def get_redis_connection() -> Redis:
    """Get or create Redis connection.
    
    Returns:
        Redis connection instance (cached)
    """
    global _redis_connection
    
    if _redis_connection is None:
        url = get_redis_url()
        _redis_connection = Redis.from_url(url, decode_responses=False)
    
    return _redis_connection


def get_queue(name: str = "default") -> Queue:
    """Get or create RQ queue.
    
    Args:
        name: Queue name (default: "default")
        
    Returns:
        RQ Queue instance
    """
    global _queue
    
    if _queue is None or _queue.name != name:
        conn = get_redis_connection()
        _queue = Queue(name, connection=conn)
    
    return _queue


def is_redis_available() -> bool:
    """Check if Redis is available.
    
    Returns:
        True if Redis is reachable, False otherwise
    """
    try:
        conn = get_redis_connection()
        conn.ping()
        return True
    except Exception:
        return False


def reset_connections() -> None:
    """Reset cached connections (useful for testing)."""
    global _redis_connection, _queue
    
    if _redis_connection is not None:
        try:
            _redis_connection.close()
        except Exception:
            pass
        _redis_connection = None
    
    _queue = None
