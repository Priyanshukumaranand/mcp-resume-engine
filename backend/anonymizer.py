"""Privacy utilities for PII handling and anonymization."""
from __future__ import annotations

import hashlib
import re
from typing import Optional

# Constant for missing/unavailable fields
UNKNOWN_VALUE = "not provided"


def generate_anon_id(email: str) -> str:
    """
    Generate a stable anonymous identifier from an email address.
    
    Uses SHA-256 hash truncated to 16 characters for readability
    while maintaining uniqueness.
    
    Args:
        email: Email address to hash
        
    Returns:
        16-character hex string identifier
    """
    if not email or email.strip().lower() == UNKNOWN_VALUE:
        # Generate from random placeholder to ensure uniqueness
        import uuid
        email = str(uuid.uuid4())
    
    normalized = email.strip().lower()
    hash_bytes = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return hash_bytes[:16]


# Regex patterns for PII detection
PHONE_PATTERNS = [
    r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}",  # International
    r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",  # US format
    r"\d{10,12}",  # Plain digits
]

EMAIL_PATTERN = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

# URLs including LinkedIn, GitHub, portfolio sites
URL_PATTERN = r"https?://[^\s<>\"{}|\\^`\[\]]+"
LINKEDIN_PATTERN = r"(?:linkedin\.com/in/|@)[a-zA-Z0-9_-]+"
GITHUB_PATTERN = r"(?:github\.com/|@)[a-zA-Z0-9_-]+"

# Physical addresses (simplified pattern)
ADDRESS_PATTERN = r"\d{1,5}\s+[\w\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|way|place|pl)\.?(?:[,\s]+[\w\s]+)?(?:[,\s]+[A-Z]{2}\s+\d{5}(?:-\d{4})?)?"


def strip_pii(text: str, preserve_name: bool = True) -> str:
    """
    Remove private/sensitive information from text.
    
    Removes:
    - Email addresses
    - Phone numbers
    - URLs (LinkedIn, GitHub, portfolio sites)
    - Physical addresses
    
    Args:
        text: Raw text containing potential PII
        preserve_name: If True, does not attempt to remove names
                      (names are allowed in chatbot responses)
                      
    Returns:
        Text with PII replaced by [REDACTED] markers
    """
    if not text:
        return text
    
    result = text
    
    # Remove emails
    result = re.sub(EMAIL_PATTERN, "[EMAIL REDACTED]", result, flags=re.IGNORECASE)
    
    # Remove phone numbers
    for pattern in PHONE_PATTERNS:
        result = re.sub(pattern, "[PHONE REDACTED]", result)
    
    # Remove URLs
    result = re.sub(URL_PATTERN, "[URL REDACTED]", result, flags=re.IGNORECASE)
    result = re.sub(LINKEDIN_PATTERN, "[LINKEDIN REDACTED]", result, flags=re.IGNORECASE)
    result = re.sub(GITHUB_PATTERN, "[GITHUB REDACTED]", result, flags=re.IGNORECASE)
    
    # Remove addresses (simplified)
    result = re.sub(ADDRESS_PATTERN, "[ADDRESS REDACTED]", result, flags=re.IGNORECASE)
    
    return result


def extract_email(text: str) -> Optional[str]:
    """
    Extract the first email address found in text.
    
    Args:
        text: Text to search for email
        
    Returns:
        Email address if found, None otherwise
    """
    if not text:
        return None
    
    match = re.search(EMAIL_PATTERN, text, re.IGNORECASE)
    return match.group(0).lower() if match else None


def is_valid_email(email: str) -> bool:
    """Check if string is a valid email format."""
    if not email:
        return False
    return bool(re.match(f"^{EMAIL_PATTERN}$", email.strip(), re.IGNORECASE))
