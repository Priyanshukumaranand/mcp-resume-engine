"""Section detection for resume text using hybrid heuristics + NLP.

Detects major resume sections: skills, experience, projects, education, achievements.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# Supported section types
SECTION_TYPES = frozenset({
    "header",
    "skills", 
    "experience",
    "projects",
    "education",
    "achievements",
    "unknown",
})

# Section weights for retrieval scoring
SECTION_WEIGHTS = {
    "skills": 1.3,       # High relevance for skill-based queries
    "projects": 1.2,     # Projects demonstrate applied skills
    "experience": 1.15,  # Work experience is valuable context
    "achievements": 1.1, # Awards/accomplishments add credibility
    "education": 1.0,    # Standard weight
    "header": 0.9,       # Contact/summary less relevant for technical queries
    "unknown": 1.0,      # Default weight
}


@dataclass
class Section:
    """A detected section in a resume."""
    section_type: str
    start_idx: int
    end_idx: int
    text: str = ""
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.section_type not in SECTION_TYPES:
            self.section_type = "unknown"


class SectionDetector:
    """Hybrid section detector using regex patterns and semantic cues.
    
    Uses a cascading approach:
    1. Header pattern matching (high confidence)
    2. Keyword density analysis (medium confidence)
    3. Fallback to unknown (low confidence)
    """
    
    # Section header patterns - ordered by specificity
    HEADER_PATTERNS = {
        "skills": [
            r"(?i)^\s*(?:technical\s+)?skills?\s*[:\-–|]?\s*$",
            r"(?i)^\s*(?:core\s+)?competenc(?:ies|e)\s*[:\-–|]?\s*$",
            r"(?i)^\s*technologies?\s*[:\-–|]?\s*$",
            r"(?i)^\s*tech(?:nical)?\s+stack\s*[:\-–|]?\s*$",
            r"(?i)^\s*proficienc(?:ies|y)\s*[:\-–|]?\s*$",
            r"(?i)^\s*tools?\s*(?:&|and)?\s*technologies?\s*[:\-–|]?\s*$",
            r"(?i)^\s*programming\s+languages?\s*[:\-–|]?\s*$",
        ],
        "experience": [
            r"(?i)^\s*(?:work\s+)?experience\s*[:\-–|]?\s*$",
            r"(?i)^\s*employment\s*(?:history)?\s*[:\-–|]?\s*$",
            r"(?i)^\s*work\s+history\s*[:\-–|]?\s*$",
            r"(?i)^\s*professional\s+experience\s*[:\-–|]?\s*$",
            r"(?i)^\s*career\s+(?:history|summary)\s*[:\-–|]?\s*$",
        ],
        "projects": [
            r"(?i)^\s*projects?\s*[:\-–|]?\s*$",
            r"(?i)^\s*(?:personal|academic|side)\s+projects?\s*[:\-–|]?\s*$",
            r"(?i)^\s*portfolio\s*[:\-–|]?\s*$",
            r"(?i)^\s*work\s+samples?\s*[:\-–|]?\s*$",
            r"(?i)^\s*key\s+projects?\s*[:\-–|]?\s*$",
        ],
        "education": [
            r"(?i)^\s*education(?:al\s+background)?\s*[:\-–|]?\s*$",
            r"(?i)^\s*academic\s*(?:background|qualifications?)?\s*[:\-–|]?\s*$",
            r"(?i)^\s*degrees?\s*[:\-–|]?\s*$",
            r"(?i)^\s*qualifications?\s*[:\-–|]?\s*$",
            r"(?i)^\s*certifications?\s*(?:&|and)?\s*(?:education)?\s*[:\-–|]?\s*$",
        ],
        "achievements": [
            r"(?i)^\s*achievements?\s*[:\-–|]?\s*$",
            r"(?i)^\s*accomplishments?\s*[:\-–|]?\s*$",
            r"(?i)^\s*awards?\s*(?:&|and)?\s*(?:honors?|recognition)?\s*[:\-–|]?\s*$",
            r"(?i)^\s*publications?\s*[:\-–|]?\s*$",
            r"(?i)^\s*honors?\s*(?:&|and)?\s*awards?\s*[:\-–|]?\s*$",
            r"(?i)^\s*recognition\s*[:\-–|]?\s*$",
        ],
    }
    
    # Keywords for section content analysis
    SECTION_KEYWORDS = {
        "skills": {
            "python", "java", "javascript", "react", "sql", "aws", "docker",
            "kubernetes", "machine learning", "deep learning", "tensorflow",
            "node", "typescript", "git", "linux", "api", "rest", "graphql",
            "mongodb", "postgresql", "redis", "html", "css", "c++", "golang",
        },
        "experience": {
            "worked", "developed", "managed", "led", "implemented", "designed",
            "built", "created", "maintained", "collaborated", "delivered",
            "company", "inc", "corp", "llc", "ltd", "present", "current",
            "senior", "junior", "engineer", "developer", "manager", "lead",
        },
        "education": {
            "university", "college", "bachelor", "master", "phd", "degree",
            "gpa", "graduated", "coursework", "major", "minor", "b.s.", "m.s.",
            "b.tech", "m.tech", "mba", "school", "institute", "certification",
        },
        "projects": {
            "built", "developed", "created", "implemented", "designed",
            "github", "deployed", "features", "users", "application", "app",
            "website", "platform", "tool", "system", "architecture",
        },
        "achievements": {
            "award", "won", "recognized", "published", "patent", "first place",
            "top", "best", "achievement", "honor", "scholarship", "featured",
            "speaker", "conference", "competition", "hackathon",
        },
    }
    
    def __init__(self):
        # Compile patterns for efficiency
        self._compiled_patterns = {
            section: [re.compile(p) for p in patterns]
            for section, patterns in self.HEADER_PATTERNS.items()
        }
    
    def detect_sections(self, text: str) -> List[Section]:
        """Detect all sections in resume text.
        
        Args:
            text: Full resume text
            
        Returns:
            List of Section objects with boundaries and types
        """
        if not text:
            return []
        
        lines = text.split("\n")
        section_boundaries: List[Tuple[str, int, float]] = []  # (type, line_idx, confidence)
        
        # First pass: detect section headers
        for i, line in enumerate(lines):
            section_type, confidence = self._match_header(line.strip())
            if section_type:
                section_boundaries.append((section_type, i, confidence))
        
        # Build sections from boundaries
        sections = self._build_sections(text, lines, section_boundaries)
        
        # Second pass: analyze content for missed sections
        sections = self._refine_with_content_analysis(sections, text)
        
        return sections
    
    def _match_header(self, line: str) -> Tuple[Optional[str], float]:
        """Match a line against section header patterns.
        
        Returns:
            Tuple of (section_type, confidence) or (None, 0)
        """
        if not line or len(line) > 100:  # Headers typically short
            return None, 0
        
        for section_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.match(line):
                    return section_type, 0.95
        
        return None, 0
    
    def _build_sections(
        self,
        text: str,
        lines: List[str],
        boundaries: List[Tuple[str, int, float]],
    ) -> List[Section]:
        """Build Section objects from detected boundaries."""
        sections = []
        
        if not boundaries:
            # No sections detected - treat entire text as unknown
            return [Section(
                section_type="header" if len(text) < 500 else "unknown",
                start_idx=0,
                end_idx=len(text),
                text=text,
                confidence=0.5,
            )]
        
        # Add header section if first boundary isn't at start
        if boundaries[0][1] > 0:
            header_end = self._get_line_position(lines, boundaries[0][1])
            header_text = text[:header_end].strip()
            if header_text:
                sections.append(Section(
                    section_type="header",
                    start_idx=0,
                    end_idx=header_end,
                    text=header_text,
                    confidence=0.8,
                ))
        
        # Build sections from boundaries
        for i, (section_type, line_idx, confidence) in enumerate(boundaries):
            start_idx = self._get_line_position(lines, line_idx)
            
            if i + 1 < len(boundaries):
                end_idx = self._get_line_position(lines, boundaries[i + 1][1])
            else:
                end_idx = len(text)
            
            section_text = text[start_idx:end_idx].strip()
            if section_text:
                sections.append(Section(
                    section_type=section_type,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    text=section_text,
                    confidence=confidence,
                ))
        
        return sections
    
    def _get_line_position(self, lines: List[str], line_idx: int) -> int:
        """Get character position of line start."""
        position = 0
        for i, line in enumerate(lines):
            if i >= line_idx:
                break
            position += len(line) + 1  # +1 for newline
        return position
    
    def _refine_with_content_analysis(
        self,
        sections: List[Section],
        full_text: str,
    ) -> List[Section]:
        """Refine section types using keyword density analysis."""
        refined = []
        
        for section in sections:
            if section.section_type == "unknown" and section.confidence < 0.8:
                # Try to identify section by content
                detected_type = self._analyze_content(section.text)
                if detected_type:
                    section.section_type = detected_type
                    section.confidence = 0.7
            
            refined.append(section)
        
        return refined
    
    def _analyze_content(self, text: str) -> Optional[str]:
        """Analyze text content to infer section type."""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        scores = {}
        for section_type, keywords in self.SECTION_KEYWORDS.items():
            matches = words.intersection(keywords)
            if matches:
                # Score based on keyword density
                scores[section_type] = len(matches) / len(keywords)
        
        if scores:
            best_match = max(scores, key=scores.get)
            if scores[best_match] > 0.1:  # At least 10% keyword match
                return best_match
        
        return None
    
    def get_section_weight(self, section_type: str) -> float:
        """Get retrieval weight for a section type."""
        return SECTION_WEIGHTS.get(section_type, 1.0)
