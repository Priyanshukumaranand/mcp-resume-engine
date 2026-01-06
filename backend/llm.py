from __future__ import annotations

import json
import os
from typing import List, Tuple, Dict, Any, Optional

try:
    import google.generativeai as genai  # type: ignore
except ImportError as exc:  # pragma: no cover - ensures clear error when dependency missing
    raise ImportError(
        "google-generativeai is required. Install dependencies via 'pip install -r requirements.txt'."
    ) from exc
from dotenv import load_dotenv

load_dotenv()

# Constant for missing/unavailable fields
UNKNOWN_VALUE = "not provided"

# Strict extraction prompt - enforces source attribution
STRICT_EXTRACTION_PROMPT = """
STRICT EXTRACTION MODE - Follow these rules EXACTLY:

1. Extract ONLY information that is EXPLICITLY written in the resume text below
2. DO NOT infer, assume, or add any information not directly stated
3. DO NOT normalize abbreviations unless they are spelled out in the text
4. DO NOT guess experience levels (junior/senior/expert) unless explicitly stated
5. DO NOT add skills, technologies, or tools not explicitly mentioned
6. For any missing field, use exactly: "not provided"

Output valid JSON with this exact structure:
{{
  "name": "extracted full name or 'not provided'",
  "email": "extracted email or 'not provided'",
  "role": {{
    "value": "exact job title from resume or 'not provided'",
    "source_span": "verbatim text containing the title"
  }},
  "skills": [
    {{"value": "exact skill name", "source_span": "text where this skill appears"}}
  ],
  "projects": [
    {{"value": "exact project name", "source_span": "text describing this project"}}
  ],
  "education": {{
    "value": "degree and institution or 'not provided'",
    "source_span": "verbatim education text"
  }},
  "experience": [
    {{"value": "company/role description", "source_span": "verbatim experience text"}}
  ],
  "summary": "brief factual summary of resume content, max 500 chars"
}}

CRITICAL RULES:
- Every extracted value MUST have a source_span quoting the actual resume text
- If a section is not found, use "not provided" as value and null as source_span
- Skills must be exact names mentioned (e.g., "Python" not "programming languages")
- Do not include Markdown formatting or code fences in output

RESUME TEXT:
{resume_text}
"""

# Evidence-based QA prompt
EVIDENCE_BASED_QA_PROMPT = """
Answer ONLY using the provided resume context. Follow these strict rules:

1. Base your answer ONLY on information explicitly stated in the context
2. DO NOT add information not present in the context
3. DO NOT make assumptions about skills, experience, or capabilities
4. If the context lacks sufficient information, respond exactly: "Insufficient information in resumes to answer this question."
5. When mentioning candidates, use their names as shown in the context
6. Keep your answer concise - one paragraph maximum

CONTEXT (Resume excerpts):
{context}

QUESTION: {question}

ANSWER:
"""


class ResumeLLM:
    """Gemini-backed helper with strict extraction and evidence-based QA."""

    def __init__(self, model_name: str | None = None) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is required to use Gemini.")
        genai.configure(api_key=self.api_key)
        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        self._model: genai.GenerativeModel | None = None

    def _model_instance(self) -> genai.GenerativeModel:
        if self._model is None:
            self._model = genai.GenerativeModel(self.model_name)
        return self._model

    def _generate(self, prompt: str, *, max_output_tokens: int = 512) -> str:
        model = self._model_instance()
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,  # Lower temperature for more factual responses
                    "max_output_tokens": max_output_tokens,
                },
            )
        except Exception as exc:  # pragma: no cover - passthrough error context
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

        text = self._extract_text(response)
        if text:
            return text

        finish_reason = None
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
        raise RuntimeError(
            "Gemini returned no text"
            + (f" (finish_reason={finish_reason})" if finish_reason is not None else "")
        )

    @staticmethod
    def _extract_text(response) -> str:
        try:
            text = (response.text or "").strip()
            if text:
                return text
        except ValueError:
            # The helper raised because no textual parts existed; fall back to manual parsing
            pass

        fragments: List[str] = []
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            parts = []
            content = getattr(candidate, "content", None)
            if content is not None and hasattr(content, "parts"):
                parts = content.parts or []
            elif hasattr(candidate, "parts"):
                parts = candidate.parts or []

            for part in parts:
                text_value = getattr(part, "text", None)
                if text_value:
                    fragments.append(text_value)

        return "\n".join(fragments).strip()

    def extract_resume_fields_strict(self, resume_text: str) -> Dict[str, Any]:
        """
        Extract resume fields in STRICT mode with source attribution.
        
        Returns structured data where every field includes source_span
        reference to the original text.
        
        Args:
            resume_text: Raw text extracted from PDF
            
        Returns:
            Dictionary with extracted fields and source attribution
        """
        text = (resume_text or "").strip()
        if not text:
            return self._empty_extraction()

        prompt = STRICT_EXTRACTION_PROMPT.format(resume_text=text[:15000])
        
        try:
            output = self._generate(prompt, max_output_tokens=1024)
            output = self._clean_json_output(output)
            data = json.loads(output)
            
            if isinstance(data, dict):
                return self._validate_extraction(data, text)
        except json.JSONDecodeError:
            pass
        except Exception:
            pass

        return self._fallback_extraction(text)

    def extract_resume_fields(self, resume_text: str) -> dict:
        """
        Legacy extraction method - calls strict extraction internally.
        
        Maintains backward compatibility with existing code.
        """
        strict_result = self.extract_resume_fields_strict(resume_text)
        
        # Convert to flat format for backward compatibility
        return {
            "name": strict_result.get("name", UNKNOWN_VALUE),
            "email": strict_result.get("email", UNKNOWN_VALUE),
            "role": self._get_field_value(strict_result.get("role")),
            "skills": self._extract_values_from_list(strict_result.get("skills", [])),
            "projects": self._extract_values_from_list(strict_result.get("projects", [])),
            "education": self._get_field_value(strict_result.get("education")),
            "experience": self._combine_experience(strict_result.get("experience", [])),
            "summary": strict_result.get("summary", ""),
        }

    def answer_question(self, question: str, context: str) -> str:
        """
        Answer question using ONLY the provided context.
        
        Implements evidence-based QA - never generates beyond context.
        """
        question = (question or "").strip()
        context = (context or "").strip()
        if not question or not context:
            return "Insufficient information in resumes to answer this question."

        prompt = EVIDENCE_BASED_QA_PROMPT.format(
            context=context[:10000],
            question=question,
        )
        
        try:
            answer = self._generate(prompt, max_output_tokens=300)
            return answer.strip()
        except RuntimeError:
            return "Insufficient information in resumes to answer this question."

    def _validate_extraction(self, data: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
        """
        Validate that extracted fields have source spans in original text.
        
        Marks fields as 'not provided' if source span cannot be verified.
        """
        import re
        
        name = self._validate_simple_field(data.get("name"), raw_text)
        
        # Fallback: try to extract name from first line if LLM didn't find it
        if name == UNKNOWN_VALUE:
            first_line = raw_text.split('\n')[0].strip() if raw_text else ""
            if first_line and len(first_line) < 50 and not re.search(r'[@\d]', first_line):
                name = first_line
        
        validated = {
            "name": name,
            "email": self._validate_email_field(data.get("email"), raw_text),
            "role": self._validate_sourced_field(data.get("role"), raw_text),
            "skills": self._validate_sourced_list(data.get("skills"), raw_text, lenient=True),
            "projects": self._validate_sourced_list(data.get("projects"), raw_text, lenient=True),
            "education": self._validate_sourced_field(data.get("education"), raw_text),
            "experience": self._validate_sourced_list(data.get("experience"), raw_text),
            "summary": str(data.get("summary", ""))[:500],
        }
        return validated

    def _validate_simple_field(self, value: Any, raw_text: str) -> str:
        """Validate simple string field exists in text."""
        if not value or value == UNKNOWN_VALUE:
            return UNKNOWN_VALUE
        
        str_value = str(value).strip()
        if not str_value or str_value.lower() == UNKNOWN_VALUE:
            return UNKNOWN_VALUE
        
        # Check if value or parts of it appear in raw text
        if str_value.lower() in raw_text.lower():
            return str_value
        
        # Check individual words for names
        words = str_value.split()
        if any(word.lower() in raw_text.lower() for word in words if len(word) > 2):
            return str_value
            
        return UNKNOWN_VALUE

    def _validate_email_field(self, value: Any, raw_text: str) -> str:
        """Validate email field."""
        if not value or value == UNKNOWN_VALUE:
            return UNKNOWN_VALUE
        
        str_value = str(value).strip().lower()
        if "@" not in str_value:
            return UNKNOWN_VALUE
        
        # Check if email appears in text
        if str_value in raw_text.lower():
            return str_value
            
        return UNKNOWN_VALUE

    def _validate_sourced_field(
        self, field: Any, raw_text: str
    ) -> Optional[Dict[str, Any]]:
        """Validate field with source attribution."""
        if not field:
            return {"value": UNKNOWN_VALUE, "source_span": None}
        
        if isinstance(field, str):
            # Simple string - try to find in text
            if field.lower() in raw_text.lower():
                return {"value": field, "source_span": field}
            return {"value": UNKNOWN_VALUE, "source_span": None}
        
        if isinstance(field, dict):
            value = str(field.get("value", UNKNOWN_VALUE))
            source = field.get("source_span")
            
            if value == UNKNOWN_VALUE:
                return {"value": UNKNOWN_VALUE, "source_span": None}
            
            # Verify source span exists in text
            if source and str(source).lower() in raw_text.lower():
                return {"value": value, "source_span": str(source)}
            elif value.lower() in raw_text.lower():
                return {"value": value, "source_span": value}
            
            return {"value": UNKNOWN_VALUE, "source_span": None}
        
        return {"value": UNKNOWN_VALUE, "source_span": None}

    def _validate_sourced_list(
        self, items: Any, raw_text: str, lenient: bool = False
    ) -> List[Dict[str, Any]]:
        """Validate list of fields with source attribution.
        
        Args:
            items: List of items to validate
            raw_text: Original resume text for verification
            lenient: If True, accept items even if not found in text (for skills)
        """
        if not items or not isinstance(items, list):
            return []
        
        validated = []
        raw_lower = raw_text.lower()
        
        for item in items:
            if isinstance(item, str):
                value = item.strip()
                if not value or value == UNKNOWN_VALUE:
                    continue
                # Check if in text, or accept if lenient mode
                if value.lower() in raw_lower:
                    validated.append({"value": value, "source_span": value})
                elif lenient and len(value) > 1:
                    # In lenient mode, accept skills that look valid
                    validated.append({"value": value, "source_span": None})
            elif isinstance(item, dict):
                value = str(item.get("value", "")).strip()
                source = item.get("source_span")
                
                if not value or value == UNKNOWN_VALUE:
                    continue
                
                # Verify in text or accept if lenient
                if source and str(source).lower() in raw_lower:
                    validated.append({"value": value, "source_span": str(source)})
                elif value.lower() in raw_lower:
                    validated.append({"value": value, "source_span": value})
                elif lenient and len(value) > 1:
                    validated.append({"value": value, "source_span": None})
        
        return validated

    def _get_field_value(self, field: Any) -> Optional[str]:
        """Extract value from sourced field."""
        if not field:
            return None
        if isinstance(field, str):
            return field if field != UNKNOWN_VALUE else None
        if isinstance(field, dict):
            value = field.get("value")
            return value if value and value != UNKNOWN_VALUE else None
        return None

    def _extract_values_from_list(self, items: List[Any]) -> List[str]:
        """Extract values from list of sourced fields."""
        values = []
        for item in items or []:
            if isinstance(item, str):
                if item and item != UNKNOWN_VALUE:
                    values.append(item)
            elif isinstance(item, dict):
                value = item.get("value")
                if value and value != UNKNOWN_VALUE:
                    values.append(value)
        return values[:15]  # Cap at 15 items

    def _combine_experience(self, items: List[Any]) -> str:
        """Combine experience items into summary text."""
        parts = []
        for item in items or []:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("value")
                if value and value != UNKNOWN_VALUE:
                    parts.append(value)
        return "; ".join(parts)[:1500]

    def _empty_extraction(self) -> Dict[str, Any]:
        """Return empty extraction result."""
        return {
            "name": UNKNOWN_VALUE,
            "email": UNKNOWN_VALUE,
            "role": {"value": UNKNOWN_VALUE, "source_span": None},
            "skills": [],
            "projects": [],
            "education": {"value": UNKNOWN_VALUE, "source_span": None},
            "experience": [],
            "summary": "",
        }

    def _fallback_extraction(self, text: str) -> Dict[str, Any]:
        """Fallback extraction when strict mode fails."""
        result = self._empty_extraction()
        result["summary"] = text[:500]
        
        import re
        
        # Try to extract email with regex
        email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
        if email_match:
            result["email"] = email_match.group(0).lower()
        
        # Try to extract name from first line (common resume format)
        first_line = text.split('\n')[0].strip() if text else ""
        # Name is usually just letters and spaces, no special chars
        if first_line and len(first_line) < 50:
            # Check if it looks like a name (not an email or phone)
            if not re.search(r'[@\d]', first_line):
                result["name"] = first_line
        
        return result

    @staticmethod
    def _clean_json_output(text: str) -> str:
        """Remove markdown code fences from JSON output."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    # Legacy methods for backward compatibility
    def _parse_response(self, response: str) -> Tuple[str | None, List[str]]:
        """Parse legacy response format."""
        response = response.strip()
        if not response:
            return None, []

        role = None
        skills: List[str] = []
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                role = self._clean_role(data.get("role"))
                skills = self._clean_skills(data.get("skills"))
                if role or skills:
                    return role, skills
        except Exception:
            pass

        return None, []

    @staticmethod
    def _clean_role(value) -> str | None:
        if not value:
            return None
        if isinstance(value, dict):
            value = value.get("value", "")
        text = str(value).strip().strip("\"'")
        return text if text and text != UNKNOWN_VALUE else None

    @staticmethod
    def _clean_skills(value) -> List[str]:
        if not value:
            return []
        if isinstance(value, str):
            candidates = value.split(",")
        elif isinstance(value, list):
            candidates = []
            for item in value:
                if isinstance(item, dict):
                    candidates.append(item.get("value", ""))
                else:
                    candidates.append(item)
        else:
            candidates = [value]
            
        cleaned: List[str] = []
        for item in candidates:
            text = str(item).strip().lower()
            if text and text != UNKNOWN_VALUE and text not in cleaned:
                cleaned.append(text)
        return cleaned[:15]
