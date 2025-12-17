from __future__ import annotations

import json
import os
from typing import List, Tuple

try:
    import google.generativeai as genai  # type: ignore
except ImportError as exc:  # pragma: no cover - ensures clear error when dependency missing
    raise ImportError(
        "google-generativeai is required. Install dependencies via 'pip install -r requirements.txt'."
    ) from exc
from dotenv import load_dotenv

load_dotenv()


class ResumeLLM:
    """Gemini-backed helper that extracts roles/skills and can craft answers."""

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
                    "temperature": 0.2,
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

    def extract_resume_fields(self, resume_text: str) -> dict:
        text = (resume_text or "").strip()
        if not text:
            return {}

        prompt = (
            "Extract structured resume data as JSON with keys: "
            "name (string), email (string), role (string), skills (array of lowercase strings), "
            "summary (<=500 chars), experience (<=1500 chars). "
            "Do not include contact headers or Markdown fences. Be concise.\n"
            f"RESUME:\n{text}"
        )
        output = self._generate(prompt, max_output_tokens=256)
        try:
            data = json.loads(output)
            if isinstance(data, dict):
                return {
                    "name": (str(data.get("name")) or "").strip(),
                    "email": (str(data.get("email")) or "").strip(),
                    "role": self._clean_role(data.get("role")),
                    "skills": self._clean_skills(data.get("skills")),
                    "summary": (str(data.get("summary")) or "").strip(),
                    "experience": (str(data.get("experience")) or "").strip(),
                }
        except Exception:
            pass

        # Fallback: reuse role/skills parsing and keep the text as summary/experience
        role, skills = self._parse_response(output)
        return {
            "name": "",
            "email": "",
            "role": role,
            "skills": skills,
            "summary": text[:1500],
            "experience": text[:1500],
        }

    def answer_question(self, question: str, context: str) -> str:
        question = (question or "").strip()
        context = (context or "").strip()
        if not question or not context:
            return "I don't have enough information to answer that yet."

        prompt = (
            "Use only the provided context to answer the question with a single concise paragraph. "
            "Do not enumerate candidates or list bullet points. "
            "If the context lacks the answer, respond exactly with 'I can't find out.'.\n"
            f"CONTEXT:\n{context}\nQUESTION: {question}\nANSWER:"
        )
        return self._generate(prompt, max_output_tokens=256)

    def _parse_response(self, response: str) -> Tuple[str | None, List[str]]:
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

        lowered = response.lower()
        if "skills" in lowered and "role" in lowered:
            role_hint = response.split("role")[-1].split("skills")[0]
            role = self._clean_role(role_hint.split(":")[-1])
            skills_part = response.split("skills")[-1]
            skills = self._clean_skills(skills_part.replace(":", "").replace("-", ","))
        else:
            role = self._clean_role(response.split("skills")[0])

        return role, skills

    @staticmethod
    def _clean_role(value) -> str | None:
        if not value:
            return None
        text = str(value).strip().strip("\"'")
        return text or None

    @staticmethod
    def _clean_skills(value) -> List[str]:
        if not value:
            return []
        if isinstance(value, str):
            candidates = value.split(",")
        else:
            candidates = value
        cleaned: List[str] = []
        for item in candidates:
            text = str(item).strip().lower()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned[:8]
