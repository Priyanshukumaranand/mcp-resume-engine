from __future__ import annotations

import os
from uuid import uuid4
from dataclasses import dataclass
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from io import BytesIO

try:  # Support running as package or as a plain script
    from .embedder import ResumeEmbedder
    from .graph import build_add_resume_graph, build_refresh_resume_graph
    from .models import (
        QARequest,
        QAResponse,
        QASource,
        Resume,
        ResumeListResponse,
    )
    from .llm import ResumeLLM
    from .vectorstore import ResumeVectorStore
except ImportError:
    from embedder import ResumeEmbedder  # type: ignore
    from graph import build_add_resume_graph, build_refresh_resume_graph  # type: ignore
    from models import (  # type: ignore
        QARequest,
        QAResponse,
        QASource,
        Resume,
        ResumeListResponse,
    )
    from llm import ResumeLLM  # type: ignore
    from vectorstore import ResumeVectorStore  # type: ignore


APP_NAME = "Hackathon Teammate Recommendation API"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIRECTORY = PROJECT_ROOT / "chroma_storage"

app = FastAPI(title=APP_NAME, version="1.0.0")
default_origins = (
    "https://ce-bootcamp.vercel.app,"
    "https://branchbase-backend.azurewebsites.net,"
    "http://localhost:3000,"
    "http://127.0.0.1:3000,"
    "http://localhost:5173,"
    "http://127.0.0.1:5173"
)
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", default_origins).split(",")
    if origin.strip()
]
if not allowed_origins:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)
embedder = ResumeEmbedder()
vector_store = ResumeVectorStore(persist_directory=str(PERSIST_DIRECTORY))
resume_llm = ResumeLLM()
add_resume_graph = build_add_resume_graph(embedder, vector_store)
refresh_resume_graph = build_refresh_resume_graph(resume_llm, embedder, vector_store)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...)) -> dict:
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    pdf_text = _extract_pdf_text(file)
    if not pdf_text:
        raise HTTPException(status_code=400, detail="Could not read any text from the PDF")

    try:
        extracted = resume_llm.extract_resume_fields(pdf_text) or {}
    except RuntimeError:
        # Fallback: still ingest the text so RAG works even if Gemini extraction fails
        extracted = {}
    resume = Resume(
        id=str(uuid4()),
        name=extracted.get("name") or "Unknown",
        email=extracted.get("email") or "unknown@example.com",
        role=extracted.get("role"),
        skills=_deduplicate_skills(extracted.get("skills") or []),
        experience=extracted.get("experience") or pdf_text,
        summary=extracted.get("summary") or pdf_text[:1500],
    )
    try:
        add_resume_graph.invoke({"resume": resume})
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"status": "created", "id": resume.id}


@app.get("/resumes", response_model=ResumeListResponse)
async def list_resumes() -> ResumeListResponse:
    resumes = vector_store.get_all_resumes()
    return ResumeListResponse(resumes=resumes)


@app.delete("/resumes/{resume_id}")
async def delete_resume(resume_id: str) -> dict:
    deleted = vector_store.delete(resume_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Resume not found")
    return {"status": "deleted", "id": resume_id}


@app.post("/resumes/reextract")
async def reextract_resumes() -> dict:
    resumes = vector_store.get_all_resumes()
    results = []
    for resume in resumes:
        try:
            refresh_resume_graph.invoke({"resume": resume})
            results.append({"id": resume.id, "status": "updated"})
        except Exception as exc:
            results.append({"id": resume.id, "status": "failed", "error": str(exc)[:200]})
    return {"count": len(results), "results": results}


@app.post("/qa", response_model=QAResponse)
async def answer_question(request: QARequest) -> QAResponse:
    if not vector_store.has_resumes():
        raise HTTPException(status_code=404, detail="No resumes stored")

    question = request.question.strip()
    query_embedding = embedder.embed_text(question)
    candidates = vector_store.query(query_embedding, top_k=request.top_k)
    if not candidates:
        raise HTTPException(status_code=404, detail="No relevant resumes found")

    matches = _prepare_matches(candidates)
    sources = [_build_qa_source(match) for match in matches]
    context = _build_qa_context(matches)
    try:
        answer = resume_llm.answer_question(question, context).strip()
    except RuntimeError:
        answer = ""

    if not answer or answer.lower() in {"i do not know", "i don't know"}:
        answer = _build_rule_based_answer(question, matches)
    if not answer:
        answer = "I can't find out."
    return QAResponse(answer=answer, sources=sources)


def _extract_pdf_text(upload: UploadFile) -> str:
    try:
        raw = upload.file.read()
    finally:
        upload.file.close()

    if not raw:
        return ""

    reader = PdfReader(BytesIO(raw))
    texts: list[str] = []
    for page in reader.pages:
        try:
            extracted = page.extract_text() or ""
        except Exception:
            extracted = ""
        if extracted:
            texts.append(extracted.strip())

    combined = "\n\n".join(texts)
    # Trim overly large payloads to keep embeddings reasonable
    return combined[:20000]


def _deduplicate_skills(skills: list[str]) -> list[str]:
    seen = set()
    cleaned = []
    for skill in skills or []:
        text = skill.strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(text)
    return cleaned[:10]


@dataclass
class QAMatch:
    resume: Resume
    chunk: str
    similarity: float


def _prepare_matches(candidates: List[tuple[Resume, str, float]]) -> List[QAMatch]:
    matches: List[QAMatch] = []
    for resume, chunk, similarity in candidates:
        matches.append(QAMatch(resume=resume, chunk=chunk, similarity=similarity))
    matches.sort(key=lambda match: match.similarity, reverse=True)
    return matches


def _build_qa_source(match: QAMatch) -> QASource:
    resume = match.resume
    excerpt = (match.chunk or resume.summary or resume.experience)[:400]
    return QASource(
        id=resume.id,
        name=resume.name,
        role=resume.role,
        skills=resume.skills,
        summary=excerpt,
    )


def _build_qa_context(matches: List[QAMatch], limit: int = 5) -> str:
    chunks: List[str] = []
    for match in matches[:limit]:
        resume = match.resume
        parts = [
            f"Name: {resume.name}",
            f"Role: {resume.role or 'Unknown'}",
            f"Skills: {', '.join(resume.skills) if resume.skills else 'None'}",
            f"Summary: {resume.summary or resume.experience}",
            f"Experience: {resume.experience}",
            "FullDocument:",
            match.chunk,
        ]
        chunks.append("\n".join(parts))
    return "\n---\n".join(chunks)


def _build_rule_based_answer(question: str, matches: List[QAMatch]) -> str:
    if not matches:
        return ""

    top = matches[0].resume
    role = top.role or "Unknown role"
    skills = ", ".join(top.skills[:5]) if top.skills else "no specific skills listed"
    summary = (matches[0].chunk or top.summary or top.experience or "").strip()
    if summary:
        summary = summary[:320] + ("…" if len(summary) > 320 else "")

    parts = [
        f"Based on the stored resumes, {top.name} ({role}) seems most relevant to '{question}'.",
        f"Key skills: {skills}.",
    ]
    if summary:
        parts.append(f"Excerpt: {summary}")
    return " ".join(parts).strip()