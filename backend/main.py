"""FastAPI application for resume processing and QA.

Privacy-first resume discovery API with semantic chunking
and evidence-based question answering.
"""
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

try:
    from .core import preprocess_resume_text, SemanticChunker
    from .embeddings import ResumeEmbedder, ResumeVectorStore
    from .retrieval import Reranker, ResponseVerifier
    from .models import (
        QARequest, QAResponse, QASource, Resume,
        ResumeListResponse, ResumeResponse,
    )
    from .llm import ResumeLLM
    from .anonymizer import generate_anon_id, strip_pii, extract_email, UNKNOWN_VALUE
except ImportError:
    from backend.core import preprocess_resume_text, SemanticChunker
    from backend.embeddings import ResumeEmbedder, ResumeVectorStore
    from backend.retrieval import Reranker, ResponseVerifier
    from backend.models import (
        QARequest, QAResponse, QASource, Resume,
        ResumeListResponse, ResumeResponse,
    )
    from backend.llm import ResumeLLM
    from backend.anonymizer import generate_anon_id, strip_pii, extract_email, UNKNOWN_VALUE

# Configuration
APP_NAME = "Resume Discovery API"
APP_VERSION = "3.0.0"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIRECTORY = PROJECT_ROOT / "chroma_storage"
CONFIDENCE_THRESHOLD = 0.35
FALLBACK_MSG = "I don't have enough information to answer this question."

# FastAPI app
app = FastAPI(title=APP_NAME, version=APP_VERSION)

# CORS
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()] or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Components
embedder = ResumeEmbedder()
vector_store = ResumeVectorStore(persist_directory=str(PERSIST_DIRECTORY))
resume_llm = ResumeLLM()
chunker = SemanticChunker()
reranker = Reranker()
verifier = ResponseVerifier()


@app.get("/health")
async def health():
    return {"status": "ok", "version": APP_VERSION}


@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """Ingest a PDF resume."""
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(400, "Only PDF uploads supported")

    pdf_text = _extract_pdf_text(file)
    if not pdf_text:
        raise HTTPException(400, "Could not read text from PDF")

    # Extract email and generate anon_id
    email = extract_email(pdf_text) or UNKNOWN_VALUE
    anon_id = generate_anon_id(email)
    
    # Preprocess and strip PII
    sanitized = strip_pii(preprocess_resume_text(pdf_text))

    # LLM extraction
    try:
        extracted = resume_llm.extract_resume_fields(pdf_text) or {}
    except RuntimeError:
        extracted = {"name": "Unknown", "summary": sanitized[:500]}

    # Create resume
    resume = Resume(
        id=str(uuid4()),
        anon_id=anon_id,
        name=extracted.get("name") or "Unknown",
        email=email,
        role=extracted.get("role"),
        skills=_dedupe(extracted.get("skills") or []),
        projects=extracted.get("projects") or [],
        education=extracted.get("education"),
        experience=extracted.get("experience") or sanitized[:1500],
        summary=extracted.get("summary") or sanitized[:500],
        raw_text=sanitized,
    )
    
    # Chunk and embed
    chunks, _ = chunker.chunk_resume(sanitized)
    section_types = [c.section_type for c in chunks]
    texts = [c.text for c in chunks]
    
    # Add candidate context prefix
    prefix = f"[Candidate: {resume.name}]\n"
    texts_with_ctx = [prefix + t for t in texts]
    
    try:
        embeddings = embedder.embed_documents(texts_with_ctx)
        vector_store.add_resume_chunks(resume, embeddings, texts_with_ctx, section_types=section_types)
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc
    
    return {
        "status": "processed",
        "id": resume.id,
        "anon_id": resume.anon_id,
        "name": resume.name,
        "role": resume.role,
        "skills": resume.skills,
        "projects": resume.projects,
        "education": resume.education,
        "chunk_count": len(chunks),
    }


@app.get("/resumes", response_model=ResumeListResponse, include_in_schema=False)
async def list_resumes():
    """List all resumes."""
    resumes = vector_store.get_all_resumes()
    return ResumeListResponse(resumes=[ResumeResponse.from_resume(r) for r in resumes])


@app.delete("/resumes/{resume_id}", include_in_schema=False)
async def delete_resume(resume_id: str):
    """Delete a resume."""
    vector_store.delete(resume_id)
    return {"status": "deleted", "id": resume_id}


@app.post("/qa", response_model=QAResponse)
async def answer_question(request: QARequest):
    """Answer questions about resumes."""
    if not vector_store.has_resumes():
        raise HTTPException(404, "No resumes stored")

    question = request.question.strip()
    query_skills = reranker.extract_skills_from_query(question)
    expanded = _expand_query(question)
    
    # Search
    embedding = embedder.embed_query(expanded)
    candidates = vector_store.query(embedding, top_k=request.top_k * 3)
    
    if not candidates:
        raise HTTPException(404, "No relevant resumes found")

    # Rerank
    ranked = reranker.rerank(question, candidates, query_skills=query_skills, top_k=request.top_k)
    
    # Calculate confidence
    similarities = [r.reranked_score for r in ranked]
    section_types = [r.section_type for r in ranked]
    confidence = verifier.calculate_confidence(similarities, len(ranked), section_types)
    
    # Build sources
    matches = [QAMatch(r.resume, r.chunk, r.reranked_score, r.section_type) for r in ranked]
    sources = [_build_source(m) for m in matches]
    evidence = [m.chunk[:200] for m in matches[:3] if m.chunk]
    
    # Generate answer
    answer = FALLBACK_MSG
    is_fallback = True
    
    if confidence >= CONFIDENCE_THRESHOLD and matches:
        context = _build_context(matches)
        try:
            raw = resume_llm.answer_question(question, context).strip()
            if raw and not _is_unhelpful(raw):
                answer = raw
                is_fallback = False
        except RuntimeError:
            pass
    
    return QAResponse(
        answer=answer,
        confidence_score=round(confidence, 3),
        source_sections=list(set(section_types)),
        evidence_snippets=evidence,
        sources=sources,
        is_fallback=is_fallback,
    )


# Helpers
@dataclass
class QAMatch:
    resume: Resume
    chunk: str
    similarity: float
    section_type: str


def _extract_pdf_text(upload: UploadFile) -> str:
    try:
        raw = upload.file.read()
    finally:
        upload.file.close()
    if not raw:
        return ""
    reader = PdfReader(BytesIO(raw))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n\n".join(texts)[:20000]


def _dedupe(skills: list) -> list:
    seen = set()
    result = []
    for s in skills or []:
        key = s.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(s.strip())
    return result[:15]


def _expand_query(q: str) -> str:
    ql = q.lower()
    if any(w in ql for w in ["skill", "tech", "know", "language"]):
        return f"{q} Technical Skills"
    if any(w in ql for w in ["project", "built", "developed"]):
        return f"{q} Projects"
    if any(w in ql for w in ["experience", "work", "job"]):
        return f"{q} Experience"
    if any(w in ql for w in ["education", "degree", "university"]):
        return f"{q} Education"
    return q


def _build_source(m: QAMatch) -> QASource:
    return QASource(
        id=m.resume.id,
        anon_id=m.resume.anon_id,
        name=m.resume.name,
        role=m.resume.role,
        skills=m.resume.skills,
        summary=(m.chunk or m.resume.summary or "")[:400],
    )


def _build_context(matches: List[QAMatch], limit: int = 5) -> str:
    parts = []
    for m in matches[:limit]:
        r = m.resume
        lines = [f"Candidate: {r.name}", f"Role: {r.role or 'Not specified'}"]
        if r.skills:
            lines.append(f"Skills: {', '.join(r.skills)}")
        if m.chunk:
            lines.append(f"Resume Content:\n{m.chunk}")
        parts.append("\n".join(lines))
    return "\n\n---\n\n".join(parts)


def _is_unhelpful(answer: str) -> bool:
    bad = {"i don't know", "i do not know", "insufficient", "not found", "no information"}
    lower = answer.lower()
    return any(p in lower for p in bad)