"""Background worker for processing Gemini API jobs.

Runs as a separate process, pulling jobs from Redis queue
and executing them with proper error handling.

Usage:
    python -m backend.queue.worker
    
Or with RQ:
    rq worker --with-scheduler
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from rq import Worker, Queue
from redis import Redis

from backend.queue.connection import get_redis_connection, get_queue
from backend.queue.jobs import IngestJob, QAJob, JobResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("worker")


def process_ingest_job(pdf_text: str, filename: str = "resume.pdf") -> Dict[str, Any]:
    """Process PDF ingestion job.
    
    Runs LLM extraction and stores results in vector store.
    This runs in the worker process, not the API process.
    
    Args:
        pdf_text: Extracted PDF text content
        filename: Original filename for reference
        
    Returns:
        Dict with processing results
    """
    # Import here to avoid loading heavy models in API process
    from backend.core import preprocess_resume_text, SemanticChunker
    from backend.embeddings import ResumeEmbedder, ResumeVectorStore
    from backend.llm import ResumeLLM
    from backend.anonymizer import generate_anon_id, strip_pii, extract_email, UNKNOWN_VALUE
    from backend.models import Resume
    
    logger.info(f"Processing ingest job for: {filename}")
    
    try:
        # Initialize components (lazy loaded in worker)
        embedder = ResumeEmbedder()
        vector_store = ResumeVectorStore(
            persist_directory=str(PROJECT_ROOT / "chroma_storage")
        )
        resume_llm = ResumeLLM()
        chunker = SemanticChunker()
        
        # Extract email and generate anon_id
        email = extract_email(pdf_text) or UNKNOWN_VALUE
        anon_id = generate_anon_id(email)
        
        # Preprocess and strip PII
        sanitized = strip_pii(preprocess_resume_text(pdf_text))
        
        # LLM extraction (this is the slow Gemini call)
        try:
            extracted = resume_llm.extract_resume_fields(pdf_text) or {}
        except RuntimeError as e:
            logger.warning(f"LLM extraction failed: {e}")
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
        
        embeddings = embedder.embed_documents(texts_with_ctx)
        vector_store.add_resume_chunks(
            resume, embeddings, texts_with_ctx, section_types=section_types
        )
        
        logger.info(f"Successfully processed resume: {resume.id}")
        
        return {
            "status": "processed",
            "id": resume.id,
            "anon_id": resume.anon_id,
            "name": resume.name,
            "skills": resume.skills,
            "projects": resume.projects,
            "education": resume.education,
            "chunk_count": len(chunks),
        }
        
    except Exception as e:
        logger.error(f"Ingest job failed: {e}")
        raise


def process_qa_job(
    question: str,
    top_k: int = 3,
    use_multihop: bool = False,
) -> Dict[str, Any]:
    """Process question answering job.
    
    Runs retrieval and LLM-based answer generation.
    
    Args:
        question: User question
        top_k: Number of results to return
        use_multihop: Whether to use multihop retrieval
        
    Returns:
        Dict with QA response
    """
    # Import here to avoid loading heavy models in API process
    from backend.core import SemanticChunker
    from backend.embeddings import ResumeEmbedder, ResumeVectorStore
    from backend.retrieval import Reranker, ResponseVerifier, MultiHopRetriever
    from backend.llm import ResumeLLM
    from backend.models import Resume
    
    logger.info(f"Processing QA job: {question[:50]}...")
    
    CONFIDENCE_THRESHOLD = 0.35
    FALLBACK_MSG = "I don't have enough information to answer this question."
    
    try:
        # Initialize components
        embedder = ResumeEmbedder()
        vector_store = ResumeVectorStore(
            persist_directory=str(PROJECT_ROOT / "chroma_storage")
        )
        resume_llm = ResumeLLM()
        reranker = Reranker()
        verifier = ResponseVerifier()
        multihop_retriever = MultiHopRetriever(
            embedder=embedder,
            vector_store=vector_store,
            llm=resume_llm,
            reranker=reranker,
            max_hops=3,
        )
        
        if not vector_store.has_resumes():
            return {
                "answer": "No resumes stored",
                "confidence_score": 0.0,
                "sources": [],
                "is_fallback": True,
            }
        
        question = question.strip()
        query_skills = reranker.extract_skills_from_query(question)
        
        # Choose retrieval strategy
        if use_multihop:
            candidates, _ = multihop_retriever.retrieve_to_tuples(
                question, top_k * 3
            )
        else:
            expanded = _expand_query(question)
            embedding = embedder.embed_query(expanded)
            candidates = vector_store.query(embedding, top_k=top_k * 3)
        
        if not candidates:
            return {
                "answer": "No relevant resumes found",
                "confidence_score": 0.0,
                "sources": [],
                "is_fallback": True,
            }
        
        # Rerank
        ranked = reranker.rerank(
            question, candidates, query_skills=query_skills, top_k=top_k
        )
        
        # Calculate confidence
        similarities = [r.reranked_score for r in ranked]
        section_types = [r.section_type for r in ranked]
        confidence = verifier.calculate_confidence(similarities, len(ranked), section_types)
        
        # Build sources
        sources = []
        evidence = []
        for r in ranked:
            sources.append({
                "id": r.resume.id,
                "anon_id": r.resume.anon_id,
                "name": r.resume.name,
                "role": r.resume.role,
                "skills": r.resume.skills,
                "summary": (r.chunk or r.resume.summary or "")[:400],
            })
            if r.chunk:
                evidence.append(r.chunk[:200])
        
        # Generate answer (this is the slow Gemini call)
        answer = FALLBACK_MSG
        is_fallback = True
        
        if confidence >= CONFIDENCE_THRESHOLD and ranked:
            context = _build_context(ranked)
            try:
                raw = resume_llm.answer_question(question, context).strip()
                if raw and not _is_unhelpful(raw):
                    answer = raw
                    is_fallback = False
            except RuntimeError:
                pass
        
        logger.info(f"QA job complete, confidence: {confidence:.3f}")
        
        return {
            "answer": answer,
            "confidence_score": round(confidence, 3),
            "source_sections": list(set(section_types)),
            "evidence_snippets": evidence[:3],
            "sources": sources,
            "is_fallback": is_fallback,
        }
        
    except Exception as e:
        logger.error(f"QA job failed: {e}")
        raise


# Helper functions (duplicated from main.py to avoid circular imports)
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


def _build_context(ranked, limit: int = 5) -> str:
    parts = []
    for r in ranked[:limit]:
        resume = r.resume
        lines = [f"Candidate: {resume.name}", f"Role: {resume.role or 'Not specified'}"]
        if resume.skills:
            lines.append(f"Skills: {', '.join(resume.skills)}")
        if r.chunk:
            lines.append(f"Resume Content:\n{r.chunk}")
        parts.append("\n".join(lines))
    return "\n\n---\n\n".join(parts)


def _is_unhelpful(answer: str) -> bool:
    bad = {"i don't know", "i do not know", "insufficient", "not found", "no information"}
    lower = answer.lower()
    return any(p in lower for p in bad)


def run_worker():
    """Run the RQ worker process."""
    logger.info("Starting worker...")
    
    redis_conn = get_redis_connection()
    queue = get_queue()
    
    # Create worker with default queue
    worker = Worker([queue], connection=redis_conn)
    
    logger.info(f"Worker listening on queue: {queue.name}")
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    run_worker()
