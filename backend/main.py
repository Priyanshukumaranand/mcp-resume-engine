from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException

try:  # Support running as package or as a plain script
    from .embedder import ResumeEmbedder
    from .graph import build_add_resume_graph, build_recommendation_graph
    from .models import (
        QARequest,
        QAResponse,
        QASource,
        RecommendationRequest,
        RecommendationResponse,
        Resume,
        ResumeListResponse,
    )
    from .llm import ResumeLLM
    from .reranker import RecommendationRanker
    from .vectorstore import ResumeVectorStore
except ImportError:
    from embedder import ResumeEmbedder  # type: ignore
    from graph import build_add_resume_graph, build_recommendation_graph  # type: ignore
    from models import (  # type: ignore
        QARequest,
        QAResponse,
        QASource,
        RecommendationRequest,
        RecommendationResponse,
        Resume,
        ResumeListResponse,
    )
    from llm import ResumeLLM  # type: ignore
    from reranker import RecommendationRanker  # type: ignore
    from vectorstore import ResumeVectorStore  # type: ignore


APP_NAME = "Hackathon Teammate Recommendation API"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIRECTORY = PROJECT_ROOT / "chroma_storage"

app = FastAPI(title=APP_NAME, version="1.0.0")
embedder = ResumeEmbedder()
vector_store = ResumeVectorStore(persist_directory=str(PERSIST_DIRECTORY))
resume_llm = ResumeLLM()
ranker = RecommendationRanker()
add_resume_graph = build_add_resume_graph(embedder, vector_store)
recommendation_graph = build_recommendation_graph(embedder, vector_store, ranker)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/add_resume")
async def add_resume(resume: Resume) -> dict:
    enriched = _ensure_resume_fields(resume)
    add_resume_graph.invoke({"resume": enriched})
    return {"status": "created", "id": enriched.id}


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


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_teammate(request: RecommendationRequest) -> RecommendationResponse:
    if not vector_store.has_resumes():
        raise HTTPException(status_code=404, detail="No resumes stored")

    state = recommendation_graph.invoke({"request": request})
    ranked = state.get("ranked", [])
    if not ranked:
        raise HTTPException(status_code=404, detail="No matching candidates")
    best = ranked[0]
    response = RecommendationResponse(
        best_match=best,
        considered=len(ranked),
    )
    return response


@app.post("/qa", response_model=QAResponse)
async def answer_question(request: QARequest) -> QAResponse:
    if not vector_store.has_resumes():
        raise HTTPException(status_code=404, detail="No resumes stored")

    question = request.question.strip()
    query_embedding = embedder.embed_text(question)
    candidates = vector_store.query(query_embedding, top_k=request.top_k)
    if not candidates:
        raise HTTPException(status_code=404, detail="No relevant resumes found")

    matches = _rerank_qa_candidates(question, candidates, vector_store, ranker)
    sources = [_build_qa_source(match) for match in matches]
    context = _build_qa_context(matches)
    try:
        answer = resume_llm.answer_question(question, context).strip()
    except RuntimeError:
        answer = ""

    requested_skills = getattr(request, "skills", []) or []
    if not answer or answer.lower() in {"i do not know", "i don't know"}:
        answer = _build_rule_based_answer(question, matches, requested_skills)
    if not answer:
        answer = "I can't find out."
    return QAResponse(answer=answer, sources=sources)


def _ensure_resume_fields(resume: Resume) -> Resume:
    needs_role = not (resume.role and resume.role.strip())
    needs_skills = len(resume.skills) == 0

    if not needs_role and not needs_skills:
        return resume

    inference_text = "\n".join(part for part in [resume.experience, resume.summary] if part)
    inferred_role, inferred_skills = resume_llm.infer_role_and_skills(inference_text)

    update: dict = {}
    if needs_role:
        update["role"] = inferred_role or "Hackathon Contributor"
    if needs_skills:
        update["skills"] = _deduplicate_skills(inferred_skills) or ["teamwork"]

    enriched = resume.model_copy(update=update)
    if not enriched.role:
        enriched = enriched.model_copy(update={"role": "Hackathon Contributor"})
    if not enriched.skills:
        enriched = enriched.model_copy(update={"skills": ["teamwork"]})
    return enriched


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
    document: str
    similarity: float
    cross_score: float
    score: float


def _rerank_qa_candidates(
    question: str,
    candidates: List[tuple[Resume, float]],
    store: ResumeVectorStore,
    ranker: RecommendationRanker | None,
) -> List[QAMatch]:
    matches: List[QAMatch] = []
    for resume, similarity in candidates:
        document = store.build_document(resume)
        cross_score = ranker.score(question, document) if ranker else 0.0
        combined = (similarity * 0.6) + (cross_score * 0.4)
        matches.append(
            QAMatch(
                resume=resume,
                document=document,
                similarity=similarity,
                cross_score=cross_score,
                score=combined,
            )
        )
    matches.sort(key=lambda match: match.score, reverse=True)
    return matches


def _build_qa_source(match: QAMatch) -> QASource:
    resume = match.resume
    excerpt = (resume.summary or match.document)[:400]
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
            match.document,
        ]
        chunks.append("\n".join(parts))
    return "\n---\n".join(chunks)


def _build_rule_based_answer(
    question: str,
    matches: List[QAMatch],
    requested_skills: List[str],
) -> str:
    if not matches:
        return ""

    normalized_requested = {skill.lower().strip() for skill in requested_skills if skill.strip()}
    if normalized_requested:
        skill_hits = []
        for match in matches:
            resume_skills = {skill.lower().strip() for skill in match.resume.skills}
            overlap = normalized_requested & resume_skills
            if overlap:
                skill_hits.append((match, overlap))
        if not skill_hits:
            return "Nobody has those skills."
        matches = [skill_hits[0][0]] + [m for m, _ in skill_hits[1:]]

    top = matches[0].resume
    role = top.role or "Unknown role"
    skills = ", ".join(top.skills[:5]) if top.skills else "no specific skills listed"
    summary = (top.summary or top.experience or "").strip()
    if summary:
        summary = summary[:320] + ("…" if len(summary) > 320 else "")

    parts = [
        f"Based on the stored resumes, {top.name} ({role}) seems most relevant to '{question}'.",
        f"Key skills: {skills}.",
    ]
    if summary:
        parts.append(f"Highlights: {summary}")
    return " ".join(parts).strip()