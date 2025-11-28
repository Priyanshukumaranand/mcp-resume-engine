from __future__ import annotations

import os
from typing import List, Sequence

from fastapi import FastAPI, HTTPException

from embedder import ResumeEmbedder
from models import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendationResult,
    Resume,
    ResumeListResponse,
)
from vectorstore import ResumeVectorStore


APP_NAME = "Hackathon Teammate Recommendation API"
PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "chroma_storage")

app = FastAPI(title=APP_NAME, version="1.0.0")
embedder = ResumeEmbedder()
vector_store = ResumeVectorStore(persist_directory=PERSIST_DIRECTORY)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/add_resume")
async def add_resume(resume: Resume) -> dict:
    document = vector_store.build_document(resume)
    embedding = embedder.embed_text(document)
    vector_store.add_resume(resume, embedding, document)
    return {"status": "created", "id": resume.id}


@app.get("/resumes", response_model=ResumeListResponse)
async def list_resumes() -> ResumeListResponse:
    resumes = vector_store.get_all_resumes()
    return ResumeListResponse(resumes=resumes)


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_teammate(request: RecommendationRequest) -> RecommendationResponse:
    all_resumes = vector_store.get_all_resumes()
    if not all_resumes:
        raise HTTPException(status_code=404, detail="No resumes stored")

    query_text = vector_store.build_query_text(request)
    query_embedding = embedder.embed_text(query_text)
    candidates = vector_store.query(query_embedding, top_k=request.top_k)

    if not candidates:
        raise HTTPException(status_code=404, detail="No matching candidates")

    ranked = _rerank_candidates(candidates, request)
    best = ranked[0]
    response = RecommendationResponse(
        best_match=best,
        considered=len(ranked),
    )
    return response


def _rerank_candidates(
    candidates: Sequence[tuple[Resume, float]], request: RecommendationRequest
) -> List[RecommendationResult]:
    request_skill_set = {skill.lower().strip() for skill in request.skills if skill.strip()}
    ranked: List[RecommendationResult] = []

    for resume, similarity in candidates:
        candidate_skills = {skill.lower().strip(): skill for skill in resume.skills}
        matches = [cand for key, cand in candidate_skills.items() if key in request_skill_set]
        skill_score = (len(matches) / max(len(request_skill_set), 1)) if request_skill_set else 0
        role_score = 1.0 if resume.role.lower() == request.role.lower() else 0
        final_score = (similarity * 0.6) + (skill_score * 0.3) + (role_score * 0.1)
        explanation = _build_explanation(resume, final_score, matches, role_score)
        ranked.append(
            RecommendationResult(
                candidate=resume,
                match_score=round(final_score, 4),
                matching_skills=matches,
                explanation=explanation,
            )
        )

    ranked.sort(key=lambda result: result.match_score, reverse=True)
    return ranked


def _build_explanation(
    resume: Resume, score: float, matches: List[str], role_score: float
) -> str:
    parts = [
        f"Score {score:.2f}",
        f"role match: {'exact' if role_score == 1 else 'different'}",
        f"skill overlap: {', '.join(matches) if matches else 'none'}",
    ]
    return "; ".join(parts)