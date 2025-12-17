from __future__ import annotations

from typing import List, TypedDict

from langgraph.graph import END, StateGraph

try:
    from .embedder import ResumeEmbedder
    from .llm import ResumeLLM
    from .models import Resume
    from .vectorstore import ResumeVectorStore
except ImportError:
    from embedder import ResumeEmbedder  # type: ignore
    from llm import ResumeLLM  # type: ignore
    from models import Resume  # type: ignore
    from vectorstore import ResumeVectorStore  # type: ignore


class AddResumeState(TypedDict, total=False):
    resume: Resume
    chunks: List[str]
    embeddings: List[List[float]]


def build_add_resume_graph(embedder: ResumeEmbedder, store: ResumeVectorStore):
    graph = StateGraph(AddResumeState)

    def chunk(state: AddResumeState):
        document = _combine_fields(state["resume"])
        chunks = _chunk_text(document)
        return {"chunks": chunks}

    def embed(state: AddResumeState):
        embeddings = embedder.embed_many(state["chunks"])
        return {"embeddings": embeddings}

    def persist(state: AddResumeState):
        store.add_resume_chunks(state["resume"], state["embeddings"], state["chunks"])
        return {}

    graph.add_node("chunk", chunk)
    graph.add_node("embed", embed)
    graph.add_node("persist", persist)

    graph.set_entry_point("chunk")
    graph.add_edge("chunk", "embed")
    graph.add_edge("embed", "persist")
    graph.add_edge("persist", END)

    return graph.compile()


class RefreshResumeState(TypedDict, total=False):
    resume: Resume
    extracted: dict
    chunks: List[str]
    embeddings: List[List[float]]


def build_refresh_resume_graph(
    resume_llm: ResumeLLM, embedder: ResumeEmbedder, store: ResumeVectorStore
):
    graph = StateGraph(RefreshResumeState)

    def extract(state: RefreshResumeState):
        text = state.get("resume", {}).experience or state.get("resume", {}).summary or ""
        extracted = resume_llm.extract_resume_fields(text) if text else {}
        return {"extracted": extracted}

    def merge(state: RefreshResumeState):
        resume = state["resume"]
        data = state.get("extracted", {}) or {}
        updated = Resume(
            id=resume.id,
            name=(data.get("name") or resume.name or "Unknown").strip(),
            email=(data.get("email") or resume.email or "unknown@example.com").strip(),
            role=(data.get("role") or resume.role),
            skills=_deduplicate_skills(data.get("skills") or resume.skills),
            experience=(data.get("experience") or resume.experience),
            summary=(data.get("summary") or resume.summary),
        )
        return {"resume": updated}

    def chunk(state: RefreshResumeState):
        document = _combine_fields(state["resume"])
        chunks = _chunk_text(document)
        return {"chunks": chunks}

    def embed(state: RefreshResumeState):
        embeddings = embedder.embed_many(state["chunks"])
        return {"embeddings": embeddings}

    def persist(state: RefreshResumeState):
        store.delete(state["resume"].id)
        store.add_resume_chunks(state["resume"], state["embeddings"], state["chunks"])
        return {}

    graph.add_node("extract", extract)
    graph.add_node("merge", merge)
    graph.add_node("chunk", chunk)
    graph.add_node("embed", embed)
    graph.add_node("persist", persist)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "merge")
    graph.add_edge("merge", "chunk")
    graph.add_edge("chunk", "embed")
    graph.add_edge("embed", "persist")
    graph.add_edge("persist", END)

    return graph.compile()


def _combine_fields(resume: Resume) -> str:
    parts = [
        resume.name,
        resume.email,
        resume.role or "",
        ", ".join(resume.skills) if resume.skills else "",
        resume.experience,
        resume.summary,
    ]
    return "\n".join(part for part in parts if part).strip()


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    clean = (text or "").strip()
    if not clean:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(clean):
        end = start + chunk_size
        chunk = clean[start:end]
        chunks.append(chunk)
        if end >= len(clean):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def _deduplicate_skills(skills: List[str]) -> List[str]:
    seen = set()
    cleaned: List[str] = []
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
