"""Multihop retrieval for complex resume queries.

Implements iterative retrieval with query decomposition
to handle queries requiring multiple search passes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional, Sequence

try:
    from ..llm import ResumeLLM
except ImportError:
    ResumeLLM = None


DECOMPOSITION_PROMPT = """Break down this question into simpler sub-questions for searching resumes.
Each sub-question should target ONE specific aspect (skill, experience type, role, etc.).
Return a JSON array of sub-questions. If the query is already simple, return an array with just the original question.

Examples:
- "Who has Python and ML experience?" → ["Who has Python skills?", "Who has ML experience?"]
- "Find React developers with backend skills" → ["Who knows React?", "Who has backend experience?"]
- "Who knows Java?" → ["Who knows Java?"]

Question: {question}

Sub-questions (JSON array only, no other text):"""


@dataclass
class HopResult:
    """Result from a single retrieval hop."""
    resume: Any
    chunk: str
    similarity: float
    section_type: str
    hop_index: int = 0


@dataclass
class MultihopResult:
    """Result from multihop retrieval."""
    candidates: List[HopResult]
    hops_executed: int
    sub_queries: List[str]
    all_hop_results: List[List[HopResult]] = field(default_factory=list)
    
    @property
    def is_multihop(self) -> bool:
        return self.hops_executed > 1


class MultiHopRetriever:
    """Iterative multihop retrieval for complex queries.
    
    Decomposes complex queries into sub-queries, executes
    sequential retrieval passes, and intersects results.
    """
    
    def __init__(
        self,
        embedder: Any,
        vector_store: Any,
        llm: Any,
        reranker: Any = None,
        max_hops: int = 3,
    ):
        """Initialize multihop retriever.
        
        Args:
            embedder: Embedding model for query encoding
            vector_store: Vector store for similarity search
            llm: LLM for query decomposition
            reranker: Optional reranker for result refinement
            max_hops: Maximum number of retrieval hops
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker
        self.max_hops = max_hops
    
    def decompose_query(self, query: str) -> List[str]:
        """Decompose a complex query into sub-queries using LLM.
        
        Args:
            query: Original user question
            
        Returns:
            List of sub-queries (at least 1)
        """
        try:
            prompt = DECOMPOSITION_PROMPT.format(question=query)
            response = self.llm._generate(prompt, max_output_tokens=256)
            text = self.llm._extract_text(response).strip()
            
            # Clean up response
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()
            
            sub_queries = json.loads(text)
            
            if isinstance(sub_queries, list) and len(sub_queries) > 0:
                # Validate entries are strings
                sub_queries = [str(q).strip() for q in sub_queries if q]
                if sub_queries:
                    return sub_queries[:self.max_hops]
        except Exception:
            pass
        
        # Fallback: return original query
        return [query]
    
    def _retrieve_hop(
        self,
        query: str,
        top_k: int,
        candidate_ids: Optional[set] = None,
    ) -> List[HopResult]:
        """Execute a single retrieval hop.
        
        Args:
            query: Query string for this hop
            top_k: Number of results to retrieve
            candidate_ids: Optional set of resume IDs to filter results
            
        Returns:
            List of HopResult from this hop
        """
        # Embed query
        embedding = self.embedder.embed_query(query)
        
        # Query vector store
        raw_results = self.vector_store.query(
            embedding,
            top_k=top_k * 2,  # Fetch more for filtering
            apply_section_weights=True,
        )
        
        # Convert to HopResult
        hop_results = []
        for resume, chunk, similarity, section_type in raw_results:
            # Filter by candidate IDs if provided
            if candidate_ids is not None and resume.id not in candidate_ids:
                continue
            
            hop_results.append(HopResult(
                resume=resume,
                chunk=chunk,
                similarity=similarity,
                section_type=section_type,
            ))
        
        return hop_results[:top_k]
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> MultihopResult:
        """Execute multihop retrieval.
        
        Decomposes the query into sub-queries, retrieves for each,
        and intersects results to find candidates matching all criteria.
        
        Args:
            query: User question
            top_k: Number of final results
            
        Returns:
            MultihopResult with candidates and metadata
        """
        # Decompose query
        sub_queries = self.decompose_query(query)
        
        # Simple query - single hop
        if len(sub_queries) <= 1:
            hop_results = self._retrieve_hop(sub_queries[0], top_k)
            return MultihopResult(
                candidates=hop_results,
                hops_executed=1,
                sub_queries=sub_queries,
                all_hop_results=[hop_results],
            )
        
        # Multihop retrieval with intersection
        all_hop_results = []
        candidate_ids: Optional[set] = None
        
        for i, sub_query in enumerate(sub_queries[:self.max_hops]):
            # Retrieve for this hop, filtering by prior candidates
            hop_results = self._retrieve_hop(
                sub_query,
                top_k * 2,  # Get more to allow filtering
                candidate_ids=candidate_ids,
            )
            
            all_hop_results.append(hop_results)
            
            if not hop_results:
                # No results for this hop - intersection is empty
                break
            
            # Update candidate ID set (intersection)
            hop_ids = {r.resume.id for r in hop_results}
            if candidate_ids is None:
                candidate_ids = hop_ids
            else:
                candidate_ids &= hop_ids
            
            # No candidates left after intersection
            if not candidate_ids:
                break
        
        # Collect final candidates (from any hop, filtered by intersection)
        final_candidates = []
        seen_ids = set()
        
        # Prioritize results from later hops (more refined)
        for hop_results in reversed(all_hop_results):
            for result in hop_results:
                if candidate_ids and result.resume.id not in candidate_ids:
                    continue
                if result.resume.id not in seen_ids:
                    seen_ids.add(result.resume.id)
                    final_candidates.append(result)
        
        # Sort by similarity
        final_candidates.sort(key=lambda x: x.similarity, reverse=True)
        
        return MultihopResult(
            candidates=final_candidates[:top_k],
            hops_executed=len(sub_queries),
            sub_queries=sub_queries,
            all_hop_results=all_hop_results,
        )
    
    def retrieve_to_tuples(
        self,
        query: str,
        top_k: int = 10,
    ) -> Tuple[List[Tuple[Any, str, float, str]], MultihopResult]:
        """Retrieve and return in tuple format compatible with existing code.
        
        Args:
            query: User question
            top_k: Number of results
            
        Returns:
            Tuple of (candidates in tuple format, MultihopResult metadata)
        """
        result = self.retrieve(query, top_k)
        
        # Convert to tuple format: (resume, chunk, similarity, section_type)
        tuples = [
            (r.resume, r.chunk, r.similarity, r.section_type)
            for r in result.candidates
        ]
        
        return tuples, result
