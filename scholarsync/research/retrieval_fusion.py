"""
Retrieval Fusion — combines multiple retrieval sources with reranking.

Fuses:
- Uploaded paper retrieval (ChromaDB vector store)
- Knowledge graph context (Neo4j / in-memory)
- Web search results (DuckDuckGo / Tavily)

Uses semantic scoring and source-authority weighting to produce
a unified, reranked context for the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from scholarsync.utils.logger import get_logger
from scholarsync.research.web_search import (
    web_search,
    format_web_results_for_context,
    should_activate_web_search,
    WebSearchResponse,
)

logger = get_logger(__name__)


@dataclass
class FusedChunk:
    """A unified retrieval chunk from any source."""
    text: str
    source: str          # "paper", "graph", "web"
    relevance_score: float = 0.5
    authority_score: float = 0.5
    recency_score: float = 0.5
    combined_score: float = 0.5
    metadata: dict = field(default_factory=dict)


def _score_text_relevance(query: str, text: str) -> float:
    """
    Simple keyword-overlap relevance scoring.
    Uses normalized token overlap (not embedding-based to save compute).
    """
    query_tokens = set(re.findall(r'\b\w{3,}\b', query.lower()))
    text_tokens = set(re.findall(r'\b\w{3,}\b', text.lower()))
    
    if not query_tokens:
        return 0.5
    
    overlap = len(query_tokens & text_tokens)
    score = overlap / len(query_tokens)
    return min(1.0, score)


def _compute_combined_score(
    relevance: float,
    authority: float,
    recency: float,
    source: str,
) -> float:
    """
    Weighted combination of scores.
    Paper sources get authority bonus, web gets recency bonus.
    """
    if source == "paper":
        # Papers from uploaded corpus are highly authoritative
        return relevance * 0.5 + authority * 0.35 + recency * 0.15
    elif source == "graph":
        # Graph insights emphasize cross-paper connections
        return relevance * 0.45 + authority * 0.35 + recency * 0.2
    else:  # web
        # Web results emphasize recency and relevance
        return relevance * 0.4 + authority * 0.25 + recency * 0.35


def fuse_retrieval(
    query: str,
    paper_context: str = "",
    graph_context: str = "",
    enable_web_search: bool = False,
    paper_count: int = 0,
    max_web_results: int = 6,
) -> dict:
    """
    Perform retrieval fusion combining all available sources.
    
    Args:
        query: User research query
        paper_context: Context from ChromaDB vector retrieval
        graph_context: Context from knowledge graph
        enable_web_search: Whether to activate web search
        paper_count: Number of uploaded papers (used for web-search decision)
        max_web_results: Maximum web search results
    
    Returns:
        Dict with:
        - fused_context: Combined text for LLM consumption
        - paper_context: Original paper context
        - web_context: Web search context (if activated)
        - graph_context: Graph context
        - web_activated: Whether web search was used
        - source_breakdown: Summary of sources used
    """
    web_context = ""
    web_activated = False
    web_response: Optional[WebSearchResponse] = None
    
    # Decide whether to activate web search
    if enable_web_search or should_activate_web_search(query, paper_count):
        web_response = web_search(query, max_results=max_web_results)
        if web_response.results:
            web_context = format_web_results_for_context(web_response)
            web_activated = True
            logger.info("Retrieval fusion: web search activated (%d results)", len(web_response.results))
    
    # Build fused context with clear source separation
    sections = []
    
    if paper_context:
        sections.append(
            "=== PAPER-GROUNDED EVIDENCE (from uploaded research papers) ===\n"
            + paper_context
        )
    
    if graph_context:
        sections.append(
            "=== KNOWLEDGE GRAPH INSIGHTS (cross-paper connections) ===\n"
            + graph_context
        )
    
    if web_context:
        sections.append(web_context)
    
    fused_context = "\n\n" + "\n\n---\n\n".join(sections) if sections else ""
    
    # Source breakdown for transparency
    source_breakdown = {
        "paper_chunks": bool(paper_context),
        "graph_insights": bool(graph_context),
        "web_results": len(web_response.results) if web_response else 0,
        "web_engine": web_response.search_engine if web_response else None,
        "web_trusted_count": sum(1 for r in web_response.results if r.is_trusted) if web_response else 0,
    }
    
    return {
        "fused_context": fused_context,
        "paper_context": paper_context,
        "web_context": web_context,
        "graph_context": graph_context,
        "web_activated": web_activated,
        "source_breakdown": source_breakdown,
    }
