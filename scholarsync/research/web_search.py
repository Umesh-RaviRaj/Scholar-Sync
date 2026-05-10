"""
Web Search Module — auxiliary research layer for ScholarSync.

Provides controlled web search to augment uploaded-paper knowledge with:
- Latest methodologies and techniques
- Recent benchmark updates
- Emerging evaluation methods
- Updated research trends

IMPORTANT: This does NOT replace RAG. It acts as an optional supplementary layer.

Supports:
- DuckDuckGo (free, no API key)
- Tavily (if TAVILY_API_KEY is set in env)

Prioritizes trusted research sources (arXiv, Semantic Scholar, ACL, HuggingFace, etc.)
"""

from __future__ import annotations

import re
import os
import json
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)

# ── Trusted Sources (prioritized in reranking) ─────────────────────────

TRUSTED_DOMAINS = [
    "arxiv.org",
    "aclanthology.org",
    "huggingface.co",
    "semanticscholar.org",
    "research.google",
    "anthropic.com",
    "openai.com",
    "microsoft.com/en-us/research",
    "github.com",
    "paperswithcode.com",
    "proceedings.neurips.cc",
    "proceedings.mlr.press",
    "aclweb.org",
    "dl.acm.org",
    "ieee.org",
]

BLOCKED_DOMAINS = [
    "medium.com",
    "towardsdatascience.com",
    "analyticsvidhya.com",
    "kdnuggets.com",
    "machinelearningmastery.com",
    "geeksforgeeks.org",
    "w3schools.com",
    "stackoverflow.com",
]

# ── Web Search Activation Patterns ──────────────────────────────────────

_WEB_SEARCH_TRIGGERS = re.compile(
    r"\b(latest|recent|new|current|2024|2025|2026|state.?of.?the.?art|"
    r"sota|emerging|trend|advancement|update|newest|cutting.?edge|"
    r"beyond|after|since|upcoming|modern)\b",
    re.IGNORECASE,
)


@dataclass
class WebSearchResult:
    """A single web search result."""
    title: str
    url: str
    snippet: str
    source_domain: str = ""
    trust_score: float = 0.5
    published_date: str = ""
    is_trusted: bool = False


@dataclass
class WebSearchResponse:
    """Aggregated web search response."""
    results: list[WebSearchResult] = field(default_factory=list)
    query: str = ""
    search_engine: str = ""
    timestamp: str = ""
    error: str = ""


def should_activate_web_search(query: str, paper_count: int = 0) -> bool:
    """
    Determine if web search should be activated based on query content.
    
    Activates when:
    - User explicitly asks for latest/recent information
    - Query mentions state-of-the-art or emerging techniques
    - No papers are uploaded (need external context)
    - Query compares against recent advancements
    """
    if _WEB_SEARCH_TRIGGERS.search(query):
        return True
    if paper_count == 0:
        return True
    return False


def _extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.replace("www.", "")
    except Exception:
        return ""


def _compute_trust_score(url: str, title: str) -> tuple[float, bool]:
    """Compute trust score based on source domain."""
    domain = _extract_domain(url)
    
    # Check blocked domains
    for blocked in BLOCKED_DOMAINS:
        if blocked in domain:
            return 0.1, False
    
    # Check trusted domains
    for trusted in TRUSTED_DOMAINS:
        if trusted in domain or trusted in url:
            return 0.95, True
    
    # Academic indicators in title
    academic_signals = ["arxiv", "paper", "proceedings", "conference", "journal", "research"]
    if any(sig in title.lower() for sig in academic_signals):
        return 0.7, False
    
    return 0.4, False


def _build_research_query(user_query: str) -> str:
    """Enhance user query for research-focused web search."""
    # Add research-specific terms if not present
    research_terms = ["research", "paper", "study", "methodology", "benchmark"]
    has_research_term = any(t in user_query.lower() for t in research_terms)
    
    if not has_research_term:
        return f"{user_query} research paper methodology"
    return user_query


# ── DuckDuckGo Search Backend ───────────────────────────────────────────

def _search_duckduckgo(query: str, max_results: int = 10) -> list[WebSearchResult]:
    """Search using DuckDuckGo (no API key required)."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("duckduckgo-search not installed. Run: pip install duckduckgo-search")
        return []
    
    results = []
    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(
                query,
                max_results=max_results,
                region="wt-wt",  # worldwide
            )
            
            for item in search_results:
                url = item.get("href", "")
                title = item.get("title", "")
                snippet = item.get("body", "")
                domain = _extract_domain(url)
                trust_score, is_trusted = _compute_trust_score(url, title)
                
                results.append(WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source_domain=domain,
                    trust_score=trust_score,
                    is_trusted=is_trusted,
                ))
    except Exception as e:
        logger.error("DuckDuckGo search failed: %s", e)
    
    return results


# ── Tavily Search Backend (higher quality, needs API key) ────────────────

def _search_tavily(query: str, max_results: int = 10) -> list[WebSearchResult]:
    """Search using Tavily API (requires TAVILY_API_KEY env variable)."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return []
    
    try:
        from tavily import TavilyClient
    except ImportError:
        logger.warning("tavily-python not installed. Run: pip install tavily-python")
        return []
    
    results = []
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_domains=TRUSTED_DOMAINS[:8],
        )
        
        for item in response.get("results", []):
            url = item.get("url", "")
            title = item.get("title", "")
            snippet = item.get("content", "")
            domain = _extract_domain(url)
            trust_score, is_trusted = _compute_trust_score(url, title)
            
            results.append(WebSearchResult(
                title=title,
                url=url,
                snippet=snippet,
                source_domain=domain,
                trust_score=trust_score,
                is_trusted=is_trusted,
                published_date=item.get("published_date", ""),
            ))
    except Exception as e:
        logger.error("Tavily search failed: %s", e)
    
    return results


# ── Main Search Function ────────────────────────────────────────────────

def web_search(
    query: str,
    max_results: int = 8,
    prefer_trusted: bool = True,
) -> WebSearchResponse:
    """
    Perform web search for research augmentation.
    
    Uses Tavily if API key is available, otherwise falls back to DuckDuckGo.
    Results are filtered and scored by trust level.
    
    Args:
        query: Research query to search for
        max_results: Maximum number of results to return
        prefer_trusted: If True, prioritize trusted academic sources
    
    Returns:
        WebSearchResponse with scored and filtered results
    """
    research_query = _build_research_query(query)
    logger.info("Web search: '%s' (enhanced: '%s')", query, research_query)
    
    # Try Tavily first (higher quality), fall back to DuckDuckGo
    results = _search_tavily(research_query, max_results=max_results + 4)
    search_engine = "tavily"
    
    if not results:
        results = _search_duckduckgo(research_query, max_results=max_results + 4)
        search_engine = "duckduckgo"
    
    if not results:
        return WebSearchResponse(
            query=query,
            search_engine=search_engine,
            timestamp=datetime.utcnow().isoformat(),
            error="No results found",
        )
    
    # Filter out blocked domains
    results = [r for r in results if r.trust_score > 0.1]
    
    # Sort by trust score (prioritize trusted sources)
    if prefer_trusted:
        results.sort(key=lambda r: r.trust_score, reverse=True)
    
    # Limit results
    results = results[:max_results]
    
    logger.info(
        "Web search returned %d results (%d trusted) via %s",
        len(results),
        sum(1 for r in results if r.is_trusted),
        search_engine,
    )
    
    return WebSearchResponse(
        results=results,
        query=query,
        search_engine=search_engine,
        timestamp=datetime.utcnow().isoformat(),
    )


def format_web_results_for_context(response: WebSearchResponse) -> str:
    """
    Format web search results as context text for LLM consumption.
    
    Clearly separates web-discovered findings from paper-grounded insights.
    """
    if not response.results:
        return ""
    
    lines = [
        "=== RECENT WEB-DISCOVERED RESEARCH (supplementary — NOT from uploaded papers) ===",
        f"Search: '{response.query}' | Source: {response.search_engine} | {response.timestamp}",
        "",
    ]
    
    for i, result in enumerate(response.results, 1):
        trust_label = "✓ TRUSTED" if result.is_trusted else "○ external"
        lines.append(f"[Web-{i}] [{trust_label}] {result.title}")
        lines.append(f"  Source: {result.source_domain} | {result.url}")
        if result.snippet:
            # Truncate long snippets
            snippet = result.snippet[:300] + "..." if len(result.snippet) > 300 else result.snippet
            lines.append(f"  Summary: {snippet}")
        lines.append("")
    
    lines.append("NOTE: Web results are supplementary. Paper-grounded claims take priority.")
    
    return "\n".join(lines)
