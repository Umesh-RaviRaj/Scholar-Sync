"""
Profile Builder — deterministic transformation of ExtractedKnowledge
into StructuredPaperProfile objects.

Zero LLM calls. Pure Python logic that reorganizes existing extractions
into structured per-paper profiles for analytical synthesis.
"""

from __future__ import annotations

import re
from typing import Sequence

from scholarsync.utils.logger import get_logger
from scholarsync.utils.schemas import (
    ExtractedKnowledge,
    StructuredPaperProfile,
    PaperMetadata,
    Flashcard,
)

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# FLASHCARD INTELLIGENCE SYSTEM — Zero-Cost Deterministic Study Card Generation
# ══════════════════════════════════════════════════════════════════════════════

def _safe_get_item(lst: list, index: int, default: str = "") -> str:
    """Safely get an item from a list by index with fallback."""
    try:
        if lst and len(lst) > index:
            item = lst[index]
            return str(item).strip() if item else default
        return default
    except (IndexError, TypeError):
        return default


def _safe_trim_text(text: str, max_length: int = 200, preserve_words: bool = True) -> str:
    """
    Safely trim text to max_length while preserving readability.
    
    Args:
        text: Input text to trim
        max_length: Maximum character length
        preserve_words: If True, avoid breaking mid-word
    
    Returns:
        Trimmed text, never empty (returns fallback if input is empty)
    """
    if not text or not text.strip():
        return "Not specified in text"
    
    text = text.strip()
    
    if len(text) <= max_length:
        return text
    
    if preserve_words:
        # Trim at last space before max_length to avoid broken words
        trimmed = text[:max_length]
        last_space = trimmed.rfind(' ')
        if last_space > max_length * 0.6:  # Only use space if it's reasonably far
            trimmed = trimmed[:last_space]
        return trimmed.rstrip('.,;:') + "..."
    
    return text[:max_length - 3] + "..."


def _extract_summary_from_extractions(
    findings: list[str],
    claims: list[str],
    methodology: list[str],
) -> str:
    """
    Build a research objective summary from available extraction data.
    Prioritizes claims about research goals, then findings, then methodology.
    """
    # Look for objective-indicating phrases in claims
    objective_keywords = ("address", "propose", "introduce", "present", "develop", "aim", "goal", "objective")
    for claim in claims:
        claim_lower = claim.lower()
        if any(kw in claim_lower for kw in objective_keywords):
            return _safe_trim_text(claim, 200)
    
    # Fall back to first significant finding
    if findings:
        return _safe_trim_text(findings[0], 200)
    
    # Fall back to methodology description
    if methodology:
        return _safe_trim_text(methodology[0], 200)
    
    return "Not specified in text"


def generate_flashcards(
    extraction: ExtractedKnowledge,
    all_findings: list[str] | None = None,
    all_claims: list[str] | None = None,
    all_methodology: list[str] | None = None,
    all_risks: list[str] | None = None,
) -> list[Flashcard]:
    """
    Generate exactly 5 deterministic study flashcards from extracted knowledge.
    
    ZERO LLM COST — Pure Python transformation of existing structured data.
    
    Flashcard Structure:
        1. Research Objective — What problem does this paper address?
        2. Key Finding I — Primary discovery/result
        3. Key Finding II — Secondary discovery/result
        4. Methodology — How was the research conducted?
        5. Critical Takeaway — Most important insight or risk
    
    Args:
        extraction: ExtractedKnowledge object for a single paper
        all_findings: Optional aggregated findings (for multi-extraction papers)
        all_claims: Optional aggregated claims
        all_methodology: Optional aggregated methodology
        all_risks: Optional aggregated risks
    
    Returns:
        Exactly 5 Flashcard objects (never fails, uses fallbacks)
    """
    # Use provided aggregated data or fall back to extraction fields
    findings = all_findings if all_findings else extraction.findings
    claims = all_claims if all_claims else extraction.claims
    methodology = all_methodology if all_methodology else extraction.methodology
    risks = all_risks if all_risks else extraction.risks
    
    paper_id = extraction.paper_id
    paper_title = extraction.paper_title or "Unknown Paper"
    
    flashcards: list[Flashcard] = []
    
    # ── Flashcard 1: Research Objective ──────────────────────────────────
    objective_text = _extract_summary_from_extractions(findings, claims, methodology)
    flashcards.append(Flashcard(
        front=f"Research Objective: {paper_title[:50]}{'...' if len(paper_title) > 50 else ''}",
        back=objective_text,
        category="objective",
        source_paper=paper_id,
    ))
    
    # ── Flashcard 2: Key Finding I ───────────────────────────────────────
    finding_1 = _safe_get_item(findings, 0, "Not specified in text")
    flashcards.append(Flashcard(
        front="Key Finding I",
        back=_safe_trim_text(finding_1, 250),
        category="finding",
        source_paper=paper_id,
    ))
    
    # ── Flashcard 3: Key Finding II ──────────────────────────────────────
    finding_2 = _safe_get_item(findings, 1, "Not specified in text")
    flashcards.append(Flashcard(
        front="Key Finding II",
        back=_safe_trim_text(finding_2, 250),
        category="finding",
        source_paper=paper_id,
    ))
    
    # ── Flashcard 4: Methodology ─────────────────────────────────────────
    method_text = _safe_get_item(methodology, 0, "Not specified in text")
    flashcards.append(Flashcard(
        front="Methodology",
        back=_safe_trim_text(method_text, 250),
        category="methodology",
        source_paper=paper_id,
    ))
    
    # ── Flashcard 5: Critical Takeaway ───────────────────────────────────
    # Priority: first claim > first risk > fallback
    takeaway = _safe_get_item(claims, 0, "")
    if not takeaway:
        takeaway = _safe_get_item(risks, 0, "Not specified in text")
    if not takeaway:
        takeaway = "Not specified in text"
    
    flashcards.append(Flashcard(
        front="Critical Takeaway",
        back=_safe_trim_text(takeaway, 250),
        category="takeaway",
        source_paper=paper_id,
    ))
    
    # Guarantee exactly 5 cards
    assert len(flashcards) == 5, f"Flashcard generation failed: expected 5, got {len(flashcards)}"
    
    logger.info(
        "Generated %d flashcards for paper '%s' (ID: %s)",
        len(flashcards),
        paper_title[:40] if paper_title else "Unknown",
        paper_id[:8] if paper_id else "unknown"
    )
    
    return flashcards

# ── Keyword matchers for classification ──────────────────────────────

_CHUNKING_KEYWORDS = re.compile(
    r"\b(chunk\w*|segment\w*|split\w*|partition\w*|window\w*|token.?split|recursive|semantic.?chunk\w*|"
    r"late.?chunk\w*|hichunk\w*|overlap\w*|fixed.?size|sentence.?split|paragraph)",
    re.IGNORECASE,
)

_RETRIEVAL_KEYWORDS = re.compile(
    r"\b(retriev\w*|search\w*|rank\w*|bm25|dense.?retriev\w*|hybrid.?retriev\w*|re.?rank\w*|"
    r"auto.?merg\w*|fusion|colbert|contextual.?retriev\w*|vector.?search|"
    r"similarity|nearest.?neighbor|knn|ann|faiss)",
    re.IGNORECASE,
)

_EMBEDDING_KEYWORDS = re.compile(
    r"\b(embed|bge|mxbai|openai.?embed|sentence.?transform|e5|"
    r"instructor|ada|cohere.?embed|all.?minilm|gte|nomic|jina)\b",
    re.IGNORECASE,
)

_DATASET_TYPES = {"dataset", "benchmark", "corpus"}
_METRIC_TYPES = {"metric", "evaluation", "score"}
_METHOD_TYPES = {"method", "algorithm", "framework", "architecture", "model"}

_SCALABILITY_KEYWORDS = re.compile(
    r"\b(scal|large.?scale|distributed|parallel|latency|throughput|"
    r"efficiency|memory|gpu|computation|overhead|cost|resource)\b",
    re.IGNORECASE,
)


def _classify_methodology_text(text: str) -> str:
    """Determine if a methodology string relates to chunking, retrieval, or general."""
    if _CHUNKING_KEYWORDS.search(text):
        return "chunking"
    if _RETRIEVAL_KEYWORDS.search(text):
        return "retrieval"
    return "general"


def _extract_embedding_models(entities: list) -> list[str]:
    """Pull embedding model names from entities."""
    models = []
    for e in entities:
        name_lower = e.name.lower()
        if _EMBEDDING_KEYWORDS.search(name_lower) or _EMBEDDING_KEYWORDS.search(e.description):
            models.append(e.name)
    return models


def build_paper_profiles(
    extractions: list[ExtractedKnowledge],
    paper_metadata: list[PaperMetadata],
) -> list[StructuredPaperProfile]:
    """
    Build one StructuredPaperProfile per paper from existing extractions.

    Groups all extractions by paper_id, then deterministically maps
    extracted fields into the structured profile schema.
    """
    # Group extractions by paper
    by_paper: dict[str, list[ExtractedKnowledge]] = {}
    for ext in extractions:
        by_paper.setdefault(ext.paper_id, []).append(ext)

    meta_map = {m.paper_id: m for m in paper_metadata}
    profiles: list[StructuredPaperProfile] = []

    for paper_id, exts in by_paper.items():
        meta = meta_map.get(paper_id)
        title = meta.title if meta else exts[0].paper_title

        # Aggregate all fields across extractions for this paper
        all_entities = []
        all_methodology = []
        all_findings = []
        all_risks = []
        all_claims = []

        for ext in exts:
            all_entities.extend(ext.entities)
            all_methodology.extend(ext.methodology)
            all_findings.extend(ext.findings)
            all_risks.extend(ext.risks)
            all_claims.extend(ext.claims)

        # ── Classify methodology items ───────────────────────────
        chunking_parts = []
        retrieval_parts = []
        general_method_parts = []

        for m in all_methodology:
            cat = _classify_methodology_text(m)
            if cat == "chunking":
                chunking_parts.append(m)
            elif cat == "retrieval":
                retrieval_parts.append(m)
            else:
                general_method_parts.append(m)

        # ── Extract typed entities ───────────────────────────────
        datasets = []
        metrics = []
        embedding_models = _extract_embedding_models(all_entities)

        for e in all_entities:
            etype = e.entity_type.lower()
            if etype in _DATASET_TYPES or "dataset" in e.name.lower():
                datasets.append(e.name)
            elif etype in _METRIC_TYPES or "metric" in e.description.lower():
                metrics.append(e.name)

        # ── Derive advantages from findings/claims ───────────────
        advantages = []
        key_contributions = []
        for f in all_findings:
            if any(w in f.lower() for w in ("improv", "better", "outperform", "superior", "achiev", "state-of-the-art")):
                advantages.append(f)
            else:
                key_contributions.append(f)

        for c in all_claims:
            if any(w in c.lower() for w in ("novel", "first", "contribut", "propos", "introduc")):
                key_contributions.append(c)
            elif any(w in c.lower() for w in ("advantage", "benefit", "efficient", "fast", "improv")):
                advantages.append(c)

        # ── Derive limitations and scalability from risks ────────
        limitations = []
        computational_cost = ""
        scalability_notes = ""

        for r in all_risks:
            if _SCALABILITY_KEYWORDS.search(r):
                if any(w in r.lower() for w in ("cost", "gpu", "memory", "computation", "resource", "overhead")):
                    computational_cost = r if not computational_cost else computational_cost + "; " + r
                else:
                    scalability_notes = r if not scalability_notes else scalability_notes + "; " + r
            else:
                limitations.append(r)

        # ── Research problem from claims ─────────────────────────
        research_problem = ""
        for c in all_claims:
            if any(w in c.lower() for w in ("address", "problem", "challenge", "gap", "issue", "solve")):
                research_problem = c
                break

        # ── Best use case heuristic ──────────────────────────────
        best_use_case = ""
        for c in all_claims + all_findings:
            if any(w in c.lower() for w in ("suited for", "best for", "ideal for", "recommend", "applicable to", "designed for")):
                best_use_case = c
                break

        # ── Generate Flashcards (Flashcard Intelligence System) ──
        # Zero-cost deterministic generation from aggregated extraction data
        paper_flashcards: list[Flashcard] = []
        try:
            # Use the first extraction as base, with aggregated data
            base_extraction = exts[0] if exts else ExtractedKnowledge(
                subtask_type=exts[0].subtask_type if exts else "entities",
                paper_id=paper_id,
                paper_title=title,
            )
            paper_flashcards = generate_flashcards(
                extraction=base_extraction,
                all_findings=all_findings,
                all_claims=all_claims,
                all_methodology=all_methodology,
                all_risks=all_risks,
            )
            logger.info("✅ Flashcards integrated into profile for '%s'", title[:40])
        except Exception as e:
            logger.error("❌ Flashcard generation failed for %s: %s", paper_id[:8], e)
            # Provide fallback empty flashcards to maintain structure
            paper_flashcards = [
                Flashcard(front="Research Objective", back="Not available", category="objective", source_paper=paper_id),
                Flashcard(front="Key Finding I", back="Not available", category="finding", source_paper=paper_id),
                Flashcard(front="Key Finding II", back="Not available", category="finding", source_paper=paper_id),
                Flashcard(front="Methodology", back="Not available", category="methodology", source_paper=paper_id),
                Flashcard(front="Critical Takeaway", back="Not available", category="takeaway", source_paper=paper_id),
            ]

        profile = StructuredPaperProfile(
            paper_id=paper_id,
            paper_title=title,
            research_problem=research_problem,
            methodology="; ".join(general_method_parts[:5]) if general_method_parts else "",
            chunking_strategy="; ".join(chunking_parts[:3]) if chunking_parts else "",
            retrieval_strategy="; ".join(retrieval_parts[:3]) if retrieval_parts else "",
            embedding_models=list(set(embedding_models))[:5],
            datasets=list(set(datasets))[:8],
            evaluation_metrics=list(set(metrics))[:8],
            advantages=advantages[:6],
            limitations=limitations[:6],
            computational_cost=computational_cost[:200],
            scalability=scalability_notes[:200],
            best_use_case=best_use_case[:200],
            key_contributions=key_contributions[:6],
            flashcards=paper_flashcards,
        )
        profiles.append(profile)

    total_flashcards = sum(len(p.flashcards) for p in profiles)
    logger.info(
        "Profile builder: created %d profiles with %d flashcards (5 per paper)",
        len(profiles),
        total_flashcards,
    )
    return profiles
