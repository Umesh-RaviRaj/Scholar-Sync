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
)

logger = get_logger(__name__)

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
        )
        profiles.append(profile)

    logger.info("Profile builder: created %d structured profiles", len(profiles))
    return profiles
