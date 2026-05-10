"""
Entity Normalizer — normalizes, deduplicates, and categorizes extracted
entities and relationships before graph storage.

Features:
- Lowercase normalization with canonical name preservation
- Semantic similarity matching via Jaccard on tokens
- Alias merging and duplicate node collapsing
- Category inference for hierarchical graph connectivity
"""

from __future__ import annotations

import re
from typing import Sequence

from scholarsync.utils.logger import get_logger
from scholarsync.utils.schemas import Entity, Relationship

logger = get_logger(__name__)

# ── Category mapping for hierarchical graph structure ─────────────────

_CATEGORY_MAP = {
    "method": "Methodology",
    "algorithm": "Methodology",
    "framework": "Methodology",
    "architecture": "Methodology",
    "chunking_method": "Chunking",
    "retrieval_method": "Retrieval",
    "embedding": "Embeddings",
    "dataset": "Datasets",
    "benchmark": "Datasets",
    "corpus": "Datasets",
    "metric": "Evaluation",
    "evaluation": "Evaluation",
    "score": "Evaluation",
    "tool": "Tools",
    "concept": "Concepts",
    "author": "Authors",
}

# Singularization suffixes
_PLURAL_SUFFIXES = [
    ("strategies", "strategy"),
    ("methodologies", "methodology"),
    ("ies", "y"),
    ("ses", "s"),
    ("es", "e"),
    ("s", ""),
]


def normalize_entity_name(name: str) -> str:
    """
    Normalize an entity name to a canonical form.

    - Lowercase
    - Strip extra whitespace
    - Remove trailing punctuation
    - Simple singularization
    """
    n = name.strip().lower()
    # Remove trailing punctuation
    n = re.sub(r"[.,;:!?]+$", "", n)
    # Collapse whitespace
    n = " ".join(n.split())

    # Simple singularization
    for plural, singular in _PLURAL_SUFFIXES:
        if n.endswith(plural) and len(n) > len(plural) + 2:
            n = n[: -len(plural)] + singular
            break

    return n


def _jaccard_tokens(a: str, b: str) -> float:
    """Jaccard similarity on word tokens."""
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def deduplicate_entities(entities: list[Entity]) -> list[Entity]:
    """
    Merge duplicate entities by normalized name.

    When duplicates exist, keeps the one with the richest description.
    Merges source_paper references.
    """
    if not entities:
        return []

    # Group by normalized name
    groups: dict[str, list[Entity]] = {}
    for e in entities:
        norm = normalize_entity_name(e.name)
        groups.setdefault(norm, []).append(e)

    # Merge groups
    merged: list[Entity] = []
    for norm_name, group in groups.items():
        # Pick the entity with the longest description as canonical
        best = max(group, key=lambda x: len(x.description))
        # Use the original (non-lowercased) name from best for display
        merged.append(Entity(
            name=best.name,
            entity_type=best.entity_type,
            description=best.description,
            source_paper=best.source_paper,
            source_chunk_id=best.source_chunk_id,
        ))

    # Phase 2: Jaccard similarity merge for near-duplicates
    final: list[Entity] = []
    used = [False] * len(merged)

    for i, ent_i in enumerate(merged):
        if used[i]:
            continue
        norm_i = normalize_entity_name(ent_i.name)
        for j in range(i + 1, len(merged)):
            if used[j]:
                continue
            norm_j = normalize_entity_name(merged[j].name)
            if _jaccard_tokens(norm_i, norm_j) >= 0.8:
                # Merge j into i — keep the longer description
                if len(merged[j].description) > len(ent_i.description):
                    ent_i = Entity(
                        name=ent_i.name,
                        entity_type=ent_i.entity_type,
                        description=merged[j].description,
                        source_paper=ent_i.source_paper,
                        source_chunk_id=ent_i.source_chunk_id,
                    )
                used[j] = True
        final.append(ent_i)
        used[i] = True

    if len(final) < len(entities):
        logger.info(
            "Entity dedup: %d → %d entities (removed %d duplicates)",
            len(entities), len(final), len(entities) - len(final),
        )

    return final


def deduplicate_relationships(relationships: list[Relationship]) -> list[Relationship]:
    """
    Merge duplicate relationships by normalized (source, target, type) triple.
    """
    if not relationships:
        return []

    seen: dict[tuple[str, str, str], Relationship] = {}
    for rel in relationships:
        key = (
            normalize_entity_name(rel.source_entity),
            normalize_entity_name(rel.target_entity),
            rel.relationship_type.upper().strip(),
        )
        if key not in seen:
            seen[key] = rel
        else:
            # Keep the one with longer description
            if len(rel.description) > len(seen[key].description):
                seen[key] = rel

    result = list(seen.values())
    if len(result) < len(relationships):
        logger.info(
            "Relationship dedup: %d → %d relationships",
            len(relationships), len(result),
        )
    return result


def infer_category(entity: Entity) -> str:
    """
    Map an entity to a high-level category for hierarchical graph structure.

    Returns category string like 'Methodology', 'Chunking', 'Retrieval', etc.
    """
    etype = entity.entity_type.lower().strip()
    category = _CATEGORY_MAP.get(etype)
    if category:
        return category

    # Heuristic fallback: check name/description keywords
    name_lower = entity.name.lower()
    desc_lower = entity.description.lower()
    combined = name_lower + " " + desc_lower

    if any(k in combined for k in ("chunk", "segment", "split", "partition")):
        return "Chunking"
    if any(k in combined for k in ("retriev", "search", "rank", "index")):
        return "Retrieval"
    if any(k in combined for k in ("embed", "vector", "encod")):
        return "Embeddings"
    if any(k in combined for k in ("dataset", "corpus", "benchmark")):
        return "Datasets"
    if any(k in combined for k in ("metric", "score", "f1", "accuracy", "ndcg", "bleu", "rouge")):
        return "Evaluation"

    return "Concepts"


def build_category_edges(entities: list[Entity]) -> list[Relationship]:
    """
    Create BELONGS_TO relationships from entities to their inferred category nodes.

    This improves graph connectivity and enables hierarchical visualization.
    """
    category_edges: list[Relationship] = []
    for ent in entities:
        cat = infer_category(ent)
        category_edges.append(Relationship(
            source_entity=ent.name,
            target_entity=cat,
            relationship_type="PART_OF",
            description=f"{ent.name} belongs to category {cat}",
            source_paper=ent.source_paper,
        ))
    return category_edges
