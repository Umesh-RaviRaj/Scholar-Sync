"""
Deduplication Utility — removes semantically duplicate insights before
synthesis to improve information density and reduce repetition.

Zero LLM calls. Uses string normalization + Jaccard similarity.
"""

from __future__ import annotations


def _normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return " ".join(text.lower().strip().split())


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def deduplicate_insights(items: list[str], threshold: float = 0.75) -> list[str]:
    """
    Remove near-duplicate strings from a list.

    Strategy:
    1. Exact normalized match → drop duplicate
    2. Substring containment → keep the longer one
    3. Jaccard similarity > threshold → keep the longer one

    Parameters
    ----------
    items : list[str]
        Raw list of insights/findings/claims.
    threshold : float
        Jaccard similarity threshold for merging (default 0.75).

    Returns
    -------
    list[str]
        Deduplicated list preserving order of first occurrence.
    """
    if not items:
        return []

    # Phase 1: exact dedup
    seen_normalized: set[str] = set()
    phase1: list[str] = []
    for item in items:
        norm = _normalize(item)
        if norm and norm not in seen_normalized:
            seen_normalized.add(norm)
            phase1.append(item)

    # Phase 2: substring containment
    phase2: list[str] = []
    for i, item in enumerate(phase1):
        norm_i = _normalize(item)
        is_subset = False
        for j, other in enumerate(phase1):
            if i == j:
                continue
            norm_j = _normalize(other)
            if norm_i in norm_j and len(norm_j) > len(norm_i):
                is_subset = True
                break
        if not is_subset:
            phase2.append(item)

    # Phase 3: Jaccard similarity dedup
    token_sets = [set(_normalize(item).split()) for item in phase2]
    keep = [True] * len(phase2)

    for i in range(len(phase2)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(phase2)):
            if not keep[j]:
                continue
            sim = _jaccard(token_sets[i], token_sets[j])
            if sim >= threshold:
                # Keep the longer (more informative) one
                if len(phase2[j]) > len(phase2[i]):
                    keep[i] = False
                    break
                else:
                    keep[j] = False

    result = [item for item, k in zip(phase2, keep) if k]
    return result
