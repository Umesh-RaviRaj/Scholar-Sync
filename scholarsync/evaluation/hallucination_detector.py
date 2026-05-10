"""
Hallucination Detection System — verifies claims against retrieved sources.

Detects:
- Unsupported numerical claims
- Fabricated metrics/benchmark values
- Unsupported conclusions
- Citation mismatches
- Invented comparisons
- Claims with no grounding in source material

For every important claim:
1. Retrieve supporting chunks
2. Check semantic alignment
3. Validate citation support
4. Compute confidence score
5. Mark unsupported claims

Unsupported claims are flagged, rewritten qualitatively, or removed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)


# ── Patterns for numerical claims that need verification ─────────────────

_NUMERICAL_CLAIM_PATTERN = re.compile(
    r"(?:achieve[sd]?|reach(?:es|ed)?|obtain(?:s|ed)?|report(?:s|ed)?|score[sd]?|"
    r"improve[sd]?|outperform[sd]?|exceed[sd]?|attain[sd]?|yield[sd]?)"
    r".*?"
    r"(\d+\.?\d*\s*%|\d+\.?\d*\s*(?:F1|accuracy|BLEU|ROUGE|nDCG|MRR|MAP|precision|recall))",
    re.IGNORECASE,
)

_FABRICATED_METRIC_PATTERN = re.compile(
    r"\b(\d{1,3}\.\d{1,4})\s*%?\s*"
    r"(?:F1|accuracy|BLEU|ROUGE|nDCG|MRR|MAP|precision|recall|score)\b",
    re.IGNORECASE,
)

_COMPARISON_PATTERN = re.compile(
    r"\b(?:outperforms?|surpass(?:es)?|beats?|exceeds?|superior\s+to|better\s+than|"
    r"worse\s+than|inferior\s+to|comparable\s+to|on\s+par\s+with)\b",
    re.IGNORECASE,
)


@dataclass
class ClaimVerification:
    """Verification result for a single claim."""
    claim: str
    is_supported: bool
    confidence: float  # 0.0 - 1.0
    support_type: str  # "full", "partial", "none", "uncertain"
    evidence_snippet: str = ""
    issue_type: str = ""  # "unsupported_number", "fabricated_metric", "no_source", "citation_mismatch"
    suggested_rewrite: str = ""


@dataclass
class HallucinationReport:
    """Full hallucination analysis report."""
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    uncertain_claims: int = 0
    hallucination_score: float = 0.0  # 0.0 = no hallucination, 1.0 = all hallucinated
    claim_verifications: list[ClaimVerification] = field(default_factory=list)
    flagged_issues: list[str] = field(default_factory=list)


def _tokenize_lower(text: str) -> set[str]:
    """Tokenize and lowercase for comparison."""
    return set(re.findall(r'\b\w{3,}\b', text.lower()))


def _check_numerical_support(
    claim: str,
    source_chunks: list[str],
) -> tuple[bool, str]:
    """
    Check if a numerical claim (e.g., "achieves 92.3% F1") is supported
    by the source material.
    
    Returns (is_supported, evidence_snippet)
    """
    # Extract numbers from claim
    numbers_in_claim = re.findall(r'\d+\.?\d*', claim)
    if not numbers_in_claim:
        return True, ""  # No numbers = not a numerical claim
    
    combined_sources = " ".join(source_chunks)
    
    # Check if the exact numbers appear in sources
    for num in numbers_in_claim:
        if num in combined_sources:
            # Find surrounding context
            idx = combined_sources.find(num)
            start = max(0, idx - 50)
            end = min(len(combined_sources), idx + 50)
            return True, combined_sources[start:end]
    
    return False, ""


def _compute_claim_support(
    claim: str,
    source_chunks: list[str],
) -> tuple[float, str, str]:
    """
    Compute how well a claim is supported by source chunks.
    
    Returns (confidence, support_type, evidence_snippet)
    """
    if not source_chunks:
        return 0.0, "none", ""
    
    claim_tokens = _tokenize_lower(claim)
    if not claim_tokens:
        return 1.0, "full", ""
    
    best_overlap = 0.0
    best_chunk = ""
    
    for chunk in source_chunks:
        chunk_tokens = _tokenize_lower(chunk)
        if not chunk_tokens:
            continue
        
        overlap = len(claim_tokens & chunk_tokens) / len(claim_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_chunk = chunk[:200]
    
    if best_overlap >= 0.6:
        return best_overlap, "full", best_chunk
    elif best_overlap >= 0.35:
        return best_overlap, "partial", best_chunk
    elif best_overlap >= 0.15:
        return best_overlap, "uncertain", best_chunk
    else:
        return best_overlap, "none", best_chunk


def detect_hallucinations(
    generated_claims: list[str],
    source_chunks: list[str],
    paper_titles: list[str] | None = None,
) -> HallucinationReport:
    """
    Detect hallucinations in generated claims by verifying against sources.
    
    Args:
        generated_claims: List of claims/sentences from generated output
        source_chunks: Retrieved source chunks used for generation
        paper_titles: Available paper titles for citation checking
    
    Returns:
        HallucinationReport with per-claim verification results
    """
    if not generated_claims:
        return HallucinationReport(hallucination_score=0.0)
    
    verifications: list[ClaimVerification] = []
    flagged_issues: list[str] = []
    
    for claim in generated_claims:
        claim = claim.strip()
        if len(claim) < 10:
            continue
        
        # Check for numerical claims
        has_numbers = bool(_NUMERICAL_CLAIM_PATTERN.search(claim))
        has_fabricated_metric = bool(_FABRICATED_METRIC_PATTERN.search(claim))
        has_comparison = bool(_COMPARISON_PATTERN.search(claim))
        
        # Compute support from sources
        confidence, support_type, evidence = _compute_claim_support(claim, source_chunks)
        
        issue_type = ""
        suggested_rewrite = ""
        
        # Flag unsupported numerical claims
        if has_numbers or has_fabricated_metric:
            num_supported, num_evidence = _check_numerical_support(claim, source_chunks)
            if not num_supported:
                issue_type = "unsupported_number"
                confidence = min(confidence, 0.2)
                support_type = "none"
                # Suggest qualitative rewrite
                suggested_rewrite = re.sub(
                    r'\d+\.?\d*\s*%',
                    "significant improvement",
                    claim,
                )
                flagged_issues.append(
                    f"Unsupported metric: '{claim[:80]}...' — numbers not found in sources"
                )
            else:
                evidence = num_evidence
        
        # Flag unsupported comparisons
        if has_comparison and support_type in ("none", "uncertain"):
            issue_type = issue_type or "unsupported_comparison"
            flagged_issues.append(
                f"Unsupported comparison: '{claim[:80]}...' — no source evidence"
            )
        
        # Citation mismatch check
        if paper_titles and "[" in claim:
            # Extract citation references like [1], [2]
            cited_refs = re.findall(r'\[(\d+)\]', claim)
            if cited_refs:
                # Check if claim content is actually from those papers
                if confidence < 0.3:
                    issue_type = issue_type or "citation_mismatch"
                    flagged_issues.append(
                        f"Citation mismatch: '{claim[:60]}...' — claim not grounded in cited paper"
                    )
        
        is_supported = confidence >= 0.4 and support_type in ("full", "partial")
        
        verifications.append(ClaimVerification(
            claim=claim,
            is_supported=is_supported,
            confidence=confidence,
            support_type=support_type,
            evidence_snippet=evidence,
            issue_type=issue_type,
            suggested_rewrite=suggested_rewrite,
        ))
    
    # Compute overall scores
    total = len(verifications)
    supported = sum(1 for v in verifications if v.is_supported)
    unsupported = sum(1 for v in verifications if not v.is_supported and v.support_type == "none")
    uncertain = total - supported - unsupported
    
    hallucination_score = unsupported / total if total > 0 else 0.0
    
    report = HallucinationReport(
        total_claims=total,
        supported_claims=supported,
        unsupported_claims=unsupported,
        uncertain_claims=uncertain,
        hallucination_score=round(hallucination_score, 4),
        claim_verifications=verifications,
        flagged_issues=flagged_issues,
    )
    
    logger.info(
        "Hallucination detection: %d claims — %d supported, %d unsupported, %d uncertain (score: %.3f)",
        total, supported, unsupported, uncertain, hallucination_score,
    )
    
    return report


def filter_unsupported_claims(
    claims: list[str],
    source_chunks: list[str],
    threshold: float = 0.3,
) -> tuple[list[str], list[str]]:
    """
    Filter out unsupported claims and return (kept, removed) lists.
    
    Claims below the confidence threshold are removed.
    Numerical claims without source support are rewritten qualitatively.
    
    Args:
        claims: List of claims to verify
        source_chunks: Source evidence
        threshold: Minimum confidence to keep a claim
    
    Returns:
        Tuple of (kept_claims, removed_claims)
    """
    report = detect_hallucinations(claims, source_chunks)
    
    kept = []
    removed = []
    
    for v in report.claim_verifications:
        if v.confidence >= threshold:
            if v.suggested_rewrite and v.issue_type == "unsupported_number":
                kept.append(v.suggested_rewrite)
            else:
                kept.append(v.claim)
        else:
            removed.append(v.claim)
    
    if removed:
        logger.info("Filtered %d unsupported claims (kept %d)", len(removed), len(kept))
    
    return kept, removed
