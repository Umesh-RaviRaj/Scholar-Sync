"""
Evaluation Pipeline Orchestrator — runs all evaluation metrics on pipeline output.

Coordinates:
- Retrieval quality evaluation (nDCG, P@K, R@K, MRR, MAP)
- Generation quality evaluation (Faithfulness, Grounding, Coherence, etc.)
- Hallucination detection (claim verification against sources)
- Cross-document synthesis scoring

All metrics are COMPUTED FROM ACTUAL DATA — no heuristic self-scoring.
"""

from __future__ import annotations

import re
from typing import Optional

from scholarsync.evaluation.metrics import compute_all_metrics
from scholarsync.evaluation.hallucination_detector import (
    detect_hallucinations,
    HallucinationReport,
)
from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)


def _extract_claims_from_text(text: str) -> list[str]:
    """Extract individual claims/sentences from generated text."""
    # Split by sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out very short fragments and headers
    claims = []
    for s in sentences:
        s = s.strip()
        if len(s) < 15:
            continue
        if s.startswith("#") or s.startswith("*"):
            continue
        if s.startswith("|"):  # Table rows
            continue
        claims.append(s)
    return claims


def _estimate_relevance_scores(
    source_chunks: list[str],
    query: str,
) -> list[float]:
    """
    Estimate retrieval relevance scores from token overlap with query.
    
    This is a real computation (not LLM-generated) based on lexical match.
    """
    query_tokens = set(re.findall(r'\b\w{3,}\b', query.lower()))
    if not query_tokens:
        return [0.5] * len(source_chunks)
    
    scores = []
    for chunk in source_chunks:
        chunk_tokens = set(re.findall(r'\b\w{3,}\b', chunk.lower()))
        if not chunk_tokens:
            scores.append(0.0)
            continue
        overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
        scores.append(min(1.0, overlap * 1.5))  # Scale up slightly
    
    return scores


def evaluate_pipeline_output(
    query: str,
    generated_text: str,
    source_chunks: list[str],
    paper_titles: list[str],
) -> dict:
    """
    Run the full evaluation pipeline on pipeline output.
    
    Args:
        query: Original research query
        generated_text: The generated literature review / response
        source_chunks: Retrieved source chunks used for generation
        paper_titles: Titles of papers in the corpus
    
    Returns:
        Comprehensive evaluation dict with all metrics and hallucination report
    """
    logger.info("Running evaluation pipeline...")
    
    # Extract claims from generated text
    claims = _extract_claims_from_text(generated_text)
    
    # Compute retrieval relevance scores
    relevance_scores = _estimate_relevance_scores(source_chunks, query)
    
    # Convert to binary relevance flags (threshold = 0.3)
    relevant_flags = [score >= 0.3 for score in relevance_scores]
    total_relevant = sum(relevant_flags)
    
    # Compute all metrics
    metrics = compute_all_metrics(
        generated_text=generated_text,
        generated_claims=claims,
        source_chunks=source_chunks,
        paper_titles=paper_titles,
        retrieval_relevance_scores=relevance_scores,
        retrieval_relevant_flags=relevant_flags,
        total_relevant_in_corpus=max(total_relevant, len(source_chunks)),
    )
    
    # Run hallucination detection
    hallucination_report = detect_hallucinations(
        generated_claims=claims,
        source_chunks=source_chunks,
        paper_titles=paper_titles,
    )
    
    # Add hallucination score to metrics
    metrics["hallucination_score"] = hallucination_report.hallucination_score
    
    # Build evaluation result
    evaluation = {
        "metrics": metrics,
        "hallucination_report": {
            "total_claims": hallucination_report.total_claims,
            "supported_claims": hallucination_report.supported_claims,
            "unsupported_claims": hallucination_report.unsupported_claims,
            "uncertain_claims": hallucination_report.uncertain_claims,
            "hallucination_score": hallucination_report.hallucination_score,
            "flagged_issues": hallucination_report.flagged_issues[:10],
        },
        "summary": {
            "retrieval_quality": _summarize_retrieval(metrics),
            "generation_quality": _summarize_generation(metrics),
            "grounding_quality": _summarize_grounding(metrics, hallucination_report),
            "overall_verdict": _overall_verdict(metrics, hallucination_report),
        },
    }
    
    logger.info(
        "Evaluation complete: overall=%.3f, hallucination=%.3f, faithfulness=%.3f",
        metrics["overall_quality"],
        hallucination_report.hallucination_score,
        metrics["faithfulness"],
    )
    
    return evaluation


def _summarize_retrieval(metrics: dict) -> str:
    """Summarize retrieval quality from metrics."""
    ndcg = metrics.get("ndcg_at_10", 0)
    precision = metrics.get("precision_at_5", 0)
    diversity = metrics.get("retrieval_diversity", 0)
    
    if ndcg >= 0.7 and precision >= 0.6:
        return "Strong — highly relevant chunks ranked appropriately"
    elif ndcg >= 0.4:
        return "Moderate — some relevant chunks but ranking could improve"
    else:
        return "Weak — retrieval may not be capturing relevant content"


def _summarize_generation(metrics: dict) -> str:
    """Summarize generation quality from metrics."""
    coherence = metrics.get("semantic_coherence", 0)
    redundancy = metrics.get("redundancy_reduction", 0)
    cross_doc = metrics.get("cross_document_synthesis", 0)
    
    if coherence >= 0.7 and redundancy >= 0.8 and cross_doc >= 0.5:
        return "Strong — coherent, non-redundant, well-synthesized"
    elif coherence >= 0.5:
        return "Moderate — generally coherent but could improve synthesis"
    else:
        return "Needs improvement — may lack coherence or have redundancy"


def _summarize_grounding(metrics: dict, report: HallucinationReport) -> str:
    """Summarize grounding quality."""
    faithfulness = metrics.get("faithfulness", 0)
    hallucination = report.hallucination_score
    
    if faithfulness >= 0.8 and hallucination <= 0.1:
        return "Excellent — claims well-grounded, minimal hallucination"
    elif faithfulness >= 0.6 and hallucination <= 0.2:
        return "Good — mostly grounded, few unsupported claims"
    elif faithfulness >= 0.4:
        return "Fair — some claims lack source support"
    else:
        return "Concerning — significant portion of claims may be unsupported"


def _overall_verdict(metrics: dict, report: HallucinationReport) -> str:
    """Produce overall evaluation verdict."""
    overall = metrics.get("overall_quality", 0)
    hallucination = report.hallucination_score
    
    if overall >= 0.75 and hallucination <= 0.1:
        return "HIGH QUALITY — reliable research output"
    elif overall >= 0.6 and hallucination <= 0.2:
        return "GOOD — generally reliable with minor issues"
    elif overall >= 0.45:
        return "MODERATE — usable but verify key claims"
    else:
        return "LOW — significant quality concerns, manual review needed"
