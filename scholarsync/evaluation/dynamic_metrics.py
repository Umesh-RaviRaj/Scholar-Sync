"""
Dynamic Evaluation Metrics — context-aware metric selection and display.

This module provides:
1. Query type classification for metric selection
2. Dynamic metric weighting based on context
3. Formatted metric display with explanations
4. Adaptive scoring based on response characteristics

All metrics are COMPUTED FROM ACTUAL DATA — no heuristic self-scoring.
"""

from __future__ import annotations

import re
from typing import Optional
from dataclasses import dataclass, field

from scholarsync.evaluation.metrics import compute_all_metrics
from scholarsync.evaluation.evaluator import evaluate_pipeline_output
from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# QUERY TYPE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class QueryClassification:
    """Classification of a research query for metric selection."""
    query_type: str           # "technical", "comparison", "survey", "methodology", "general"
    complexity: str           # "simple", "moderate", "complex"
    requires_citations: bool
    requires_synthesis: bool
    requires_code: bool
    research_depth: str       # "shallow", "moderate", "deep"
    confidence: float


def classify_query(query: str) -> QueryClassification:
    """
    Classify a research query to determine which metrics are most relevant.
    
    This affects:
    - Which metrics to emphasize in display
    - How to weight the overall score
    - What explanations to show
    """
    query_lower = query.lower()
    
    # Detect query type
    query_type = "general"
    if any(kw in query_lower for kw in ["implement", "code", "algorithm", "function", "api", "library"]):
        query_type = "technical"
    elif any(kw in query_lower for kw in ["compare", "versus", "vs", "difference", "better", "pros cons"]):
        query_type = "comparison"
    elif any(kw in query_lower for kw in ["survey", "overview", "review", "state of the art", "landscape"]):
        query_type = "survey"
    elif any(kw in query_lower for kw in ["method", "approach", "technique", "how to", "process", "steps"]):
        query_type = "methodology"
    
    # Detect complexity
    word_count = len(query.split())
    has_multiple_aspects = any(kw in query_lower for kw in ["and", "also", "including", "as well as"])
    
    if word_count > 30 or has_multiple_aspects:
        complexity = "complex"
    elif word_count > 15:
        complexity = "moderate"
    else:
        complexity = "simple"
    
    # Detect requirements
    requires_citations = any(kw in query_lower for kw in [
        "cite", "reference", "paper", "study", "research", "literature", "evidence"
    ])
    requires_synthesis = any(kw in query_lower for kw in [
        "synthesize", "combine", "integrate", "across", "multiple", "compare"
    ])
    requires_code = any(kw in query_lower for kw in [
        "code", "implement", "example", "snippet", "function", "class"
    ])
    
    # Research depth
    if any(kw in query_lower for kw in ["deep", "comprehensive", "thorough", "detailed", "extensive"]):
        research_depth = "deep"
    elif any(kw in query_lower for kw in ["brief", "quick", "simple", "basic", "overview"]):
        research_depth = "shallow"
    else:
        research_depth = "moderate"
    
    # Confidence based on clarity of classification
    confidence = 0.7
    if query_type != "general":
        confidence += 0.15
    if complexity == "complex":
        confidence += 0.1
    
    return QueryClassification(
        query_type=query_type,
        complexity=complexity,
        requires_citations=requires_citations,
        requires_synthesis=requires_synthesis,
        requires_code=requires_code,
        research_depth=research_depth,
        confidence=min(1.0, confidence),
    )


# ══════════════════════════════════════════════════════════════════════════
# METRIC DESCRIPTIONS
# ══════════════════════════════════════════════════════════════════════════

METRIC_DESCRIPTIONS = {
    # Retrieval metrics
    "ndcg_at_10": {
        "name": "Ranking Quality (nDCG@10)",
        "description": "Measures how well relevant content is ranked at the top of retrieval results.",
        "good_threshold": 0.7,
        "category": "retrieval",
    },
    "precision_at_5": {
        "name": "Precision@5",
        "description": "Fraction of top-5 retrieved chunks that are actually relevant to the query.",
        "good_threshold": 0.6,
        "category": "retrieval",
    },
    "recall_at_10": {
        "name": "Recall@10",
        "description": "Fraction of all relevant content that appears in top-10 results.",
        "good_threshold": 0.5,
        "category": "retrieval",
    },
    "mrr": {
        "name": "Mean Reciprocal Rank",
        "description": "How early the first relevant result appears in the ranking.",
        "good_threshold": 0.5,
        "category": "retrieval",
    },
    "map": {
        "name": "Mean Average Precision",
        "description": "Average precision across all relevant positions in the ranking.",
        "good_threshold": 0.5,
        "category": "retrieval",
    },
    # Generation metrics
    "faithfulness": {
        "name": "Faithfulness",
        "description": "Measures whether generated claims are grounded in the source evidence.",
        "good_threshold": 0.7,
        "category": "grounding",
    },
    "citation_alignment": {
        "name": "Citation Alignment",
        "description": "Evaluates whether citations accurately reference the source material.",
        "good_threshold": 0.7,
        "category": "grounding",
    },
    "semantic_grounding": {
        "name": "Semantic Grounding",
        "description": "Token-level overlap between generated content and source evidence.",
        "good_threshold": 0.5,
        "category": "grounding",
    },
    "context_utilization": {
        "name": "Context Utilization",
        "description": "How effectively retrieved chunks are used in the response.",
        "good_threshold": 0.6,
        "category": "generation",
    },
    "redundancy_reduction": {
        "name": "Redundancy Reduction",
        "description": "Measures removal of repetitive or duplicate information.",
        "good_threshold": 0.8,
        "category": "generation",
    },
    "retrieval_diversity": {
        "name": "Source Diversity",
        "description": "Evaluates variation and breadth of information sources used.",
        "good_threshold": 0.5,
        "category": "retrieval",
    },
    "semantic_coherence": {
        "name": "Semantic Coherence",
        "description": "Measures logical flow and consistency across response sections.",
        "good_threshold": 0.6,
        "category": "generation",
    },
    "cross_document_synthesis": {
        "name": "Cross-Document Synthesis",
        "description": "How well insights from multiple papers are integrated together.",
        "good_threshold": 0.5,
        "category": "synthesis",
    },
    "hallucination_score": {
        "name": "Hallucination Risk",
        "description": "Fraction of claims that may not be supported by sources.",
        "good_threshold": 0.2,  # Inverted - lower is better
        "category": "grounding",
        "inverted": True,
    },
    "overall_quality": {
        "name": "Overall Quality",
        "description": "Weighted average of all generation quality metrics.",
        "good_threshold": 0.65,
        "category": "overall",
    },
}


# ══════════════════════════════════════════════════════════════════════════
# DYNAMIC METRIC SELECTION
# ══════════════════════════════════════════════════════════════════════════

def select_relevant_metrics(
    query_class: QueryClassification,
    has_multiple_papers: bool = True,
    has_web_search: bool = False,
) -> list[str]:
    """
    Select which metrics are most relevant for this query type.
    
    Returns ordered list of metric keys to display.
    """
    # Always include core metrics
    metrics = ["overall_quality", "faithfulness", "hallucination_score"]
    
    # Add based on query type
    if query_class.query_type == "technical":
        metrics.extend(["semantic_grounding", "context_utilization", "precision_at_5"])
    elif query_class.query_type == "comparison":
        metrics.extend(["cross_document_synthesis", "retrieval_diversity", "semantic_coherence"])
    elif query_class.query_type == "survey":
        metrics.extend(["cross_document_synthesis", "retrieval_diversity", "redundancy_reduction", "recall_at_10"])
    elif query_class.query_type == "methodology":
        metrics.extend(["semantic_grounding", "context_utilization", "semantic_coherence"])
    else:
        metrics.extend(["semantic_grounding", "semantic_coherence", "context_utilization"])
    
    # Add citation metrics if needed
    if query_class.requires_citations:
        if "citation_alignment" not in metrics:
            metrics.append("citation_alignment")
    
    # Add synthesis metrics for multi-paper queries
    if query_class.requires_synthesis or has_multiple_papers:
        if "cross_document_synthesis" not in metrics:
            metrics.append("cross_document_synthesis")
    
    # Add retrieval metrics for complex queries
    if query_class.complexity == "complex":
        for m in ["ndcg_at_10", "mrr"]:
            if m not in metrics:
                metrics.append(m)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_metrics = []
    for m in metrics:
        if m not in seen:
            seen.add(m)
            unique_metrics.append(m)
    
    return unique_metrics


# ══════════════════════════════════════════════════════════════════════════
# FORMATTED METRIC DISPLAY
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class FormattedMetric:
    """A single metric formatted for display."""
    key: str
    name: str
    score: float
    score_display: str      # e.g., "8.7/10"
    description: str
    quality_label: str      # "Excellent", "Good", "Fair", "Needs Improvement"
    category: str
    is_inverted: bool = False


def format_metric_for_display(key: str, score: float) -> FormattedMetric:
    """Format a single metric for UI display."""
    info = METRIC_DESCRIPTIONS.get(key, {
        "name": key.replace("_", " ").title(),
        "description": "Evaluation metric",
        "good_threshold": 0.5,
        "category": "other",
    })
    
    is_inverted = info.get("inverted", False)
    threshold = info["good_threshold"]
    
    # Convert to 10-point scale for display
    if is_inverted:
        # For inverted metrics (like hallucination), lower is better
        display_score = round((1.0 - score) * 10, 1)
        if score <= threshold * 0.5:
            quality = "Excellent"
        elif score <= threshold:
            quality = "Good"
        elif score <= threshold * 1.5:
            quality = "Fair"
        else:
            quality = "Needs Improvement"
    else:
        display_score = round(score * 10, 1)
        if score >= threshold * 1.2:
            quality = "Excellent"
        elif score >= threshold:
            quality = "Good"
        elif score >= threshold * 0.7:
            quality = "Fair"
        else:
            quality = "Needs Improvement"
    
    return FormattedMetric(
        key=key,
        name=info["name"],
        score=score,
        score_display=f"{display_score}/10",
        description=info["description"],
        quality_label=quality,
        category=info["category"],
        is_inverted=is_inverted,
    )


def format_metrics_for_ui(
    metrics: dict,
    query: str,
    has_multiple_papers: bool = True,
    has_web_search: bool = False,
) -> dict:
    """
    Format evaluation metrics for frontend display.
    
    Returns structured data for rendering in the UI.
    """
    # Classify query to select relevant metrics
    query_class = classify_query(query)
    relevant_keys = select_relevant_metrics(query_class, has_multiple_papers, has_web_search)
    
    # Format each metric
    formatted_metrics = []
    for key in relevant_keys:
        if key in metrics:
            formatted = format_metric_for_display(key, metrics[key])
            formatted_metrics.append({
                "key": formatted.key,
                "name": formatted.name,
                "score": formatted.score,
                "score_display": formatted.score_display,
                "description": formatted.description,
                "quality_label": formatted.quality_label,
                "category": formatted.category,
            })
    
    # Group by category
    categories = {}
    for m in formatted_metrics:
        cat = m["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(m)
    
    # Overall summary
    overall = metrics.get("overall_quality", 0)
    hallucination = metrics.get("hallucination_score", 0)
    
    if overall >= 0.75 and hallucination <= 0.1:
        verdict = "HIGH QUALITY"
        verdict_description = "Reliable research output with strong evidence grounding"
    elif overall >= 0.6 and hallucination <= 0.2:
        verdict = "GOOD"
        verdict_description = "Generally reliable with minor areas for improvement"
    elif overall >= 0.45:
        verdict = "MODERATE"
        verdict_description = "Usable but verify key claims against sources"
    else:
        verdict = "NEEDS REVIEW"
        verdict_description = "Manual review recommended for accuracy"
    
    return {
        "query_classification": {
            "type": query_class.query_type,
            "complexity": query_class.complexity,
            "research_depth": query_class.research_depth,
        },
        "metrics": formatted_metrics,
        "categories": categories,
        "summary": {
            "verdict": verdict,
            "verdict_description": verdict_description,
            "overall_score": round(overall * 10, 1),
            "hallucination_risk": round(hallucination * 100, 1),
        },
        "metric_count": len(formatted_metrics),
    }


# ══════════════════════════════════════════════════════════════════════════
# FULL DYNAMIC EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def run_dynamic_evaluation(
    query: str,
    generated_text: str,
    source_chunks: list[str],
    paper_titles: list[str],
    has_web_search: bool = False,
) -> dict:
    """
    Run full dynamic evaluation with context-aware metric selection.
    
    This is the main entry point for evaluation in deep research mode.
    
    Returns:
        Complete evaluation result with dynamically selected and formatted metrics.
    """
    logger.info("Running dynamic evaluation for query type detection...")
    
    # Run base evaluation
    evaluation = evaluate_pipeline_output(
        query=query,
        generated_text=generated_text,
        source_chunks=source_chunks,
        paper_titles=paper_titles,
    )
    
    # Format for UI with dynamic metric selection
    has_multiple_papers = len(paper_titles) > 1
    formatted = format_metrics_for_ui(
        metrics=evaluation["metrics"],
        query=query,
        has_multiple_papers=has_multiple_papers,
        has_web_search=has_web_search,
    )
    
    # Combine results
    result = {
        "raw_metrics": evaluation["metrics"],
        "hallucination_report": evaluation["hallucination_report"],
        "evaluation_summary": evaluation["summary"],
        "formatted_display": formatted,
        "query_analysis": formatted["query_classification"],
    }
    
    logger.info(
        "Dynamic evaluation complete: type=%s, verdict=%s, metrics_shown=%d",
        formatted["query_classification"]["type"],
        formatted["summary"]["verdict"],
        formatted["metric_count"],
    )
    
    return result
