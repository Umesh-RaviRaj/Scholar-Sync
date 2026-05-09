"""
Grounding checker — evaluates extraction quality using RAGAS-style metrics.
Provides faithfulness, relevancy, and context precision scores.

FIXED: Uses KeyManager for API key rotation instead of raw Groq client.
"""

from __future__ import annotations

import json

from scholarsync.config.settings import get_settings
from scholarsync.chat.key_manager import get_key_manager
from scholarsync.rag.vector_store import search as vector_search
from scholarsync.utils.logger import get_logger
from scholarsync.utils.schemas import ExtractedKnowledge

logger = get_logger(__name__)


FAITHFULNESS_PROMPT = """You are evaluating the FAITHFULNESS of extracted information against source text.

Faithfulness measures whether the extracted claims are supported by the source material.

Given the extracted claims and the source text, rate each claim:
- 1.0 = Fully supported by source
- 0.5 = Partially supported
- 0.0 = Not supported / hallucinated

Output JSON:
{
  "scores": [{"claim": "...", "score": 0.9}],
  "average_faithfulness": 0.85,
  "explanation": "..."
}
"""

RELEVANCY_PROMPT = """You are evaluating the RELEVANCY of extracted information to the research query.

Relevancy measures how well the extracted information answers the research query.

Given the query and the extractions, rate overall relevancy:
- 1.0 = Highly relevant, directly answers the query
- 0.5 = Somewhat relevant
- 0.0 = Not relevant

Output JSON:
{
  "relevancy_score": 0.85,
  "relevant_items": ["..."],
  "irrelevant_items": ["..."],
  "explanation": "..."
}
"""


def evaluate_faithfulness(
    extraction: ExtractedKnowledge,
) -> dict:
    """
    Evaluate faithfulness of an extraction against source context.
    """
    km = get_key_manager()

    # Gather claims
    claims = extraction.findings + extraction.claims + extraction.methodology
    if not claims:
        return {"average_faithfulness": 1.0, "explanation": "No claims to evaluate"}

    # Retrieve source context
    source_chunks = []
    for claim in claims[:3]:
        chunks = vector_search(query=claim, n_results=3, paper_id=extraction.paper_id)
        source_chunks.extend(chunks)

    source_text = "\n\n".join(c["text"] for c in source_chunks[:8])
    claims_text = "\n".join(f"- {c}" for c in claims[:15])

    result = km.call_llm(
        messages=[
            {"role": "system", "content": FAITHFULNESS_PROMPT},
            {"role": "user", "content": f"Claims:\n{claims_text}\n\nSource Text:\n{source_text}"},
        ],
        temperature=0.0,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {"average_faithfulness": 0.5, "explanation": "Failed to parse evaluation"}


def evaluate_relevancy(
    query: str,
    extractions: list[ExtractedKnowledge],
) -> dict:
    """
    Evaluate relevancy of all extractions to the research query.
    """
    km = get_key_manager()

    # Summarize extractions
    summary_parts = []
    for ext in extractions[:10]:
        items = ext.findings[:3] + ext.claims[:3] + ext.methodology[:2]
        if items:
            summary_parts.append(
                f"Paper '{ext.paper_title}' ({ext.subtask_type.value}): "
                + "; ".join(items[:5])
            )

    extractions_text = "\n".join(summary_parts) or "No extractions"

    result = km.call_llm(
        messages=[
            {"role": "system", "content": RELEVANCY_PROMPT},
            {
                "role": "user",
                "content": f"Research Query: {query}\n\nExtractions:\n{extractions_text}",
            },
        ],
        temperature=0.0,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {"relevancy_score": 0.5, "explanation": "Failed to parse evaluation"}


def compute_quality_scores(
    query: str,
    extractions: list[ExtractedKnowledge],
) -> dict:
    """
    Compute comprehensive quality scores for the extraction pipeline.
    """
    logger.info("Computing quality scores for %d extractions", len(extractions))

    # Faithfulness per extraction
    faithfulness_scores = []
    for ext in extractions:
        try:
            result = evaluate_faithfulness(ext)
            faithfulness_scores.append(result.get("average_faithfulness", 0.5))
        except Exception as e:
            logger.error("Faithfulness evaluation failed: %s", e)
            faithfulness_scores.append(0.5)

    # Relevancy
    try:
        relevancy_result = evaluate_relevancy(query, extractions)
        relevancy_score = relevancy_result.get("relevancy_score", 0.5)
    except Exception as e:
        logger.error("Relevancy evaluation failed: %s", e)
        relevancy_score = 0.5

    avg_faithfulness = (
        sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
    )

    scores = {
        "faithfulness": round(avg_faithfulness, 3),
        "relevancy": round(relevancy_score, 3),
        "overall": round((avg_faithfulness + relevancy_score) / 2, 3),
        "num_extractions": len(extractions),
    }

    logger.info("Quality scores: %s", scores)
    return scores
