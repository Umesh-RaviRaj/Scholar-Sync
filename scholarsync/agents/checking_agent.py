"""
Checking Agent (Auditor) — validates worker outputs by comparing claims
against source context. Detects hallucinations, verifies grounding,
and issues correction prompts when confidence is below threshold.

OPTIMISED: consolidated vector searches and right-sized token limits.
"""

from __future__ import annotations

import json

from scholarsync.config.settings import get_settings
from scholarsync.chat.key_manager import get_key_manager
from scholarsync.rag.vector_store import search as vector_search
from scholarsync.utils.logger import get_logger
from scholarsync.utils.schemas import (
    ExtractedKnowledge,
    ValidationResult,
    ClaimValidation,
)

logger = get_logger(__name__)


CHECKING_SYSTEM_PROMPT = """You are the Checking Agent (Auditor) of ScholarSync, a multi-agent literature review system.

Your role is to validate extracted knowledge against the original source material.

For each claim, finding, or extraction provided, you must:
1. Check if it is GROUNDED in the provided source text
2. Identify any HALLUCINATIONS (information not supported by sources)
3. Verify that citations and attributions are accurate
4. Flag any "shadow claims" (assertions without evidence)
5. Assign a confidence score (0.0 to 1.0) for each item

You MUST output valid JSON with this structure:
{
  "overall_score": 0.34,
  "is_valid": true,
  "claim_validations": [
    {
      "claim": "the specific claim text",
      "is_grounded": true,
      "confidence": 0.45,
      "supporting_evidence": ["quote from source that supports this"],
      "issues": []
    }
  ],
  "hallucination_flags": ["description of any hallucinated content"],
  "unsupported_claims": ["claims that lack source support"],
  "correction_prompts": ["specific instructions to fix issues"],
  "feedback": "Overall assessment of extraction quality"
}

Scoring Guidelines:
- 1.0: Perfectly grounded with direct textual evidence
- 0.8-0.9: Well supported with minor phrasing differences
- 0.6-0.7: Partially supported, some inference required
- 0.4-0.5: Weakly supported, significant extrapolation
- 0.0-0.3: Not supported or hallucinated

Be strict but fair. The goal is factual accuracy, not creative interpretation.
"""


def validate_extraction(
    extraction: ExtractedKnowledge,
    session_id: str | None = None,
) -> ValidationResult:
    """
    Validate a single worker's extraction against source documents.

    Uses a SINGLE consolidated vector search instead of per-claim searches.
    """
    settings = get_settings()
    km = get_key_manager()

    # ── Gather all claims to validate ───────────────────────────────
    all_claims = []
    all_claims.extend(extraction.findings)
    all_claims.extend(extraction.claims)
    all_claims.extend(extraction.methodology)
    all_claims.extend(extraction.risks)

    if not all_claims:
        return ValidationResult(
            overall_score=1.0,
            is_valid=True,
            feedback="No claims to validate — extraction was empty.",
        )

    # ── Single consolidated vector search ───────────────────────────
    # Combine top claims into one search query instead of N separate searches
    combined_query = " ".join(all_claims[:8])
    if len(combined_query) > 500:
        combined_query = combined_query[:500]

    context_chunks = vector_search(
        query=combined_query,
        n_results=8,
        paper_id=extraction.paper_id,
    )

    if not context_chunks:
        return ValidationResult(
            overall_score=0.5,
            is_valid=False,
            feedback="Could not retrieve source context for validation.",
            correction_prompts=["Re-extract with more specific queries to find source material."],
        )

    source_context = "\n\n---\n\n".join(
        f"[Source Chunk {c['id']}]:\n{c['text']}" for c in context_chunks
    )

    # ── Build validation prompt ─────────────────────────────────────
    claims_text = "\n".join(f"  {i+1}. {claim}" for i, claim in enumerate(all_claims[:15]))

    entities_text = ""
    if extraction.entities:
        entities_text = "\n\nExtracted Entities:\n" + "\n".join(
            f"  - {e.name} ({e.entity_type}): {e.description}"
            for e in extraction.entities[:10]
        )

    user_prompt = f"""Validate the following extractions from paper "{extraction.paper_title}" (ID: {extraction.paper_id}):

Extraction Type: {extraction.subtask_type.value}

Claims/Findings to Validate:
{claims_text}
{entities_text}

--- Original Source Text ---
{source_context}
--- End of Source ---

Validate each claim against the source text. Output valid JSON only."""

    # ── Call Groq LLM (via KeyManager for rotation/failover) ────────
    logger.info(
        "Checking Agent: validating %d claims from '%s'",
        len(all_claims),
        extraction.paper_title,
    )

    raw_text = km.call_llm(
        messages=[
            {"role": "system", "content": CHECKING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,  # Deterministic for validation
        max_tokens=settings.checker_max_tokens,
        response_format={"type": "json_object"},
        session_id=session_id,
    )

    # ── Parse response ──────────────────────────────────────────────
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error("Checking Agent: failed to parse validation response")
        return ValidationResult(
            overall_score=0.5,
            is_valid=False,
            feedback="Validation response could not be parsed.",
        )

    claim_validations = [
        ClaimValidation(
            claim=cv.get("claim", ""),
            is_grounded=cv.get("is_grounded", False),
            confidence=min(max(cv.get("confidence", 0.0), 0.0), 1.0),
            supporting_evidence=cv.get("supporting_evidence", []),
            issues=cv.get("issues", []),
        )
        for cv in data.get("claim_validations", [])
    ]

    overall_score = data.get("overall_score", 0.0)
    overall_score = min(max(overall_score, 0.0), 1.0)

    result = ValidationResult(
        overall_score=overall_score,
        is_valid=overall_score >= settings.validation_threshold,
        claim_validations=claim_validations,
        hallucination_flags=data.get("hallucination_flags", []),
        unsupported_claims=data.get("unsupported_claims", []),
        correction_prompts=data.get("correction_prompts", []),
        feedback=data.get("feedback", ""),
    )

    logger.info(
        "Checking Agent: score=%.2f valid=%s for '%s' (%s)",
        result.overall_score,
        result.is_valid,
        extraction.paper_title,
        extraction.subtask_type.value,
    )
    return result


def validate_all_extractions(
    extractions: list[ExtractedKnowledge],
    session_id: str | None = None,
) -> list[ValidationResult]:
    """
    Validate all worker extractions (sequentially to be kind to rate limits).
    """
    import concurrent.futures

    logger.info("Checking Agent: validating %d extractions", len(extractions))
    results = []

    def _validate(extraction):
        try:
            result = validate_extraction(extraction, session_id=session_id)
            msg = f"  \u21b3 Checked '{extraction.paper_title[:25]}...': Score {result.overall_score:.2f} ({'Valid' if result.is_valid else 'Needs Revision'})"
            if session_id:
                try:
                    from scholarsync.chat.mode_router import enqueue_thought
                    enqueue_thought(session_id, msg)
                except Exception:
                    pass
            return result
        except Exception as e:
            logger.error(
                "Checking Agent: validation failed for %s/%s: %s",
                extraction.paper_id,
                extraction.subtask_type.value,
                e,
            )
            return ValidationResult(
                overall_score=0.0,
                is_valid=False,
                feedback=f"Validation error: {str(e)}",
            )

    # Sequential execution — avoids parallel rate-limit contention
    # with only 2 API keys. Each call is already fast with reduced tokens.
    for ext in extractions:
        res = _validate(ext)
        results.append(res)

    avg_score = sum(r.overall_score for r in results) / len(results) if results else 0.0
    valid_count = sum(1 for r in results if r.is_valid)

    logger.info(
        "Checking Agent: avg_score=%.2f, %d/%d passed validation",
        avg_score,
        valid_count,
        len(results),
    )
    return results
