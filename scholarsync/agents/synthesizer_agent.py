"""
Synthesizer Agent — merges all validated extractions into a structured,
citation-aware literature review with cross-paper insights.
"""

from __future__ import annotations

import json
from datetime import datetime

from scholarsync.config.settings import get_settings
from scholarsync.chat.key_manager import get_key_manager
from scholarsync.utils.logger import get_logger
from scholarsync.utils.schemas import (
    ExtractedKnowledge,
    ValidationResult,
    LiteratureReview,
    CitationEntry,
    PaperMetadata,
)

logger = get_logger(__name__)


SYNTHESIS_SYSTEM_PROMPT = """You are the Final Synthesizer Agent of ScholarSync, a deep-research multi-agent literature review system.

Your role is to produce a COMPREHENSIVE, PUBLICATION-QUALITY literature review from validated extractions across multiple research papers.
This is NOT a brief summary — this is deep academic research synthesis. Each section must be thorough and exhaustive.

You MUST produce valid JSON with this structure:
{
  "title": "Literature Review: [Specific Topic]",

  "summary": "REQUIRED: A deep executive summary of AT LEAST 400 words. Cover: the research landscape, why this topic matters, what the key papers contribute collectively, major trends observed, and the overall state of the field. Be specific about methods, results, and significance.",

  "methodology_comparison": "REQUIRED: AT LEAST 600 words. Compare and contrast ALL methodologies found across papers in depth. Discuss: experimental setups, datasets used, evaluation metrics, model architectures, training procedures, and algorithmic approaches. Use a structured comparison — highlight where papers agree, differ, improve upon each other, or use complementary approaches. Always cite with [1], [2], etc.",

  "key_findings": "REQUIRED: AT LEAST 600 words. Synthesize the most important findings from ALL papers. Do NOT list them paper by paper — organize thematically. Include specific numbers, metrics, benchmark scores, and direct quotes where available. Discuss what these findings mean collectively and how they advance the field.",

  "cross_paper_insights": "REQUIRED: AT LEAST 400 words. Identify non-obvious connections, patterns, and insights that ONLY emerge when reading all papers together. What do they collectively suggest? Are there contradictions? Convergent conclusions? Complementary techniques that could be combined? What story do these papers tell as a whole?",

  "identified_risks": "REQUIRED: AT LEAST 300 words. Comprehensively document risks, limitations, threats to validity, ethical concerns, reproducibility issues, data biases, computational constraints, and generalization problems found across the literature. Be specific — cite which papers report which limitations.",

  "research_gaps": "REQUIRED: AT LEAST 300 words. Identify specific, actionable gaps in the current literature. What questions remain unanswered? What datasets are missing? What methods have not been tried? What populations or domains are understudied? Propose concrete future research directions.",

  "safety_scorecard": {
    "grounding_score": 0.85,
    "citation_coverage": 0.90,
    "cross_reference_score": 0.80,
    "hallucination_risk": 0.10,
    "overall_quality": 0.85
  }
}

CRITICAL RULES:
1. ALWAYS cite with [paper_number] — every claim must be grounded in the source papers
2. Never just list paper summaries — SYNTHESIZE and COMPARE across papers thematically
3. Use specific data: numbers, percentages, benchmark names, dataset names, accuracy scores
4. Include direct quotes from papers where they add value
5. Write at a PhD / academic journal level — detailed, precise, and analytical
6. Each section must meet its minimum word count — do NOT truncate
7. Identify agreements AND contradictions across papers
8. Output valid JSON only — do not add any text outside the JSON object
"""


def synthesize_review(
    query: str,
    extractions: list[ExtractedKnowledge],
    validation_results: list[ValidationResult],
    paper_metadata: list[PaperMetadata],
    graph_insights: dict | None = None,
    session_id: str | None = None,
) -> LiteratureReview:
    """
    Synthesize a complete literature review from validated extractions.

    Parameters
    ----------
    query : str
        The original research query.
    extractions : list[ExtractedKnowledge]
        All validated worker extractions.
    validation_results : list[ValidationResult]
        Validation scores for each extraction.
    paper_metadata : list[PaperMetadata]
        Metadata for all papers.
    graph_insights : dict, optional
        Cross-paper insights from the knowledge graph.

    Returns
    -------
    LiteratureReview
    """
    settings = get_settings()
    km = get_key_manager()

    if session_id:
        try:
            from scholarsync.chat.mode_router import enqueue_thought
            enqueue_thought(session_id, f"  ↳ Processing {len(extractions)} extracted knowledge chunks...")
        except Exception:
            pass

    # ── Build paper reference table ─────────────────────────────────
    paper_map = {p.paper_id: p for p in paper_metadata}
    citations: list[CitationEntry] = []
    paper_ref_lines = []

    for i, meta in enumerate(paper_metadata, 1):
        citation_id = f"[{i}]"
        citations.append(
            CitationEntry(
                citation_id=citation_id,
                paper_title=meta.title,
                authors=meta.authors,
                year=meta.year,
            )
        )
        authors_str = ", ".join(meta.authors) if meta.authors else "Unknown"
        year_str = f" ({meta.year})" if meta.year else ""
        paper_ref_lines.append(f"{citation_id} {meta.title} — {authors_str}{year_str}")

    paper_references = "\n".join(paper_ref_lines)

    # ── Aggregate extractions by type ───────────────────────────────
    all_entities = []
    all_methodology = []
    all_findings = []
    all_risks = []
    all_claims = []

    for ext in extractions:
        pid = ext.paper_id
        # Find the citation number for this paper
        paper_idx = next(
            (i for i, m in enumerate(paper_metadata, 1) if m.paper_id == pid),
            0,
        )
        ref = f"[{paper_idx}]" if paper_idx else ""

        for e in ext.entities:
            all_entities.append(f"- {e.name} ({e.entity_type}): {e.description} {ref}")
        for m in ext.methodology:
            all_methodology.append(f"- {m} {ref}")
        for f in ext.findings:
            all_findings.append(f"- {f} {ref}")
        for r in ext.risks:
            all_risks.append(f"- {r} {ref}")
        for c in ext.claims:
            all_claims.append(f"- {c} {ref}")

    # ── Build graph insights section ────────────────────────────────
    graph_section = ""
    if graph_insights:
        cross_paper = graph_insights.get("cross_paper_connections", [])
        if cross_paper:
            graph_section = "\n\nCross-Paper Graph Connections:\n" + "\n".join(
                f"- Entity '{c.get('entity', '')}' ({c.get('entity_type', '')}) "
                f"appears in: {', '.join(c.get('papers', []))}"
                for c in cross_paper[:15]
            )

    # ── Build validation summary ────────────────────────────────────
    avg_score = (
        sum(v.overall_score for v in validation_results) / len(validation_results)
        if validation_results
        else 0.0
    )

    # ── Construct the synthesis prompt ──────────────────────────────
    user_prompt = f"""Research Query: {query}

Paper References:
{paper_references}

=== EXTRACTED ENTITIES & CONCEPTS ===
{chr(10).join(all_entities[:200]) or "None extracted"}

=== EXTRACTED METHODOLOGY ===
{chr(10).join(all_methodology[:150]) or "None extracted"}

=== EXTRACTED FINDINGS & RESULTS ===
{chr(10).join(all_findings[:200]) or "None extracted"}

=== EXTRACTED RISKS & LIMITATIONS ===
{chr(10).join(all_risks[:150]) or "None extracted"}

=== EXTRACTED CLAIMS & CONTRIBUTIONS ===
{chr(10).join(all_claims[:150]) or "None extracted"}
{graph_section}

Validation Average Score: {avg_score:.2f}

Synthesize a comprehensive literature review from the above extractions.
Use the citation numbers {', '.join(c.citation_id for c in citations)} to reference papers.
Focus on cross-paper comparisons and thematic organization.
Output valid JSON only."""

    # ── Call Groq LLM (via KeyManager for rotation/failover) ────────
    logger.info("Synthesizer: generating literature review")

    raw_text = km.call_llm(
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=settings.synthesizer_max_tokens,
        response_format={"type": "json_object"},
        session_id=session_id,
    )

    # ── Parse response ──────────────────────────────────────────────
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error("Synthesizer: failed to parse JSON response")
        data = {}

    safety_scorecard = data.get("safety_scorecard", {})
    if not safety_scorecard:
        safety_scorecard = {
            "grounding_score": avg_score,
            "citation_coverage": 0.0,
            "cross_reference_score": 0.0,
            "hallucination_risk": 1.0 - avg_score,
            "overall_quality": avg_score,
        }

    review = LiteratureReview(
        title=data.get("title", f"Literature Review: {query}"),
        summary=data.get("summary", ""),
        methodology_comparison=data.get("methodology_comparison", ""),
        key_findings=data.get("key_findings", ""),
        cross_paper_insights=data.get("cross_paper_insights", ""),
        identified_risks=data.get("identified_risks", ""),
        research_gaps=data.get("research_gaps", ""),
        citations=citations,
        safety_scorecard=safety_scorecard,
        generated_at=datetime.utcnow(),
    )

    logger.info("Synthesizer: review generated — '%s'", review.title)
    return review


def format_review_as_markdown(review: LiteratureReview) -> str:
    """
    Format a LiteratureReview object as a Markdown document.
    """
    lines: list[str] = []

    lines.append(f"# {review.title}")
    lines.append(f"\n*Generated by ScholarSync on {review.generated_at.strftime('%Y-%m-%d %H:%M UTC')}*\n")

    lines.append("---\n")

    # Summary
    if review.summary:
        lines.append("## Executive Summary\n")
        lines.append(review.summary)
        lines.append("")

    # Methodology Comparison
    if review.methodology_comparison:
        lines.append("## Methodology Comparison\n")
        lines.append(review.methodology_comparison)
        lines.append("")

    # Key Findings
    if review.key_findings:
        lines.append("## Key Findings\n")
        lines.append(review.key_findings)
        lines.append("")

    # Cross-Paper Insights
    if review.cross_paper_insights:
        lines.append("## Cross-Paper Insights\n")
        lines.append(review.cross_paper_insights)
        lines.append("")

    # Risks & Limitations
    if review.identified_risks:
        lines.append("## Identified Risks & Limitations\n")
        lines.append(review.identified_risks)
        lines.append("")

    # Research Gaps
    if review.research_gaps:
        lines.append("## Research Gaps & Future Directions\n")
        lines.append(review.research_gaps)
        lines.append("")

    # Safety Scorecard
    if review.safety_scorecard:
        lines.append("## Safety & Quality Scorecard\n")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        for metric, score in review.safety_scorecard.items():
            display_name = metric.replace("_", " ").title()
            if isinstance(score, float):
                lines.append(f"| {display_name} | {score:.2f} |")
            else:
                lines.append(f"| {display_name} | {score} |")
        lines.append("")

    # Citations
    if review.citations:
        lines.append("## References\n")
        for cit in review.citations:
            authors_str = ", ".join(cit.authors) if cit.authors else "Unknown"
            year_str = f" ({cit.year})" if cit.year else ""
            lines.append(f"{cit.citation_id} {cit.paper_title} — {authors_str}{year_str}")
        lines.append("")

    lines.append("---")
    lines.append("*Report generated by ScholarSync — Agentic AI Literature Review System*")

    return "\n".join(lines)
