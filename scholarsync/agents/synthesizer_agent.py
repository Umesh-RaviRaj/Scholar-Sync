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
    StructuredPaperProfile,
)
from scholarsync.agents.dedup import deduplicate_insights
from scholarsync.agents.profile_builder import build_paper_profiles

logger = get_logger(__name__)


SYNTHESIS_SYSTEM_PROMPT = """You are the Final Synthesizer Agent of ScholarSync — a senior AI research analyst, NOT a summarization bot.

You produce ANALYTICAL, COMPARATIVE, DECISIVE literature reviews. You behave like a domain expert who ranks approaches, explains tradeoffs, and delivers technically justified verdicts.

You MUST produce valid JSON with this structure:
{
  "title": "Literature Review: [Specific Topic]",

  "summary": "Executive summary (300+ words). State the research landscape, the BEST approach identified, and WHY it dominates [cite paper_number]. Be decisive. EVERY factual claim MUST have [paper_X] citation.",

  "methodology_comparison": "CRITICAL SECTION (500+ words). Compare ALL methodologies DIMENSION BY DIMENSION:\n- Retrieval quality [cite which paper reports what]\n- Semantic coherence [cite performance data]\n- Scalability [cite dataset sizes, throughput numbers]\n- Computational efficiency [cite latency, cost metrics]\n- Context preservation [cite evaluation results]\n- Long-document handling [cite capabilities]\n- Production suitability [cite deployment data]\n- Retrieval latency [cite specific ms numbers]\n- Embedding efficiency [cite vector dimensions, memory]\n- Chunking intelligence [cite chunk strategies]\n\nFor each dimension: state which paper/method wins [cite], by how much [numbers], and why [evidence]. Build a clear ranking with citations. Do NOT repeat summaries — COMPARE and CONTRAST with evidence.",

  "key_findings": "Thematic synthesis (400+ words). Organize by theme, not by paper. Include metrics [cite], benchmarks [cite], specific numbers [cite]. State what the collective evidence proves [cite sources]. MANDATORY: Every sentence with a fact must include [paper_X] citation. Never repeat the same insight twice.",

  "cross_paper_insights": "Non-obvious connections (300+ words). Contradictions, complementary techniques, convergent conclusions, emergent patterns. What story do these papers tell TOGETHER that no single paper reveals alone?",

  "identified_risks": "Limitations and risks (250+ words). Cite which papers report which problems. Group by: scalability risks, data risks, reproducibility risks, production risks.",

  "research_gaps": "Gaps and future work (250+ words). Specific actionable gaps. What datasets are missing? What hybrid approaches remain untested? What production scenarios are unaddressed?",

  "safety_scorecard": {
    "grounding_score": 0.92,
    "citation_coverage": 0.95,
    "cross_reference_score": 0.88,
    "hallucination_risk": 0.05,
    "overall_quality": 0.90
  }
  
NOTE: These scorecard values are TARGETS. Your actual report must aim for:
- 95%+ citation coverage (nearly every claim cited)
- <5% hallucination risk (only state what papers explicitly say)
- 90%+ overall quality (detailed, analytical, evidence-based)
}

CRITICAL RULES (STRICT - NO EXCEPTIONS):
1. ALWAYS cite with [paper_number]. EVERY SINGLE CLAIM must trace to a source. No exceptions.
2. NEVER make assertions without citing supporting evidence from the papers.
3. NEVER repeat the same insight in multiple sections. Each sentence must add new information.
4. NEVER use weak conclusions like "it depends on the use case" or "all methods have tradeoffs". Be DECISIVE — pick winners, explain WHY, acknowledge limitations.
5. Compare dimension-by-dimension, not paper-by-paper.
6. Use SPECIFIC data: exact numbers, percentages, benchmark names, dataset sizes, performance metrics.
7. Rank approaches explicitly: "Method X outperforms Y by Z% because..."
8. Discuss production suitability: latency (ms), cost ($), scalability (users/req), deployment complexity.
9. When stating findings, use phrases like "According to [paper_X]...", "Paper [Y] reports...", "[Z] demonstrates..."
10. If information is not explicitly in the papers, DO NOT INFER OR GUESS. State "Not reported in reviewed papers."
11. Output valid JSON only. No markdown, no preamble.
"""


def synthesize_review(
    query: str,
    extractions: list[ExtractedKnowledge],
    validation_results: list[ValidationResult],
    paper_metadata: list[PaperMetadata],
    graph_insights: dict | None = None,
    session_id: str | None = None,
    structured_profiles: list[StructuredPaperProfile] | None = None,
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
    structured_profiles : list[StructuredPaperProfile], optional
        Pre-built structured profiles (zero-cost deterministic extraction).

    Returns
    -------
    LiteratureReview
    """
    settings = get_settings()
    km = get_key_manager()

    if session_id:
        try:
            from scholarsync.chat.mode_router import enqueue_thought
            enqueue_thought(session_id, f"  ↳ Building structured profiles and analytical synthesis...")
        except Exception:
            pass

    # ── Build structured profiles if not provided ───────────────────
    if not structured_profiles:
        structured_profiles = build_paper_profiles(extractions, paper_metadata)

    # ── Build paper reference table ─────────────────────────────────
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

    # ── Build structured profile comparison table ───────────────────
    profile_sections = []
    for i, prof in enumerate(structured_profiles, 1):
        idx = next(
            (j for j, m in enumerate(paper_metadata, 1) if m.paper_id == prof.paper_id),
            i,
        )
        section = f"""[{idx}] "{prof.paper_title}"
  Problem: {prof.research_problem or 'Not specified'}
  Methodology: {prof.methodology or 'Not specified'}
  Chunking: {prof.chunking_strategy or 'Not specified'}
  Retrieval: {prof.retrieval_strategy or 'Not specified'}
  Embeddings: {', '.join(prof.embedding_models) or 'Not specified'}
  Datasets: {', '.join(prof.datasets[:4]) or 'Not specified'}
  Metrics: {', '.join(prof.evaluation_metrics[:4]) or 'Not specified'}
  Advantages: {'; '.join(prof.advantages[:3]) or 'Not specified'}
  Limitations: {'; '.join(prof.limitations[:3]) or 'Not specified'}
  Scalability: {prof.scalability or 'Not specified'}
  Cost: {prof.computational_cost or 'Not specified'}
  Best use: {prof.best_use_case or 'Not specified'}
  Contributions: {'; '.join(prof.key_contributions[:3]) or 'Not specified'}"""
        profile_sections.append(section)

    profiles_text = "\n\n".join(profile_sections)

    # ── Aggregate and deduplicate supporting evidence ────────────────
    all_findings_raw = []
    all_claims_raw = []

    for ext in extractions:
        paper_idx = next(
            (j for j, m in enumerate(paper_metadata, 1) if m.paper_id == ext.paper_id),
            0,
        )
        ref = f" [{paper_idx}]" if paper_idx else ""
        for f in ext.findings:
            all_findings_raw.append(f"{f}{ref}")
        for c in ext.claims:
            all_claims_raw.append(f"{c}{ref}")

    # Deduplicate
    deduped_findings = deduplicate_insights(all_findings_raw)
    deduped_claims = deduplicate_insights(all_claims_raw)

    # ── Build graph insights section ────────────────────────────────
    graph_section = ""
    if graph_insights:
        cross_paper = graph_insights.get("cross_paper_connections", [])
        if cross_paper:
            graph_section = "\nCross-Paper Connections:\n" + "\n".join(
                f"- '{c.get('entity', '')}' ({c.get('entity_type', '')}) shared by: {', '.join(c.get('papers', []))}"
                for c in cross_paper[:10]
            )

    # ── Validation summary ──────────────────────────────────────────
    avg_score = (
        sum(v.overall_score for v in validation_results) / len(validation_results)
        if validation_results
        else 0.0
    )

    # ── Construct the analytical synthesis prompt ───────────────────
    user_prompt = f"""Research Query: {query}

Paper References:
{paper_references}

=== STRUCTURED PAPER PROFILES (compare these dimension-by-dimension) ===
{profiles_text}

=== KEY FINDINGS (deduplicated) ===
{chr(10).join('- ' + f for f in deduped_findings[:40]) or 'None'}

=== KEY CLAIMS (deduplicated) ===
{chr(10).join('- ' + c for c in deduped_claims[:30]) or 'None'}
{graph_section}

Validation Score: {avg_score:.2f}

INSTRUCTIONS:
1. Compare papers dimension-by-dimension using the structured profiles above.
2. Rank approaches — declare which methodology is STRONGEST and WHY.
3. Never repeat the same insight. Each sentence must add new information.
4. Be decisive in conclusions — avoid hedging.
Output valid JSON only."""

    # ── Call Groq LLM ───────────────────────────────────────────────
    logger.info("Synthesizer: generating analytical literature review")

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
