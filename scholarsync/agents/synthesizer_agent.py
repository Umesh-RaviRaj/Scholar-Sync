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


SYNTHESIS_SYSTEM_PROMPT = """You are the Final Synthesizer Agent of ScholarSync — a senior AI research analyst producing PUBLICATION-QUALITY literature reviews.

You generate COMPREHENSIVE, ANALYTICAL, CITATION-RICH research reports. You behave like a domain expert who synthesizes evidence, ranks approaches, explains tradeoffs, identifies patterns, and delivers technically justified conclusions.

CRITICAL OUTPUT REQUIREMENTS:
- Generate LONG-FORM, DETAILED responses (aim for 2500-4000 words total)
- Each section must be SUBSTANTIVE and INFORMATION-DENSE
- Quality over brevity — exhaustively cover the research landscape
- Every claim MUST have a [paper_X] citation

You MUST produce valid JSON with this structure:
{
  "title": "Literature Review: [Specific Topic]",

  "summary": "COMPREHENSIVE EXECUTIVE SUMMARY (500-700 words). 
  
  Structure your summary as follows:
  1. RESEARCH LANDSCAPE: What is the current state of this field? What problems are being addressed? [cite sources]
  2. DOMINANT APPROACHES: What methodologies emerge as most effective and why? [cite performance data]
  3. KEY INNOVATIONS: What novel contributions do these papers make? [cite specific innovations]
  4. PRACTICAL IMPLICATIONS: What does this mean for practitioners? [cite deployment data]
  5. VERDICT: Which approach is BEST for different scenarios and why? Be DECISIVE with evidence.
  
  EVERY factual claim MUST have [paper_X] citation. Include specific metrics, percentages, and benchmark results.",

  "methodology_comparison": "CRITICAL ANALYSIS SECTION (800-1200 words).
  
  Compare ALL methodologies across MULTIPLE DIMENSIONS with evidence:
  
  DIMENSION 1 — RETRIEVAL QUALITY:
  - Which methods achieve highest precision/recall? [cite specific numbers]
  - How do dense vs sparse vs hybrid approaches compare? [cite benchmarks]
  - What retrieval strategies are most effective? [cite evidence]
  
  DIMENSION 2 — SCALABILITY & PERFORMANCE:
  - Dataset sizes tested [cite specific numbers]
  - Throughput and latency measurements [cite ms/query, QPS]
  - Memory and computational requirements [cite GB, GPU hours]
  - How do methods scale with document length? [cite evidence]
  
  DIMENSION 3 — SEMANTIC UNDERSTANDING:
  - Context preservation quality [cite evaluation scores]
  - Handling of complex queries [cite examples]
  - Cross-document reasoning capabilities [cite evidence]
  
  DIMENSION 4 — PRODUCTION READINESS:
  - Deployment complexity [cite infrastructure requirements]
  - Cost efficiency [cite compute costs if available]
  - Reliability and fault tolerance [cite evidence]
  
  DIMENSION 5 — CHUNKING & EMBEDDING:
  - Chunking strategies compared [cite which paper uses what]
  - Embedding model performance [cite dimension, quality metrics]
  - Optimal chunk sizes [cite experimental evidence]
  
  For EACH dimension: declare a WINNER, state by HOW MUCH (with numbers), and explain WHY with evidence.
  Build a clear RANKING with justification. Avoid hedging — be analytical and decisive.",

  "key_findings": "THEMATIC SYNTHESIS (600-900 words).
  
  Organize by THEME, not by paper. Each theme should synthesize across multiple sources:
  
  THEME 1 — [Major Finding Category]:
  - What do the papers collectively demonstrate? [cite all relevant sources]
  - What specific metrics support this? [cite numbers]
  - How consistent is the evidence? Note any contradictions [cite]
  
  THEME 2 — [Second Finding Category]:
  - Synthesize evidence from multiple papers [cite]
  - Include specific experimental results [cite metrics]
  - Explain practical significance [cite applications]
  
  THEME 3 — [Third Finding Category]:
  - Cross-reference findings [cite multiple papers]
  - Highlight patterns and trends [cite evidence]
  - Discuss implications [cite]
  
  [Continue for all major themes...]
  
  EVERY sentence with a fact MUST include [paper_X] citation. Never repeat insights. Each sentence adds new information.",

  "cross_paper_insights": "EMERGENT PATTERNS & SYNTHESIS (400-600 words).
  
  Identify NON-OBVIOUS connections that emerge only from reading multiple papers together:
  
  1. CONVERGENT CONCLUSIONS: What findings appear across multiple papers? [cite all]
  2. CONTRADICTIONS: Where do papers disagree? What explains the differences? [cite both sides]
  3. COMPLEMENTARY TECHNIQUES: Which methods could be combined for better results? [cite evidence]
  4. EVOLUTION OF APPROACHES: How has the field progressed? [cite timeline]
  5. METHODOLOGICAL PATTERNS: What experimental setups are most common/reliable? [cite]
  6. HIDDEN DEPENDENCIES: What assumptions do multiple papers share? [cite]
  
  What story do these papers tell TOGETHER that no single paper reveals alone?",

  "identified_risks": "LIMITATIONS & RISK ANALYSIS (350-500 words).
  
  Organize by risk category:
  
  SCALABILITY RISKS:
  - Memory constraints [cite which papers report]
  - Computational bottlenecks [cite evidence]
  - Performance degradation patterns [cite]
  
  DATA & GENERALIZATION RISKS:
  - Dataset biases [cite which papers address]
  - Domain transfer limitations [cite evidence]
  - Evaluation metric limitations [cite]
  
  REPRODUCIBILITY RISKS:
  - Missing implementation details [cite]
  - Hyperparameter sensitivity [cite evidence]
  - Hardware dependencies [cite]
  
  PRODUCTION RISKS:
  - Deployment challenges [cite]
  - Maintenance complexity [cite evidence]
  - Edge case handling [cite]
  
  For each risk: cite which papers report it and what mitigations (if any) are suggested.",

  "research_gaps": "FUTURE DIRECTIONS & RESEARCH GAPS (350-500 words).
  
  Identify SPECIFIC, ACTIONABLE research opportunities:
  
  1. UNTESTED COMBINATIONS: Which promising method combinations remain unexplored? [cite basis]
  2. MISSING DATASETS: What benchmarks or evaluation scenarios are needed? [cite current gaps]
  3. SCALE CHALLENGES: What happens at 10x or 100x current scale? [cite limitations]
  4. DOMAIN GAPS: Which application domains need more research? [cite coverage]
  5. THEORETICAL GAPS: What fundamental questions remain unanswered? [cite evidence]
  6. PRACTICAL GAPS: What production scenarios are unaddressed? [cite]
  7. HYBRID OPPORTUNITIES: What cross-pollination between approaches could yield improvements? [cite basis]
  
  Prioritize gaps by potential impact and feasibility."
}

CRITICAL RULES (STRICT - NO EXCEPTIONS):
1. ALWAYS cite with [paper_number]. EVERY SINGLE CLAIM must trace to a source. No exceptions.
2. NEVER make assertions without citing supporting evidence from the papers.
3. NEVER repeat the same insight in multiple sections. Each sentence must add NEW information.
4. NEVER use weak conclusions like "it depends" or "all methods have tradeoffs". Be DECISIVE — pick winners, explain WHY, acknowledge limitations.
5. Compare DIMENSION-BY-DIMENSION, not paper-by-paper.
6. Use SPECIFIC data: exact numbers, percentages, benchmark names, dataset sizes, performance metrics.
7. Rank approaches EXPLICITLY: "Method X outperforms Y by Z% because..."
8. Discuss PRODUCTION SUITABILITY: latency (ms), cost ($), scalability, deployment complexity.
9. Use evidence phrases: "According to [paper_X]...", "Paper [Y] demonstrates...", "[Z] reports..."
10. If information is not in the papers, state "Not reported in reviewed papers." — NEVER guess.
11. AIM FOR DEPTH: Each section should be SUBSTANTIVE. Avoid shallow summaries.
12. SYNTHESIZE across sources: Connect findings, identify patterns, build a coherent narrative.
13. Output VALID JSON only. No markdown, no preamble, no explanation outside the JSON.
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

    review = LiteratureReview(
        title=data.get("title", f"Literature Review: {query}"),
        summary=data.get("summary", ""),
        methodology_comparison=data.get("methodology_comparison", ""),
        key_findings=data.get("key_findings", ""),
        cross_paper_insights=data.get("cross_paper_insights", ""),
        identified_risks=data.get("identified_risks", ""),
        research_gaps=data.get("research_gaps", ""),
        citations=citations,
        safety_scorecard={},  # Deprecated - kept for schema compatibility
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
