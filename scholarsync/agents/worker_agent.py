"""
Worker Agent — reads assigned document chunks via RAG retrieval and
extracts structured knowledge (entities, methodology, findings, risks,
claims) using Groq LLM with Pydantic-structured output.

OPTIMISED: batches ALL subtasks for a single paper into ONE LLM call
instead of one call per (subtask × paper), reducing API calls by ~80%.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from scholarsync.config.settings import get_settings
from scholarsync.chat.key_manager import get_key_manager
from scholarsync.rag.vector_store import search as vector_search
from scholarsync.utils.logger import get_logger
from scholarsync.utils.schemas import (
    ExtractedKnowledge,
    Entity,
    Relationship,
    SubTask,
    SubTaskType,
    PaperMetadata,
)

logger = get_logger(__name__)


EXTRACTION_SYSTEM_PROMPT = """You are a Worker Agent in ScholarSync, a multi-agent literature review system.

Your job is to extract structured knowledge from research paper text based on MULTIPLE subtasks simultaneously.

You MUST output valid JSON matching this schema:
{
  "entities": [
    {"name": "...", "entity_type": "method|dataset|metric|concept|tool|framework|embedding|chunking_method|retrieval_method|architecture|benchmark", "description": "..."}
  ],
  "methodology": ["description of method 1", "..."],
  "findings": ["finding 1 with specific numbers/metrics", "..."],
  "risks": ["risk/limitation 1", "..."],
  "claims": ["claim 1", "..."],
  "supporting_quotes": ["direct quote from text 1", "..."],
  "relationships": [
    {"source_entity": "...", "target_entity": "...", "relationship_type": "USES|IMPROVES|OUTPERFORMS|EVALUATED_ON|CHUNKS_WITH|RETRIEVES_WITH|EMBEDS_WITH|COMPARES_WITH|OPTIMIZES|EXTENDS|LIMITS|BASED_ON|PART_OF", "description": "..."}
  ]
}

ENTITY EXTRACTION RULES:
- EXTRACT: methodologies, frameworks, chunking methods, retrieval methods, embedding models, datasets, evaluation metrics, architectures, tools, benchmark systems, algorithms.
- DO NOT EXTRACT: random hardware specs, email addresses, URLs, author affiliations, page numbers, implementation noise, generic programming languages.
- Use SPECIFIC entity types: "chunking_method" for chunking approaches, "retrieval_method" for retrieval techniques, "embedding" for embedding models, "framework" for systems/frameworks, "benchmark" for evaluation benchmarks.
- Normalize names: use the canonical/standard name (e.g., "BM25" not "bm25 algorithm", "FAISS" not "Facebook AI Similarity Search").

RELATIONSHIP EXTRACTION RULES:
- Use STRONG semantic edge types. Never use generic "relates_to" or "references".
- USES: X uses Y as a component
- IMPROVES: X improves upon Y
- OUTPERFORMS: X achieves better results than Y
- EVALUATED_ON: method X evaluated on dataset Y
- CHUNKS_WITH: system uses chunking method Y
- RETRIEVES_WITH: system uses retrieval method Y
- EMBEDS_WITH: system uses embedding model Y
- COMPARES_WITH: X is compared against Y
- OPTIMIZES: X optimizes aspect Y
- EXTENDS: X extends/builds upon Y
- LIMITS: limitation X constrains method Y
- BASED_ON: X is based on / derived from Y
- PART_OF: X is a component of system Y

GOOD example entities:
- {"name": "Semantic Chunking", "entity_type": "chunking_method", "description": "Splits text at semantic boundaries using embedding similarity"}
- {"name": "BGE-M3", "entity_type": "embedding", "description": "Multi-lingual multi-granularity embedding model"}
- {"name": "NDCG@10", "entity_type": "metric", "description": "Normalized discounted cumulative gain at rank 10"}

BAD example entities (do NOT extract these):
- {"name": "NVIDIA A100", "entity_type": "tool", "description": "GPU used for training"}
- {"name": "Python 3.9", "entity_type": "tool", "description": "Programming language"}

STRICT EXTRACTION RULES (NO EXCEPTIONS):
1. ONLY extract information explicitly present in the provided text. NO inferences.
2. Include DIRECT QUOTES from the text to support every extraction.
3. Be precise and concise — avoid vague generalizations.
4. Cover ALL requested extraction categories thoroughly.
5. Always provide entity relationships when entities are mentioned together.
6. DO NOT HALLUCINATE — if you cannot find explicit evidence, do not extract.
7. ALWAYS include SPECIFIC NUMBERS: accuracy %, F1 score, latency (ms), dataset size, performance metrics, benchmark scores.
8. For methodology: extract HOW it works, WHAT makes it different, SPECIFIC parameters/configurations.
9. For findings: extract QUANTITATIVE results, comparisons, statistical significance.
10. For entities: use full canonical names and include version numbers if mentioned (e.g., "BERT-base-uncased", "GPT-3.5-turbo").
11. Quality over quantity: Better to extract 10 high-quality, well-supported claims than 50 vague ones.
"""


def _extract_all_for_paper(
    subtasks: list[SubTask],
    paper_id: str,
    paper_title: str,
    session_id: str | None = None,
) -> ExtractedKnowledge:
    """
    Execute ALL subtasks for a single paper in ONE LLM call.

    Retrieves chunks once, builds a combined prompt covering all subtask
    types, and parses the unified response.
    """
    settings = get_settings()
    km = get_key_manager()
    chunk_count = settings.worker_chunk_count

    # ── Build a combined search query from all subtasks ──────────────
    combined_query = " ".join(
        f"{st.prompt} {st.description}" for st in subtasks
    )
    # Truncate to avoid overly long embedding input
    if len(combined_query) > 500:
        combined_query = combined_query[:500]

    chunks = vector_search(
        query=combined_query,
        n_results=chunk_count,
        paper_id=paper_id,
    )

    if not chunks:
        logger.warning("No chunks found for paper %s", paper_id)
        return ExtractedKnowledge(
            subtask_type=SubTaskType.ENTITIES,
            paper_id=paper_id,
            paper_title=paper_title,
        )

    # ── Build context from retrieved chunks ─────────────────────────
    context_parts = []
    source_chunk_ids = []
    for chunk in chunks:
        context_parts.append(
            f"[Chunk {chunk['id']}, Page {chunk['metadata'].get('page_number', '?')}]:\n"
            f"{chunk['text']}"
        )
        source_chunk_ids.append(chunk["id"])

    context = "\n\n---\n\n".join(context_parts)

    # ── Build combined subtask instructions ─────────────────────────
    task_instructions = "\n".join(
        f"  {i}. [{st.task_type.value.upper()}] {st.description}: {st.prompt}"
        for i, st in enumerate(subtasks, 1)
    )

    user_prompt = f"""Paper: "{paper_title}" (ID: {paper_id})

Extraction Tasks (complete ALL of these):
{task_instructions}

--- Retrieved Text Chunks ---
{context}
--- End of Chunks ---

Extract structured knowledge covering ALL the above tasks from the text chunks.
Output valid JSON only."""

    # ── Call Groq LLM (via KeyManager for rotation/failover) ────────
    logger.info(
        "Worker: extracting %d tasks from '%s'",
        len(subtasks), paper_title,
    )

    raw_text = km.call_llm(
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=settings.groq_temperature,
        max_tokens=settings.worker_max_tokens,
        response_format={"type": "json_object"},
        session_id=session_id,
    )

    # ── Parse response into Pydantic model ──────────────────────────
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error("Worker: failed to parse JSON for paper %s", paper_id)
        data = {}

    entities = [
        Entity(
            name=e.get("name", ""),
            entity_type=e.get("entity_type", "concept"),
            description=e.get("description", ""),
            source_paper=paper_id,
            source_chunk_id=source_chunk_ids[0] if source_chunk_ids else "",
        )
        for e in data.get("entities", [])
        if e.get("name")
    ]

    relationships = [
        Relationship(
            source_entity=r.get("source_entity", ""),
            target_entity=r.get("target_entity", ""),
            relationship_type=r.get("relationship_type", "related_to"),
            description=r.get("description", ""),
            source_paper=paper_id,
        )
        for r in data.get("relationships", [])
        if r.get("source_entity") and r.get("target_entity")
    ]

    return ExtractedKnowledge(
        subtask_type=SubTaskType.ENTITIES,  # Combined extraction
        paper_id=paper_id,
        paper_title=paper_title,
        entities=entities,
        methodology=data.get("methodology", []),
        findings=data.get("findings", []),
        risks=data.get("risks", []),
        claims=data.get("claims", []),
        supporting_quotes=data.get("supporting_quotes", []),
        source_chunk_ids=source_chunk_ids,
        relationships=relationships,
    )


def run_worker_agents(
    subtasks: list[SubTask],
    paper_metadata: list[PaperMetadata],
    max_workers: int = 2,
    session_id: str | None = None,
) -> list[ExtractedKnowledge]:
    """
    Execute worker agents — ONE batched call per paper (not per subtask).

    Each paper gets a single LLM call covering all subtask types.
    """
    logger.info(
        "Running batched extraction: %d subtasks for %d papers = %d LLM calls",
        len(subtasks),
        len(paper_metadata),
        len(paper_metadata),  # One call per paper now!
    )

    all_extractions: list[ExtractedKnowledge] = []

    # Build job list: one per paper (batching all subtasks)
    jobs = [
        (subtasks, p.paper_id, p.title)
        for p in paper_metadata
    ]

    # Execute in parallel with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(
                _extract_all_for_paper, sts, pid, ptitle, session_id
            ): (pid, ptitle)
            for sts, pid, ptitle in jobs
        }

        for future in as_completed(future_to_job):
            paper_id, paper_title = future_to_job[future]
            try:
                extraction = future.result()
                all_extractions.append(extraction)
                msg = (
                    f"  \u21b3 Worker completed: '{paper_title[:30]}...' "
                    f"({len(extraction.entities)} entities, "
                    f"{len(extraction.findings)} findings, "
                    f"{len(extraction.relationships)} relationships)"
                )
                logger.info(
                    "Worker completed: paper %s (%d entities, %d findings, %d rels)",
                    paper_id,
                    len(extraction.entities),
                    len(extraction.findings),
                    len(extraction.relationships),
                )
                if session_id:
                    try:
                        from scholarsync.chat.mode_router import enqueue_thought
                        enqueue_thought(session_id, msg)
                    except Exception:
                        pass
            except Exception as e:
                logger.error(
                    "Worker failed: paper %s: %s",
                    paper_id,
                    e,
                )

    logger.info("All workers completed: %d extractions total", len(all_extractions))
    return all_extractions
