"""
Worker Agent — reads assigned document chunks via RAG retrieval and
extracts structured knowledge (entities, methodology, findings, risks,
claims) using Groq LLM with Pydantic-structured output.

Supports parallel execution across multiple subtasks.
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

Your job is to extract structured knowledge from research paper text based on a specific subtask.

You MUST output valid JSON matching this schema:
{
  "entities": [
    {"name": "...", "entity_type": "method|dataset|metric|concept|tool|author", "description": "..."}
  ],
  "methodology": ["description of method 1", "..."],
  "findings": ["finding 1", "..."],
  "risks": ["risk/limitation 1", "..."],
  "claims": ["claim 1", "..."],
  "supporting_quotes": ["direct quote from text 1", "..."],
  "relationships": [
    {"source_entity": "...", "target_entity": "...", "relationship_type": "uses|compares_with|improves_upon|based_on|evaluated_on", "description": "..."}
  ]
}

Rules:
1. Only extract information actually present in the provided text.
2. Include direct quotes from the text to support extractions.
3. Be precise and concise — avoid vague generalizations.
4. Focus on the subtask type requested — prioritize that extraction category.
5. Always provide entity relationships when entities are mentioned together.
6. Do NOT hallucinate — only extract what is explicitly stated.
"""


def extract_from_paper(
    subtask: SubTask,
    paper_id: str,
    paper_title: str,
) -> ExtractedKnowledge:
    """
    Execute a single extraction subtask for one paper.

    Uses RAG to retrieve relevant chunks, then LLM to extract structured knowledge.
    """
    settings = get_settings()
    km = get_key_manager()

    # ── Retrieve relevant chunks via vector search ──────────────────
    search_query = f"{subtask.prompt} {subtask.description}"
    chunks = vector_search(
        query=search_query,
        n_results=15,
        paper_id=paper_id,
    )

    if not chunks:
        logger.warning("No chunks found for paper %s, subtask %s", paper_id, subtask.task_type)
        return ExtractedKnowledge(
            subtask_type=subtask.task_type,
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

    user_prompt = f"""Paper: "{paper_title}" (ID: {paper_id})

Subtask: {subtask.task_type.value} — {subtask.description}

Specific Instructions: {subtask.prompt}

--- Retrieved Text Chunks ---
{context}
--- End of Chunks ---

Extract structured knowledge from the above text chunks. Focus on {subtask.task_type.value}.
Output valid JSON only."""

    # ── Call Groq LLM (via KeyManager for rotation/failover) ────────
    logger.info("Worker: extracting %s from '%s'", subtask.task_type.value, paper_title)

    raw_text = km.call_llm(
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=settings.groq_temperature,
        max_tokens=settings.groq_max_tokens,
        response_format={"type": "json_object"},
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
        subtask_type=subtask.task_type,
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
    Execute all worker agents in parallel.

    Each (subtask, paper) combination is processed independently.
    """
    logger.info(
        "Running %d subtasks × %d papers = %d worker jobs",
        len(subtasks),
        len(paper_metadata),
        len(subtasks) * len(paper_metadata),
    )

    paper_map = {p.paper_id: p for p in paper_metadata}
    all_extractions: list[ExtractedKnowledge] = []

    # Build job list: each subtask × each paper
    jobs = []
    for subtask in subtasks:
        for paper_id in subtask.assigned_paper_ids:
            paper = paper_map.get(paper_id)
            if paper:
                jobs.append((subtask, paper_id, paper.title))

    # Execute in parallel with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(extract_from_paper, st, pid, ptitle): (st, pid)
            for st, pid, ptitle in jobs
        }

        for future in as_completed(future_to_job):
            subtask, paper_id = future_to_job[future]
            try:
                extraction = future.result()
                all_extractions.append(extraction)
                msg = f"  ↳ Worker completed: {subtask.task_type.value} for paper '{paper_map[paper_id].title[:20]}...' ({len(extraction.entities)} entities, {len(extraction.findings)} findings)"
                logger.info(
                    "Worker completed: %s for paper %s (%d entities, %d findings)",
                    subtask.task_type.value,
                    paper_id,
                    len(extraction.entities),
                    len(extraction.findings),
                )
                if session_id:
                    try:
                        from scholarsync.chat.mode_router import enqueue_thought
                        enqueue_thought(session_id, msg)
                    except Exception:
                        pass
            except Exception as e:
                logger.error(
                    "Worker failed: %s for paper %s: %s",
                    subtask.task_type.value,
                    paper_id,
                    e,
                )

    logger.info("All workers completed: %d extractions total", len(all_extractions))
    return all_extractions
