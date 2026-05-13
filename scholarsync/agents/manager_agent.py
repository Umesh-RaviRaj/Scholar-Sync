"""
Manager Agent (Orchestrator) — decomposes the user's research query into
subtasks and generates prompts for worker agents.

Uses Groq LLM with structured output to produce a list of SubTask objects.
"""

from __future__ import annotations

import json
import uuid

from scholarsync.config.settings import get_settings
from scholarsync.chat.key_manager import get_key_manager
from scholarsync.utils.logger import get_logger
from scholarsync.utils.schemas import SubTask, SubTaskType, PaperMetadata

logger = get_logger(__name__)


PLANNING_SYSTEM_PROMPT = """You are the Manager Agent of ScholarSync, a multi-agent literature review system.

Your role is to decompose a user's research query into specific subtasks for worker agents.

Each subtask should target one of these extraction types:
- entities: Extract named entities (methods, datasets, metrics, concepts, tools)
- methodology: Extract research methodologies, experimental designs, approaches
- findings: Extract key results, outcomes, conclusions
- risks: Extract limitations, risks, challenges, threats to validity
- claims: Extract specific claims made by the authors

For each subtask, provide:
1. A clear task_type from the list above
2. A detailed description of what to extract
3. A specific prompt that a worker agent will use to query the papers

You MUST output valid JSON — an array of subtask objects.

Example output:
[
  {
    "task_type": "entities",
    "description": "Extract all named methods, algorithms, frameworks, and tools mentioned",
    "prompt": "Identify and list all named methods, algorithms, frameworks, datasets, and tools mentioned in this research paper. For each entity, provide its name, type, and a brief description of how it is used in the paper."
  },
  {
    "task_type": "findings",
    "description": "Extract key experimental results and performance metrics",
    "prompt": "Extract the key findings, experimental results, and performance metrics reported in this paper. Include specific numbers, comparisons, and statistical significance where available."
  }
]

CRITICAL REQUIREMENTS:
- Generate between 6 and 8 subtasks to THOROUGHLY cover the research query
- ALWAYS include: entities, methodology, findings, risks, AND claims
- Each subtask must request SPECIFIC NUMBERS, METRICS, and QUANTITATIVE DATA
- Each subtask must emphasize extracting EVIDENCE and CITATIONS for evaluation
- Prompts must be DETAILED and demand comprehensive extraction

This ensures high-quality, evidence-based synthesis with good source diversity.
"""


def decompose_query(
    query: str,
    paper_metadata: list[PaperMetadata],
    session_id: str | None = None,
) -> list[SubTask]:
    """
    Use the Manager Agent to decompose a research query into subtasks.

    Parameters
    ----------
    query : str
        The user's research query.
    paper_metadata : list[PaperMetadata]
        Metadata for all uploaded papers.
    session_id : str, optional
        Pipeline session ID for token budget tracking.

    Returns
    -------
    list[SubTask]
        A list of subtasks for worker agents.
    """
    settings = get_settings()
    km = get_key_manager()

    paper_context = "\n".join(
        f"- Paper [{m.paper_id}]: \"{m.title}\" by {', '.join(m.authors) or 'Unknown'}"
        for m in paper_metadata
    )

    user_prompt = f"""Research Query: {query}

Available Papers:
{paper_context}

Decompose this research query into subtasks for worker agents. Each worker will process
all papers for their assigned subtask type. Output a JSON array of subtask objects."""

    logger.info("Manager Agent: decomposing query into subtasks")

    raw_text = km.call_llm(
        messages=[
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=settings.groq_temperature,
        max_tokens=settings.manager_max_tokens,
        response_format={"type": "json_object"},
        session_id=session_id,
    )

    # Parse the JSON response
    try:
        parsed = json.loads(raw_text)
        # Handle both {"subtasks": [...]} and [...] formats
        if isinstance(parsed, dict):
            tasks_data = parsed.get("subtasks", parsed.get("tasks", []))
        elif isinstance(parsed, list):
            tasks_data = parsed
        else:
            tasks_data = []
    except json.JSONDecodeError:
        logger.error("Manager Agent: failed to parse JSON response")
        tasks_data = _default_subtasks()

    # Convert to SubTask models
    paper_ids = [m.paper_id for m in paper_metadata]
    subtasks: list[SubTask] = []

    for i, task_data in enumerate(tasks_data):
        task_type_str = task_data.get("task_type", "entities")
        try:
            task_type = SubTaskType(task_type_str)
        except ValueError:
            task_type = SubTaskType.ENTITIES

        subtasks.append(
            SubTask(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                task_type=task_type,
                description=task_data.get("description", ""),
                assigned_paper_ids=paper_ids,  # All workers process all papers
                prompt=task_data.get("prompt", ""),
                status="pending",
            )
        )

    if not subtasks:
        logger.warning("Manager Agent: no subtasks generated, using defaults")
        subtasks = _build_default_subtasks(paper_ids)

    logger.info("Manager Agent: created %d subtasks", len(subtasks))
    return subtasks


def _default_subtasks() -> list[dict]:
    """Fallback subtask definitions."""
    return [
        {
            "task_type": "entities",
            "description": "Extract named entities: methods, datasets, tools, concepts",
            "prompt": "Identify all named entities including methods, algorithms, datasets, tools, metrics, and key concepts. For each, provide the name, type, and how it's used.",
        },
        {
            "task_type": "methodology",
            "description": "Extract research methodology and experimental design",
            "prompt": "Describe the research methodology, experimental setup, data collection methods, and evaluation approaches used in this work.",
        },
        {
            "task_type": "findings",
            "description": "Extract key findings, results, and conclusions",
            "prompt": "List the key findings, experimental results, performance metrics, and major conclusions. Include specific numbers and comparisons.",
        },
        {
            "task_type": "risks",
            "description": "Extract limitations, risks, and future work",
            "prompt": "Identify limitations, potential risks, threats to validity, biases, and suggested future work directions mentioned in this paper.",
        },
        {
            "task_type": "claims",
            "description": "Extract specific claims and assertions",
            "prompt": "Extract specific claims made by the authors, including assertions about their method's superiority, novel contributions, and comparative statements.",
        },
    ]


def _build_default_subtasks(paper_ids: list[str]) -> list[SubTask]:
    """Build default SubTask objects when LLM parsing fails."""
    defaults = _default_subtasks()
    return [
        SubTask(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            task_type=SubTaskType(d["task_type"]),
            description=d["description"],
            assigned_paper_ids=paper_ids,
            prompt=d["prompt"],
            status="pending",
        )
        for d in defaults
    ]
