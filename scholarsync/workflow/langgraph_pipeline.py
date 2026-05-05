"""
LangGraph Workflow Pipeline — orchestrates the full ScholarSync multi-agent
literature review process as a state graph.

Pipeline:
  UserInput → ManagerAgent → WorkerAgents (parallel) → GraphRAG
  → CheckingAgent → (correction loop if fail) → FinalSynthesizer → Output
"""

from __future__ import annotations

import uuid
from typing import Any, TypedDict, Annotated

from langgraph.graph import StateGraph, END

from scholarsync.config.settings import get_settings
from scholarsync.utils.logger import get_logger
from scholarsync.utils.schemas import (
    PipelineStatus,
    PaperMetadata,
    WorkflowState,
    SubTask,
    ExtractedKnowledge,
    ValidationResult,
    LiteratureReview,
)
from scholarsync.agents.manager_agent import decompose_query
from scholarsync.agents.worker_agent import run_worker_agents
from scholarsync.agents.checking_agent import validate_all_extractions
from scholarsync.agents.synthesizer_agent import synthesize_review, format_review_as_markdown
from scholarsync.rag.graph_rag import (
    add_entities,
    add_relationships,
    add_paper_node,
    query_cross_paper_connections,
    query_entity_graph_summary,
)

logger = get_logger(__name__)


# ── LangGraph State Schema ──────────────────────────────────────────

class GraphState(TypedDict):
    """State that flows through the LangGraph pipeline."""
    session_id: str
    query: str
    paper_metadata: list[dict]
    status: str
    progress_messages: list[str]

    # Manager output
    subtasks: list[dict]

    # Worker output
    extractions: list[dict]

    # Validation
    validation_results: list[dict]
    correction_count: int

    # Graph data
    graph_insights: dict

    # Final output
    final_report: dict | None
    report_markdown: str

    # Errors
    errors: list[str]


# ── Node Functions ──────────────────────────────────────────────────

def manager_node(state: GraphState) -> GraphState:
    """Manager Agent: decompose query into subtasks."""
    logger.info("Pipeline: Manager Agent starting")
    state["status"] = PipelineStatus.PLANNING.value
    msg = "🧠 Manager Agent: Analyzing research query..."
    state["progress_messages"].append(msg)
    
    try:
        from scholarsync.chat.mode_router import enqueue_thought
        enqueue_thought(state["session_id"], msg)
    except Exception:
        pass

    try:
        paper_metadata = [PaperMetadata(**p) for p in state["paper_metadata"]]
        subtasks = decompose_query(state["query"], paper_metadata)
        state["subtasks"] = [st.model_dump() for st in subtasks]
        
        success_msg = f"✅ Manager Agent: Created {len(subtasks)} subtasks"
        state["progress_messages"].append(success_msg)
        try:
            from scholarsync.chat.mode_router import enqueue_thought
            for st in subtasks:
                enqueue_thought(state["session_id"], f"  ↳ Subtask: {st.task_type.value} - {st.description}")
            enqueue_thought(state["session_id"], success_msg)
        except Exception:
            pass

    except Exception as e:
        err_msg = f"❌ Manager Agent error: {str(e)}"
        logger.error("Manager Agent error: %s", e)
        state["errors"].append(f"Manager Agent error: {str(e)}")
        state["progress_messages"].append(err_msg)
        try:
            from scholarsync.chat.mode_router import enqueue_thought
            enqueue_thought(state["session_id"], err_msg)
        except Exception:
            pass

    return state


def worker_node(state: GraphState) -> GraphState:
    """Worker Agents: extract structured knowledge in parallel."""
    logger.info("Pipeline: Worker Agents starting")
    state["status"] = PipelineStatus.EXTRACTING.value
    msg = "⛏️ Worker Agents: Extracting knowledge from papers..."
    state["progress_messages"].append(msg)
    try:
        from scholarsync.chat.mode_router import enqueue_thought
        enqueue_thought(state["session_id"], msg)
    except Exception:
        pass

    try:
        subtasks = [SubTask(**st) for st in state["subtasks"]]
        paper_metadata = [PaperMetadata(**p) for p in state["paper_metadata"]]
        from scholarsync.chat.mode_router import enqueue_thought
        try:
            enqueue_thought(state["session_id"], f"  ↳ Running {len(subtasks)*len(paper_metadata)} extraction jobs in parallel...")
        except Exception:
            pass
            
        extractions = run_worker_agents(subtasks, paper_metadata, session_id=state["session_id"])
        state["extractions"] = [ext.model_dump() for ext in extractions]
        
        success_msg = f"✅ Worker Agents: Completed {len(extractions)} extractions"
        state["progress_messages"].append(success_msg)
        try:
            enqueue_thought(state["session_id"], success_msg)
        except Exception:
            pass
    except Exception as e:
        err_msg = f"❌ Worker Agents error: {str(e)}"
        logger.error("Worker Agents error: %s", e)
        state["errors"].append(f"Worker Agents error: {str(e)}")
        state["progress_messages"].append(err_msg)
        try:
            from scholarsync.chat.mode_router import enqueue_thought
            enqueue_thought(state["session_id"], err_msg)
        except Exception:
            pass

    return state


def graph_rag_node(state: GraphState) -> GraphState:
    """GraphRAG: build knowledge graph from extracted entities."""
    logger.info("Pipeline: GraphRAG starting")
    state["status"] = PipelineStatus.BUILDING_GRAPH.value
    state["progress_messages"].append("🔗 GraphRAG: Building knowledge graph...")

    try:
        extractions = [ExtractedKnowledge(**ext) for ext in state["extractions"]]
        paper_metadata = [PaperMetadata(**p) for p in state["paper_metadata"]]

        # Add paper nodes
        for meta in paper_metadata:
            try:
                add_paper_node(meta.paper_id, meta.title, meta.authors, meta.year)
            except Exception as e:
                logger.warning("Could not add paper node %s: %s", meta.paper_id, e)

        # Add entities and relationships from extractions
        all_entities = []
        all_relationships = []
        for ext in extractions:
            all_entities.extend(ext.entities)
            all_relationships.extend(ext.relationships)

        try:
            entity_count = add_entities(all_entities)
            rel_count = add_relationships(all_relationships)
        except Exception as e:
            logger.warning("Graph storage error (Neo4j may not be available): %s", e)
            entity_count = len(all_entities)
            rel_count = len(all_relationships)

        # Query cross-paper insights
        graph_insights = {"cross_paper_connections": [], "summary": {}}
        try:
            cross_paper = query_cross_paper_connections()
            graph_summary = query_entity_graph_summary()
            graph_insights = {
                "cross_paper_connections": cross_paper,
                "summary": graph_summary,
            }
        except Exception as e:
            logger.warning("Graph query failed (Neo4j may not be available): %s", e)

        state["graph_insights"] = graph_insights
        state["progress_messages"].append(
            f"✅ GraphRAG: {entity_count} entities, {rel_count} relationships mapped"
        )
    except Exception as e:
        logger.error("GraphRAG error: %s", e)
        state["errors"].append(f"GraphRAG error: {str(e)}")
        state["graph_insights"] = {"cross_paper_connections": [], "summary": {}}
        state["progress_messages"].append(f"⚠️ GraphRAG: Continued without graph ({str(e)})")

    return state


def checking_node(state: GraphState) -> GraphState:
    """Checking Agent: validate extractions."""
    logger.info("Pipeline: Checking Agent starting")
    state["status"] = PipelineStatus.VALIDATING.value
    state["progress_messages"].append("🔍 Checking Agent: Validating extractions...")

    try:
        extractions = [ExtractedKnowledge(**ext) for ext in state["extractions"]]
        validation_results = validate_all_extractions(extractions)
        state["validation_results"] = [vr.model_dump() for vr in validation_results]

        avg_score = (
            sum(vr.overall_score for vr in validation_results) / len(validation_results)
            if validation_results
            else 0.0
        )
        valid_count = sum(1 for vr in validation_results if vr.is_valid)

        state["progress_messages"].append(
            f"✅ Checking Agent: Score {avg_score:.2f} — "
            f"{valid_count}/{len(validation_results)} passed"
        )
    except Exception as e:
        logger.error("Checking Agent error: %s", e)
        state["errors"].append(f"Checking Agent error: {str(e)}")
        state["progress_messages"].append(f"❌ Checking Agent error: {str(e)}")

    return state


def should_correct(state: GraphState) -> str:
    """Conditional edge: decide if correction loop is needed."""
    settings = get_settings()
    validation_results = state.get("validation_results", [])
    correction_count = state.get("correction_count", 0)

    if not validation_results:
        return "synthesize"

    results = [ValidationResult(**vr) for vr in validation_results]
    avg_score = sum(r.overall_score for r in results) / len(results)

    if avg_score >= settings.validation_threshold:
        logger.info("Validation passed (%.2f >= %.2f)", avg_score, settings.validation_threshold)
        return "synthesize"

    if correction_count >= settings.max_correction_loops:
        logger.warning(
            "Max corrections reached (%d), proceeding to synthesis", correction_count
        )
        state["progress_messages"].append(
            f"⚠️ Max correction loops reached ({correction_count}). Proceeding with best results."
        )
        return "synthesize"

    logger.info(
        "Validation failed (%.2f < %.2f), correction loop #%d",
        avg_score,
        settings.validation_threshold,
        correction_count + 1,
    )
    return "correct"


def correction_node(state: GraphState) -> GraphState:
    """Correction loop: re-run workers with feedback from checking agent."""
    state["correction_count"] = state.get("correction_count", 0) + 1
    state["status"] = PipelineStatus.CORRECTING.value
    state["progress_messages"].append(
        f"🔄 Correction Loop #{state['correction_count']}: Re-extracting with feedback..."
    )

    logger.info("Pipeline: Correction Loop #%d", state["correction_count"])

    # Re-run workers (they will get potentially different chunks due to random variance)
    try:
        subtasks = [SubTask(**st) for st in state["subtasks"]]
        paper_metadata = [PaperMetadata(**p) for p in state["paper_metadata"]]
        extractions = run_worker_agents(subtasks, paper_metadata)
        state["extractions"] = [ext.model_dump() for ext in extractions]
        state["progress_messages"].append(
            f"✅ Correction: Re-extracted {len(extractions)} items"
        )
    except Exception as e:
        logger.error("Correction error: %s", e)
        state["errors"].append(f"Correction error: {str(e)}")

    return state


def synthesizer_node(state: GraphState) -> GraphState:
    """Synthesizer Agent: produce final literature review."""
    logger.info("Pipeline: Synthesizer starting")
    state["status"] = PipelineStatus.SYNTHESIZING.value
    state["progress_messages"].append("📝 Synthesizer: Generating literature review...")

    try:
        extractions = [ExtractedKnowledge(**ext) for ext in state["extractions"]]
        validation_results = [ValidationResult(**vr) for vr in state.get("validation_results", [])]
        paper_metadata = [PaperMetadata(**p) for p in state["paper_metadata"]]
        graph_insights = state.get("graph_insights")

        review = synthesize_review(
            query=state["query"],
            extractions=extractions,
            validation_results=validation_results,
            paper_metadata=paper_metadata,
            graph_insights=graph_insights,
        )

        report_md = format_review_as_markdown(review)

        state["final_report"] = review.model_dump()
        state["report_markdown"] = report_md
        state["status"] = PipelineStatus.COMPLETED.value
        state["progress_messages"].append("✅ Literature review generated successfully!")
    except Exception as e:
        logger.error("Synthesizer error: %s", e)
        state["errors"].append(f"Synthesizer error: {str(e)}")
        state["status"] = PipelineStatus.FAILED.value
        state["progress_messages"].append(f"❌ Synthesizer error: {str(e)}")

    return state


# ── Build the LangGraph Pipeline ────────────────────────────────────

def build_pipeline() -> StateGraph:
    """
    Construct the full LangGraph workflow.

    Graph:
      manager → workers → graph_rag → checking → (correct | synthesize)
      correct → checking  (loop back)
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("manager", manager_node)
    workflow.add_node("workers", worker_node)
    workflow.add_node("graph_rag", graph_rag_node)
    workflow.add_node("checking", checking_node)
    workflow.add_node("correction", correction_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Define edges
    workflow.set_entry_point("manager")
    workflow.add_edge("manager", "workers")
    workflow.add_edge("workers", "graph_rag")
    workflow.add_edge("graph_rag", "checking")

    # Conditional: checking → synthesize or correct
    workflow.add_conditional_edges(
        "checking",
        should_correct,
        {
            "synthesize": "synthesizer",
            "correct": "correction",
        },
    )

    # Correction loops back to checking
    workflow.add_edge("correction", "checking")

    # Synthesizer is the end
    workflow.add_edge("synthesizer", END)

    logger.info("LangGraph pipeline built successfully")
    return workflow


def run_pipeline(
    session_id: str,
    query: str,
    paper_metadata: list[PaperMetadata],
) -> GraphState:
    """
    Execute the full pipeline synchronously.

    Returns the final state.
    """
    logger.info("Starting pipeline for session %s: '%s'", session_id, query)

    # Build and compile the graph
    workflow = build_pipeline()
    app = workflow.compile()

    # Initial state
    initial_state: GraphState = {
        "session_id": session_id,
        "query": query,
        "paper_metadata": [p.model_dump() for p in paper_metadata],
        "status": PipelineStatus.PENDING.value,
        "progress_messages": ["🚀 Pipeline started!"],
        "subtasks": [],
        "extractions": [],
        "validation_results": [],
        "correction_count": 0,
        "graph_insights": {},
        "final_report": None,
        "report_markdown": "",
        "errors": [],
    }

    # Run the graph
    final_state = app.invoke(initial_state)

    logger.info("Pipeline completed with status: %s", final_state.get("status"))
    return final_state
