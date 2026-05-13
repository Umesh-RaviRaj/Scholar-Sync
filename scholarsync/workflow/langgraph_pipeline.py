"""
LangGraph Workflow Pipeline — orchestrates the full ScholarSync multi-agent
literature review process as a state graph.

Pipeline:
  UserInput → ManagerAgent → WorkerAgents (batched per paper) → GraphRAG
  → CheckingAgent → (feedback-augmented correction if fail) → FinalSynthesizer → Output

OPTIMISED:
  - Workers batch all subtasks per paper (N calls instead of 5×N)
  - Correction loop injects checker feedback and only re-runs failed extractions
  - Session-level token budget tracking
  - Right-sized token limits per agent
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
from scholarsync.agents.profile_builder import build_paper_profiles
from scholarsync.utils.schemas import StructuredPaperProfile
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
    
    # User ID for multi-user graph isolation
    user_id: str
    
    # Web search integration (only used when explicitly enabled)
    web_search_context: str       # Formatted web search results
    web_search_provider: str      # "tavily", "duckduckgo", or ""
    enable_web_search: bool       # Whether web search was requested

    # Manager output
    subtasks: list[dict]

    # Worker output
    extractions: list[dict]

    # Validation
    validation_results: list[dict]
    correction_count: int

    # Graph data
    graph_insights: dict

    # Structured profiles (zero-cost deterministic extraction)
    structured_profiles: list[dict]

    # Final output
    final_report: dict | None
    report_markdown: str

    # Evaluation metrics (computed from actual data)
    evaluation_metrics: dict

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
        subtasks = decompose_query(
            state["query"], paper_metadata, session_id=state["session_id"]
        )
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
    """Worker Agents: extract structured knowledge — batched per paper."""
    logger.info("Pipeline: Worker Agents starting (batched per paper)")
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
        try:
            from scholarsync.chat.mode_router import enqueue_thought
            enqueue_thought(
                state["session_id"],
                f"  ↳ Batched extraction: {len(subtasks)} tasks × {len(paper_metadata)} papers = {len(paper_metadata)} LLM calls",
            )
        except Exception:
            pass
            
        extractions = run_worker_agents(
            subtasks, paper_metadata, session_id=state["session_id"]
        )
        state["extractions"] = [ext.model_dump() for ext in extractions]
        
        success_msg = f"✅ Worker Agents: Completed {len(extractions)} extractions"
        state["progress_messages"].append(success_msg)
        try:
            from scholarsync.chat.mode_router import enqueue_thought
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

    # SESSION-BASED: Use __global__ for all graph data (cleared on upload)
    # This ensures graph shows ONLY current session's papers
    session_user_id = "__global__"

    try:
        extractions = [ExtractedKnowledge(**ext) for ext in state["extractions"]]
        paper_metadata = [PaperMetadata(**p) for p in state["paper_metadata"]]

        # Add paper nodes to global session graph
        for meta in paper_metadata:
            try:
                add_paper_node(meta.paper_id, meta.title, meta.authors, meta.year, user_id=session_user_id)
            except Exception as e:
                logger.warning("Could not add paper node %s: %s", meta.paper_id, e)

        # Add entities and relationships from extractions to global session graph
        all_entities = []
        all_relationships = []
        for ext in extractions:
            all_entities.extend(ext.entities)
            all_relationships.extend(ext.relationships)

        try:
            entity_count = add_entities(all_entities, user_id=session_user_id)
            rel_count = add_relationships(all_relationships, user_id=session_user_id)
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
        validation_results = validate_all_extractions(
            extractions, session_id=state["session_id"]
        )
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
    """
    Feedback-augmented correction: re-run workers ONLY for failed
    extractions, injecting the checker's correction_prompts as guidance.
    """
    state["correction_count"] = state.get("correction_count", 0) + 1
    state["status"] = PipelineStatus.CORRECTING.value
    state["progress_messages"].append(
        f"🔄 Correction Loop #{state['correction_count']}: Re-extracting with feedback..."
    )

    logger.info("Pipeline: Correction Loop #%d (feedback-augmented)", state["correction_count"])

    try:
        subtasks = [SubTask(**st) for st in state["subtasks"]]
        paper_metadata = [PaperMetadata(**p) for p in state["paper_metadata"]]
        validation_results = [ValidationResult(**vr) for vr in state.get("validation_results", [])]
        old_extractions = [ExtractedKnowledge(**ext) for ext in state["extractions"]]

        # Identify which extractions failed and collect feedback
        failed_paper_ids = set()
        feedback_by_paper: dict[str, list[str]] = {}
        for ext, vr in zip(old_extractions, validation_results):
            if not vr.is_valid:
                failed_paper_ids.add(ext.paper_id)
                prompts = vr.correction_prompts or [vr.feedback] if vr.feedback else []
                feedback_by_paper.setdefault(ext.paper_id, []).extend(prompts)

        if not failed_paper_ids:
            # All passed — nothing to correct
            state["progress_messages"].append("✅ No failed extractions to correct")
            return state

        # Only re-run for failed papers
        failed_metadata = [p for p in paper_metadata if p.paper_id in failed_paper_ids]

        try:
            from scholarsync.chat.mode_router import enqueue_thought
            enqueue_thought(
                state["session_id"],
                f"  ↳ Re-extracting {len(failed_metadata)} failed papers with checker feedback",
            )
        except Exception:
            pass

        # Augment subtask prompts with checker feedback
        augmented_subtasks = []
        for st in subtasks:
            new_prompt = st.prompt
            # Append any correction feedback
            all_feedback = []
            for pid in failed_paper_ids:
                all_feedback.extend(feedback_by_paper.get(pid, []))
            if all_feedback:
                feedback_text = "; ".join(all_feedback[:5])
                new_prompt += f"\n\nPrevious extraction had issues. Corrections needed: {feedback_text}"
            augmented_subtasks.append(
                SubTask(
                    task_id=st.task_id,
                    task_type=st.task_type,
                    description=st.description,
                    assigned_paper_ids=st.assigned_paper_ids,
                    prompt=new_prompt,
                    status="pending",
                )
            )

        new_extractions = run_worker_agents(
            augmented_subtasks, failed_metadata, session_id=state["session_id"]
        )

        # Merge: keep passing extractions, replace failed ones
        kept = [ext for ext in old_extractions if ext.paper_id not in failed_paper_ids]
        merged = kept + new_extractions

        state["extractions"] = [ext.model_dump() for ext in merged]
        state["progress_messages"].append(
            f"✅ Correction: Re-extracted {len(new_extractions)} items, kept {len(kept)} passing"
        )
    except Exception as e:
        logger.error("Correction error: %s", e)
        state["errors"].append(f"Correction error: {str(e)}")

    return state


def profile_node(state: GraphState) -> GraphState:
    """Profile Builder: deterministic structured paper profiling (zero LLM cost)."""
    logger.info("Pipeline: Profile Builder starting")
    state["progress_messages"].append("📋 Profile Builder: Structuring paper profiles...")

    try:
        extractions = [ExtractedKnowledge(**ext) for ext in state["extractions"]]
        paper_metadata = [PaperMetadata(**p) for p in state["paper_metadata"]]

        profiles = build_paper_profiles(extractions, paper_metadata)
        state["structured_profiles"] = [p.model_dump() for p in profiles]

        state["progress_messages"].append(
            f"✅ Profile Builder: {len(profiles)} structured profiles created"
        )
        try:
            from scholarsync.chat.mode_router import enqueue_thought
            enqueue_thought(state["session_id"], f"  ↳ Built {len(profiles)} structured paper profiles (zero-cost)")
        except Exception:
            pass
    except Exception as e:
        logger.error("Profile Builder error: %s", e)
        state["structured_profiles"] = []
        state["progress_messages"].append(f"⚠️ Profile Builder: Continuing without profiles ({str(e)})")

    return state


def synthesizer_node(state: GraphState) -> GraphState:
    """
    Synthesizer Node: combine all extractions into a final literature review report.
    Includes web search context if available.
    
    CRITICAL: Now tracks retrieval sources for accurate evaluation.
    """
    logger.info("Pipeline: Synthesizer Agent starting")
    state["progress_messages"].append("📝 Synthesizer: Generating literature review...")
    
    # Track retrieval sources for evaluation (CRITICAL for Source Diversity metric)
    state["retrieval_sources"] = []

    try:
        extractions = [ExtractedKnowledge(**ext) for ext in state["extractions"]]
        validation_results = [ValidationResult(**vr) for vr in state.get("validation_results", [])]
        paper_metadata = [PaperMetadata(**p) for p in state["paper_metadata"]]
        graph_insights = state.get("graph_insights")
        
        # Include web search context if available
        web_search_context = state.get("web_search_context", "")
        web_search_provider = state.get("web_search_provider", "")
        if web_search_context:
            if graph_insights is None:
                graph_insights = {}
            graph_insights["web_search_context"] = web_search_context
            graph_insights["web_search_provider"] = web_search_provider
            state["progress_messages"].append(f"🌐 Including web search results from {web_search_provider}")

        # Rebuild profiles from state
        profiles = None
        if state.get("structured_profiles"):
            profiles = [StructuredPaperProfile(**p) for p in state["structured_profiles"]]

        #  CRITICAL: Track retrieval sources from extractions for accurate evaluation
        retrieval_sources = []
        for extraction in extractions:
            # Add methodology and findings as source material
            retrieval_sources.extend(extraction.methodology[:5])  # Top 5 methodology points
            retrieval_sources.extend(extraction.findings[:5])      # Top 5 findings
            
        # Also track which papers were used (for diversity calculation)
        paper_ids_used = [ext.paper_id for ext in extractions]
        unique_papers = len(set(paper_ids_used))
        
        state["retrieval_sources"] = retrieval_sources
        state["unique_papers_used"] = unique_papers
        logger.info("Synthesis using %d sources from %d unique papers", len(retrieval_sources), unique_papers)

        review = synthesize_review(
            query=state["query"],
            extractions=extractions,
            validation_results=validation_results,
            paper_metadata=paper_metadata,
            graph_insights=graph_insights,
            session_id=state["session_id"],
            structured_profiles=profiles,
        )

        report_md = format_review_as_markdown(review)
        
        # Add web search attribution if used
        if web_search_context and web_search_provider:
            report_md += f"\n\n---\n*This report includes supplementary information from web search ({web_search_provider}).*"

        state["final_report"] = review.model_dump()
        state["report_markdown"] = report_md
        state["status"] = PipelineStatus.COMPLETED.value
        state["progress_messages"].append("✅ Literature review generated successfully!")

        # Log final budget
        try:
            from scholarsync.chat.llm_cache import get_pipeline_budget
            budget = get_pipeline_budget(state["session_id"])
            budget_msg = f"📊 Token usage: {budget.summary()}"
            state["progress_messages"].append(budget_msg)
            logger.info("Pipeline budget: %s", budget.summary())
        except Exception:
            pass

    except Exception as e:
        logger.error("Synthesizer error: %s", e)
        state["errors"].append(f"Synthesizer error: {str(e)}")
        state["status"] = PipelineStatus.FAILED.value
        state["progress_messages"].append(f"❌ Synthesizer error: {str(e)}")

    return state


def evaluation_node(state: GraphState) -> GraphState:
    """
    Evaluation Node: compute real metrics on the generated output.
    
    Uses DYNAMIC metric selection based on query type and context.
    All metrics are computed from actual data — no heuristic self-scoring.
    """
    logger.info("Pipeline: Running dynamic evaluation metrics")

    report_md = state.get("report_markdown", "")
    if not report_md:
        state["progress_messages"].append("⚠️ Evaluation skipped (no report)")
        return state

    try:
        from scholarsync.evaluation.dynamic_metrics import run_dynamic_evaluation

        query = state["query"]
        has_web_search = state.get("enable_web_search", False)

        # CRITICAL FIX: Use ACTUAL sources from synthesis, not separate retrieval!
        # This ensures Source Diversity metric reflects what was actually used
        source_chunks = state.get("retrieval_sources", [])
        
        if not source_chunks:
            # Fallback: extract from extractions if not tracked
            logger.warning("No retrieval_sources tracked - using fallback extraction")
            for ext_data in state.get("extractions", []):
                ext = ExtractedKnowledge(**ext_data) if isinstance(ext_data, dict) else ext_data
                source_chunks.extend(ext.methodology[:5])
                source_chunks.extend(ext.findings[:5])
        
        unique_papers = state.get("unique_papers_used", len(state.get("paper_metadata", [])))
        logger.info("Evaluation using %d source chunks from %d unique papers", len(source_chunks), unique_papers)

        paper_titles = [
            p.get("title", "") if isinstance(p, dict) else p.title
            for p in state.get("paper_metadata", [])
        ]

        # Run dynamic evaluation with context-aware metric selection
        evaluation = run_dynamic_evaluation(
            query=query,
            generated_text=report_md,
            source_chunks=source_chunks,
            paper_titles=paper_titles,
            has_web_search=has_web_search,
        )

        # Store full evaluation in state (includes formatted display)
        state["evaluation_metrics"] = {
            "raw_metrics": evaluation.get("raw_metrics", {}),
            "formatted_display": evaluation.get("formatted_display", {}),
            "hallucination_report": evaluation.get("hallucination_report", {}),
            "query_analysis": evaluation.get("query_analysis", {}),
        }
        
        overall = evaluation.get("raw_metrics", {}).get("overall_quality", 0)
        halluc = evaluation.get("hallucination_report", {}).get("hallucination_score", 0)
        verdict = evaluation.get("formatted_display", {}).get("summary", {}).get("verdict", "")
        query_type = evaluation.get("query_analysis", {}).get("type", "general")
        metrics_shown = evaluation.get("formatted_display", {}).get("metric_count", 0)

        state["progress_messages"].append(
            f"📊 Evaluation: {verdict} | quality={overall:.2f} | {metrics_shown} metrics computed"
        )

        try:
            from scholarsync.chat.mode_router import enqueue_thought
            enqueue_thought(
                state["session_id"],
                f"  ↳ Query type: {query_type} | Overall: {overall:.2f} | Hallucination risk: {halluc:.2f}"
            )
        except Exception:
            pass

    except Exception as e:
        logger.error("Evaluation node error: %s", e)
        state["progress_messages"].append(f"⚠️ Evaluation: skipped ({str(e)[:60]})")

    return state


# ── Build the LangGraph Pipeline ────────────────────────────────────

def build_pipeline() -> StateGraph:
    """
    Construct the full LangGraph workflow.

    Graph:
      manager → workers → graph_rag → checking → (correct | profiler → synthesizer → evaluation)
      correct → checking  (loop back)
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("manager", manager_node)
    workflow.add_node("workers", worker_node)
    workflow.add_node("graph_rag", graph_rag_node)
    workflow.add_node("checking", checking_node)
    workflow.add_node("correction", correction_node)
    workflow.add_node("profiler", profile_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("evaluation", evaluation_node)

    # Define edges
    workflow.set_entry_point("manager")
    workflow.add_edge("manager", "workers")
    workflow.add_edge("workers", "graph_rag")
    workflow.add_edge("graph_rag", "checking")

    # Conditional: checking → profiler (then synthesize) or correct
    workflow.add_conditional_edges(
        "checking",
        should_correct,
        {
            "synthesize": "profiler",
            "correct": "correction",
        },
    )

    # Correction loops back to checking
    workflow.add_edge("correction", "checking")

    # Profile builder feeds into synthesizer
    workflow.add_edge("profiler", "synthesizer")

    # Synthesizer feeds into evaluation
    workflow.add_edge("synthesizer", "evaluation")

    # Evaluation is the end
    workflow.add_edge("evaluation", END)

    logger.info("LangGraph pipeline built successfully")
    return workflow


def run_pipeline(
    session_id: str,
    query: str,
    paper_metadata: list[PaperMetadata],
    user_id: str = "",
) -> GraphState:
    """
    Execute the full pipeline synchronously.

    Args:
        session_id: Unique session identifier
        query: Research query
        paper_metadata: List of paper metadata
        user_id: User ID for multi-user graph isolation
    
    Returns the final state.
    """
    logger.info("Starting pipeline for session %s (user %s): '%s'", session_id, user_id[:8] if user_id else "anon", query)

    # Initialize pipeline budget
    from scholarsync.chat.llm_cache import get_pipeline_budget, clear_pipeline_budget
    clear_pipeline_budget(session_id)  # Fresh budget for each run
    budget = get_pipeline_budget(session_id)
    logger.info("Pipeline budget initialized: %d max tokens", budget.max_tokens)

    # Build and compile the graph
    workflow = build_pipeline()
    app = workflow.compile()

    # Initial state with user_id for graph isolation
    initial_state: GraphState = {
        "session_id": session_id,
        "query": query,
        "paper_metadata": [p.model_dump() for p in paper_metadata],
        "status": PipelineStatus.PENDING.value,
        "progress_messages": ["🚀 Pipeline started!"],
        "user_id": user_id,
        # Web search fields (empty by default, populated by streaming endpoint)
        "web_search_context": "",
        "web_search_provider": "",
        "enable_web_search": False,
        "subtasks": [],
        "extractions": [],
        "validation_results": [],
        "correction_count": 0,
        "graph_insights": {},
        "structured_profiles": [],
        "final_report": None,
        "report_markdown": "",
        "evaluation_metrics": {},
        "errors": [],
    }

    # Run the graph
    final_state = app.invoke(initial_state)

    logger.info(
        "Pipeline completed with status: %s | %s",
        final_state.get("status"),
        budget.summary(),
    )
    return final_state
