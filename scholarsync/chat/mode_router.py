"""
Mode Router — routes chat messages through either Normal or Deep Research
pipeline based on the user's selected mode.

Features:
- Intent detection (greeting vs simple vs complex)
- Response length control based on query type
- Streaming support via generators
- Context bypass for greetings/small talk
"""

from __future__ import annotations

import json
import re
import asyncio
from typing import AsyncGenerator, Any

from scholarsync.chat.key_manager import get_key_manager
from scholarsync.chat.graphrag_service import get_context
from scholarsync.config.settings import get_settings
from scholarsync.utils.logger import get_logger
from scholarsync.research.retrieval_fusion import fuse_retrieval

logger = get_logger(__name__)


# ── Intent Classification ────────────────────────────────────────────

GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|howdy|hiya|yo|sup|good\s*(morning|afternoon|evening|night)|"
    r"what'?s\s*up|how\s*are\s*you|how\s*do\s*you\s*do|greetings|namaste|hola|"
    r"thanks?(\s*you)?|thank\s*you|bye|goodbye|see\s*ya|nice\s*to\s*meet|"
    r"how'?s?\s*it\s*going|what'?s\s*good|hey\s*there)[\s!?.]*$",
    re.IGNORECASE,
)


def classify_intent(query: str) -> str:
    """
    Classify user query intent.
    Returns: 'greeting', 'simple', or 'complex'
    """
    q = query.strip()

    # Greeting / small talk
    if GREETING_PATTERNS.match(q):
        return "greeting"

    # Simple question heuristics: short, single clause, no technical depth
    word_count = len(q.split())
    has_complex_verb = bool(re.search(r'\b(compare|analyze|explain|describe|evaluate|discuss|summarize|review)\b', q, re.IGNORECASE))
    if word_count <= 6 and not has_complex_verb:
        return "simple"

    return "complex"


# ── System Prompts ───────────────────────────────────────────────────

GREETING_SYSTEM_PROMPT = """You are ScholarSync, a friendly AI research assistant.
The user is greeting you or making small talk. Respond warmly in 1-2 short sentences.
Do NOT reference papers, context, or research unless asked. Keep it natural and brief."""

NORMAL_SYSTEM_PROMPT = """You are ScholarSync, an expert AI research assistant for academic literature.
Be concise unless explicitly asked to explain in detail.

Response rules by query type:
- Simple factual questions → 2-4 sentences, direct answer
- Complex analytical questions → structured response with 2 paragraphs max
- Always reference source papers by name when citing claims
- Use markdown sparingly (bold key terms, bullet lists only when needed)
- NO paragraph dumping, NO unnecessary headers, NO walls of text
- Be conversational and helpful, not robotic

SOURCE GROUNDING RULES:
- Clearly distinguish paper-grounded claims from web-discovered information
- If web results are present, prefix web-sourced insights with [Web] tag
- NEVER present web findings as if they came from uploaded papers
- Paper evidence always takes priority over web sources
- If a web claim contradicts paper evidence, note the discrepancy"""


DEEP_DECOMPOSE_PROMPT = """You are a research planning agent. Given a research question, decompose it into exactly 4 focused sub-questions covering: definitions/background, methodologies, key findings, and limitations.

Output JSON: {"sub_questions": ["...", "...", "...", "..."]}"""


DEEP_SYNTHESIS_PROMPT = """You are ScholarSync in DEEP RESEARCH MODE — a SENIOR AI RESEARCH ANALYST producing publication-quality analysis.

You are NOT a summarization bot. You are an analytical, comparative, decisive researcher.

MANDATORY FORMAT:

# [Descriptive Research Title]

## Executive Summary
2+ paragraphs — the overall research landscape, key tension points, and decisive conclusions.

## Key Insights
5–8 bullet points with **bold headings** and 2–3 sentence explanation. Each must add unique information.

## Detailed Analysis

### Background & Definitions
Foundational concepts, key terms, evolution of the field — 2+ paragraphs.

### Methodology Comparison (DIMENSION-BY-DIMENSION)
For EACH major approach, compare:
- Core mechanism and architecture
- Strengths vs. weaknesses (tradeoff analysis)
- Computational cost and scalability
- Best use cases
RANK approaches — declare which is STRONGEST and WHY.
Include a comparison table if 3+ methods exist.

### Quantitative Results & Findings
Specific metrics, benchmark numbers, performance comparisons — 3+ paragraphs.
ONLY include numbers explicitly found in source material.
If exact numbers are unavailable, use qualitative language ("significant improvement", "marginal gain").
NEVER fabricate benchmark values.

### Cross-Paper Reasoning & Synthesis
How findings from different papers relate, contradict, or reinforce each other.
Identify: agreements, conflicts, complementary insights, emergent patterns.

### Tradeoff Analysis
For each approach: what do you gain vs. what do you lose?
Practical implications for different deployment scenarios.

### Limitations & Open Challenges
Specific limitations per approach, unresolved issues, failure modes — 2+ paragraphs.

## Future Directions
Open questions, promising research directions, gaps in current work — 1+ paragraph.

## Conclusion
DECISIVE synthesis — 2+ paragraphs. State clear recommendations:
- Best approach for [scenario A] is X because...
- Best approach for [scenario B] is Y because...
Do NOT hedge unnecessarily. Be analytical and conclusive.

## References
List all source papers referenced with [1], [2], etc.

SOURCE GROUNDING RULES:
- Clearly distinguish paper-grounded claims from web-discovered information
- If web results are present, prefix web-sourced insights with [Web]
- NEVER present web findings as if they came from uploaded papers
- NEVER invent benchmark numbers or specific metrics
- Paper evidence ALWAYS takes priority over web sources

CRITICAL: 
1. Every sentence must add NEW information — no repetition.
2. Be DECISIVE in conclusions — avoid vague hedging.
3. Use specific paper names and data when citing claims.
4. Compare approaches HEAD-TO-HEAD with clear winners declared.
5. No generic summarization — provide ANALYTICAL DEPTH."""


# ── Greeting Handler ─────────────────────────────────────────────────

def _handle_greeting(message: str) -> str:
    """Handle greetings with a short, friendly response — no context needed."""
    km = get_key_manager()
    return km.call_llm(
        messages=[
            {"role": "system", "content": GREETING_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        max_tokens=100,
    )


# ── Normal Mode Handler ─────────────────────────────────────────────

def _handle_normal(
    chat_id: str,
    message: str,
    history: list[dict],
    intent: str,
) -> str:
    """Normal mode: retrieve context + optional web search + single LLM call."""
    settings = get_settings()
    km = get_key_manager()

    history_text = _format_history(history, max_messages=4)

    # Get paper context via vector + graph retrieval
    paper_context = get_context(
        query=message,
        depth=settings.normal_mode_graph_depth,
        top_k=settings.normal_mode_top_k,
    )
    if len(paper_context) > 4000:
        paper_context = paper_context[:4000] + "\n\n[context truncated]"

    # Retrieval fusion: optionally augment with web search
    fusion = fuse_retrieval(
        query=message,
        paper_context=paper_context,
        enable_web_search=False,  # Auto-detect based on query
        paper_count=1,  # Assume papers uploaded in normal mode
    )
    context = fusion["fused_context"] or paper_context

    # Adjust max_tokens based on intent
    if intent == "simple":
        length_hint = "Answer concisely in 2-4 sentences."
        max_tok = 512
    else:
        length_hint = "Give a clear, structured answer. Be thorough but avoid unnecessary padding."
        max_tok = 2048

    user_prompt = f"""Conversation history:
{history_text}

Research context:
{context}

Question: {message}

{length_hint}"""

    return km.call_llm(
        messages=[
            {"role": "system", "content": NORMAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tok,
    )


# ── Streaming Handlers ───────────────────────────────────────────────

def _stream_greeting(message: str):
    """Stream a greeting response."""
    km = get_key_manager()
    return km.call_llm_stream(
        messages=[
            {"role": "system", "content": GREETING_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        max_tokens=100,
    )


def _stream_normal(
    chat_id: str,
    message: str,
    history: list[dict],
    intent: str,
):
    """Stream a normal mode response."""
    settings = get_settings()
    km = get_key_manager()

    history_text = _format_history(history, max_messages=4)
    context = get_context(
        query=message,
        depth=settings.normal_mode_graph_depth,
        top_k=settings.normal_mode_top_k,
    )
    if len(context) > 4000:
        context = context[:4000] + "\n\n[context truncated]"

    if intent == "simple":
        length_hint = "Answer concisely in 2-4 sentences."
        max_tok = 512
    else:
        length_hint = "Give a clear, structured answer. Be thorough but avoid unnecessary padding."
        max_tok = 2048

    user_prompt = f"""Conversation history:
{history_text}

Research context from papers:
{context}

Question: {message}

{length_hint}"""

    return km.call_llm_stream(
        messages=[
            {"role": "system", "content": NORMAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tok,
    )


import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Queues to pass live internal agent thoughts to the SSE stream
THOUGHT_QUEUES: dict[str, queue.Queue] = {}

def enqueue_thought(session_id: str, message: str):
    q = THOUGHT_QUEUES.get(session_id)
    if q is not None:
        q.put(message)

async def _stream_deep_research(
    chat_id: str,
    message: str,
    history: list[dict],
):
    """
    Deep research streaming endpoint.

    First tries the full LangGraph multi-agent pipeline (if papers are in
    the in-memory session store). Falls back to the LLM-based deep research
    approach that queries ChromaDB directly (works even after server restart
    since ChromaDB is persisted on disk).
    """
    from scholarsync.utils.schemas import PipelineStatus

    # Try to get paper metadata from the in-memory session store
    session = None
    try:
        from scholarsync.api.main import sessions
        session = sessions.get(chat_id)
    except Exception:
        pass

    has_papers_in_session = session and session.get("paper_metadata")

    if has_papers_in_session:
        # Full LangGraph multi-agent pipeline path
        from scholarsync.workflow.langgraph_pipeline import build_pipeline

        yield {"event": "agent_thought", "data": "🚀 Full multi-agent pipeline starting..."}

        q = queue.Queue()
        THOUGHT_QUEUES[chat_id] = q

        initial_state = {
            "session_id": chat_id,
            "query": message,
            "paper_metadata": [p.model_dump() for p in session["paper_metadata"]],
            "status": PipelineStatus.PENDING.value,
            "progress_messages": ["🚀 Multi-Agent Pipeline initialized..."],
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

        app = build_pipeline().compile()

        loop = asyncio.get_running_loop()

        def run_graph():
            try:
                return app.invoke(initial_state)
            except Exception as e:
                logger.error("Langgraph error: %s", e)
                q.put(f"ERROR: {e}")
                return None
            finally:
                q.put("DONE")

        executor = ThreadPoolExecutor(max_workers=1)
        future = loop.run_in_executor(executor, run_graph)

        # Continuously read from the queue and yield to SSE as agent thoughts
        while True:
            try:
                msg = q.get_nowait()
                if msg == "DONE":
                    break
                if msg.startswith("ERROR:"):
                    yield {"event": "error", "data": msg}
                    break
                yield {"event": "agent_thought", "data": msg}
            except queue.Empty:
                if future.done():
                    break
                await asyncio.sleep(0.1)

        THOUGHT_QUEUES.pop(chat_id, None)

        final_state = await future
        if not final_state:
            yield {"event": "error", "data": "Pipeline failed to execute."}
            return

        report_md = final_state.get("report_markdown", "")
        if report_md:
            yield {"event": "progress", "data": "✅ Report generated successfully."}
            chunk_size = 30
            for i in range(0, len(report_md), chunk_size):
                yield {"event": "token", "data": report_md[i:i+chunk_size]}
                await asyncio.sleep(0.01)
        else:
            errs = final_state.get("errors", [])
            yield {"event": "error", "data": "Pipeline failed: " + ", ".join(errs)}

        yield {"event": "done", "data": ""}

    else:
        # Fallback: LLM-based deep research using ChromaDB context directly.
        # This works even after server restart since ChromaDB is persisted.
        yield {"event": "agent_thought", "data": "🔬 Deep research mode — analyzing papers via vector search..."}

        loop = asyncio.get_running_loop()

        try:
            # Run the synchronous deep research in a thread
            events = await loop.run_in_executor(
                None,
                lambda: list(_handle_deep_research_sync(chat_id, message, history)),
            )

            yield {"event": "agent_thought", "data": "✅ Analysis complete — generating comprehensive report..."}

            for event in events:
                if event.get("event") == "token":
                    text = event["data"]
                    # Stream tokens in small chunks for smooth UI rendering
                    chunk_size = 30
                    for i in range(0, len(text), chunk_size):
                        yield {"event": "token", "data": text[i:i+chunk_size]}
                        await asyncio.sleep(0.01)
                else:
                    yield event

        except Exception as e:
            logger.error("Deep research fallback error: %s", e)
            yield {"event": "error", "data": f"Deep research failed: {str(e)}"}

        yield {"event": "done", "data": ""}




# ── Shared Helpers ───────────────────────────────────────────────────

def _format_history(history: list[dict], max_messages: int = 4) -> str:
    """Format recent conversation history (trimmed to save tokens)."""
    if not history:
        return "(No prior conversation)"
    recent = history[-max_messages:]
    parts = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        if len(content) > 300:
            content = content[:300] + "…"
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


# ── Public Entry Points ──────────────────────────────────────────────

async def route_message(
    chat_id: str,
    message: str,
    history: list[dict],
    deep_research: bool = False,
) -> str:
    """Non-streaming route (backward compat)."""
    intent = classify_intent(message)
    logger.info("ModeRouter: chat=%s mode=%s intent=%s", chat_id, "DEEP" if deep_research else "NORMAL", intent)

    loop = asyncio.get_event_loop()

    if intent == "greeting" and not deep_research:
        return await loop.run_in_executor(None, _handle_greeting, message)

    if deep_research:
        # For non-streaming deep, collect all tokens
        parts = []
        for event in _handle_deep_research_sync(chat_id, message, history):
            if event.get("event") == "token":
                parts.append(event["data"])
        return "".join(parts)

    return await loop.run_in_executor(
        None, _handle_normal, chat_id, message, history, intent,
    )


def _handle_deep_research_sync(chat_id, message, history):
    """Non-streaming deep research for backward compat."""
    settings = get_settings()
    km = get_key_manager()
    history_text = _format_history(history, max_messages=3)

    # Decompose
    result = km.call_llm(
        messages=[
            {"role": "system", "content": DEEP_DECOMPOSE_PROMPT},
            {"role": "user", "content": f"Research question: {message}\n\nDecompose into 4 sub-questions. JSON only."},
        ],
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    try:
        sub_questions = json.loads(result).get("sub_questions", [])[:4]
        if not sub_questions: raise ValueError()
    except Exception:
        sub_questions = [
            f"Key concepts of: {message}",
            f"Methodologies for: {message}",
            f"Main findings of: {message}",
            f"Limitations of: {message}",
        ]

    all_contexts = []
    for i, sq in enumerate(sub_questions, 1):
        ctx = get_context(query=sq, depth=settings.deep_research_graph_depth, top_k=settings.deep_research_top_k)
        if len(ctx) > 1500:
            ctx = ctx[:1500] + "..."
        all_contexts.append(f"Sub-question {i}: {sq}\n\n{ctx}")

    # Augment with web search if query triggers it
    fusion = fuse_retrieval(
        query=message,
        paper_context="",
        enable_web_search=False,  # Auto-detect
        paper_count=0,
    )
    if fusion["web_activated"] and fusion["web_context"]:
        web_ctx = fusion["web_context"]
        if len(web_ctx) > 2000:
            web_ctx = web_ctx[:2000] + "\n[web results truncated]"
        all_contexts.append(f"Web-augmented research:\n\n{web_ctx}")

    aggregated = "\n\n---\n\n".join(all_contexts)
    if len(aggregated) > 6000:
        aggregated = aggregated[:6000] + "\n\n[truncated]"

    synthesis_prompt = f"""Original research question: {message}

Context:
{history_text}

Research from {len(sub_questions)} sub-queries:

{aggregated}

Produce a comprehensive analysis following the MANDATORY FORMAT."""

    text = km.call_llm(
        messages=[
            {"role": "system", "content": DEEP_SYNTHESIS_PROMPT},
            {"role": "user", "content": synthesis_prompt},
        ],
        max_tokens=4096,
    )
    yield {"event": "token", "data": text}


async def route_message_stream(
    chat_id: str,
    message: str,
    history: list[dict],
    deep_research: bool = False,
) -> AsyncGenerator[dict, None]:
    """
    Streaming route — yields dicts with 'event' and 'data' keys.
    Events: 'progress', 'token', 'done'
    """
    intent = classify_intent(message)
    logger.info("ModeRouter[stream]: chat=%s mode=%s intent=%s", chat_id, "DEEP" if deep_research else "NORMAL", intent)

    loop = asyncio.get_event_loop()

    if intent == "greeting" and not deep_research:
        stream = await loop.run_in_executor(None, _stream_greeting, message)
        for chunk in stream:
            yield {"event": "token", "data": chunk}
        yield {"event": "done", "data": ""}
        return

    if deep_research:
        gen = _stream_deep_research(chat_id, message, history)
        async for event in gen:
            yield event
        return

    stream = await loop.run_in_executor(
        None, _stream_normal, chat_id, message, history, intent,
    )
    for chunk in stream:
        yield {"event": "token", "data": chunk}
    yield {"event": "done", "data": ""}
