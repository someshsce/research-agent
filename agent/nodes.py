"""LangGraph nodes: router, planner, executor, reflector, synthesizer, refusal.

Each post-router node is type-aware: it picks the right prompt and schema based
on `state['query_type']`. The executor and tool layer are shared across all
query types.
"""
from __future__ import annotations

import json
import os
from typing import Any

from anthropic import Anthropic

from .prompts import (
    COMPANY_PLANNER_PROMPT,
    COMPANY_REFLECTOR_PROMPT,
    COMPANY_SYNTHESIZER_PROMPT,
    COMPARISON_PLANNER_PROMPT,
    COMPARISON_REFLECTOR_PROMPT,
    COMPARISON_SYNTHESIZER_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    TOPIC_PLANNER_PROMPT,
    TOPIC_REFLECTOR_PROMPT,
    TOPIC_SYNTHESIZER_PROMPT,
)
from .schemas import (
    AgentState,
    CompanyBrief,
    ComparisonReport,
    QueryType,
    RefusalReport,
    ResearchTask,
    RoutingDecision,
    ToolName,
    TopicReport,
)
from .tools import ResearchTools

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")


# Prompt and schema dispatch tables — keyed by query type
PLANNER_PROMPTS = {
    QueryType.COMPANY: COMPANY_PLANNER_PROMPT,
    QueryType.TOPIC: TOPIC_PLANNER_PROMPT,
    QueryType.COMPARISON: COMPARISON_PLANNER_PROMPT,
}

REFLECTOR_PROMPTS = {
    QueryType.COMPANY: COMPANY_REFLECTOR_PROMPT,
    QueryType.TOPIC: TOPIC_REFLECTOR_PROMPT,
    QueryType.COMPARISON: COMPARISON_REFLECTOR_PROMPT,
}

SYNTHESIZER_PROMPTS = {
    QueryType.COMPANY: COMPANY_SYNTHESIZER_PROMPT,
    QueryType.TOPIC: TOPIC_SYNTHESIZER_PROMPT,
    QueryType.COMPARISON: COMPARISON_SYNTHESIZER_PROMPT,
}

SCHEMAS = {
    QueryType.COMPANY: CompanyBrief,
    QueryType.TOPIC: TopicReport,
    QueryType.COMPARISON: ComparisonReport,
}


_client: Anthropic | None = None
_tools: ResearchTools | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


def _get_tools() -> ResearchTools:
    global _tools
    if _tools is None:
        _tools = ResearchTools()
    return _tools


def _log(state: AgentState, msg: str) -> None:
    state.setdefault("logs", []).append(msg)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from a model response.

    Tolerant of:
      - markdown code fences (```json ... ```)
      - leading/trailing prose
      - common LLM JSON drift (trailing commas, unclosed brackets, etc.) — handled by json_repair
      - truncated output (cut off mid-string) — json_repair will close brackets
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    # First try strict json — fastest happy path.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to json_repair, which fixes the common LLM mistakes
    # (trailing commas, unescaped quotes, truncated objects/arrays).
    from json_repair import repair_json
    repaired = repair_json(text, return_objects=True)
    if isinstance(repaired, dict):
        return repaired
    raise json.JSONDecodeError(
        f"json_repair returned non-dict: {type(repaired).__name__}", text, 0
    )


def _summarize_results(completed_tasks: list[dict], full: bool = False) -> str:
    char_limit = 1500 if full else 500
    parts: list[str] = []
    for ct in completed_tasks:
        task = ct["task"]
        result = ct["result"]
        parts.append(f"\n--- [{task['tool']}] {task['query']} ---")
        if "error" in result and result.get("error"):
            parts.append(f"ERROR: {result['error']}")
            continue
        for r in result.get("results", []):
            title = r.get("title", "")
            url = r.get("url", "")
            body = r.get("summary") or r.get("content") or ""
            parts.append(f"• {title}")
            parts.append(f"  {body[:char_limit]}")
            if url:
                parts.append(f"  Source: {url}")
    return "\n".join(parts) if parts else "(no results yet)"


def _parse_tasks(raw: Any, state: AgentState) -> list[ResearchTask]:
    """Parse raw task entries from a model response into ResearchTask objects.

    Defensive: skips entries that are the wrong shape (e.g. plain strings)
    or fail Pydantic validation, logging each skip. Returns whatever it
    could parse — possibly an empty list.
    """
    if not isinstance(raw, list):
        _log(state, f"   ⚠️  Expected list of tasks, got {type(raw).__name__}; ignoring")
        return []

    tasks: list[ResearchTask] = []
    for t in raw:
        if not isinstance(t, dict):
            _log(state, f"   ⚠️  Skipping non-object task: {t!r}")
            continue
        try:
            tasks.append(ResearchTask(**t))
        except Exception as e:  # noqa: BLE001
            _log(state, f"   ⚠️  Skipping malformed task ({e}): {t!r}")
    return tasks


def _target_label(routing: RoutingDecision, query: str) -> str:
    if routing.query_type == QueryType.COMPANY:
        return f"Company: {routing.company_name or query}"
    if routing.query_type == QueryType.TOPIC:
        return f"Topic: {routing.topic or query}"
    if routing.query_type == QueryType.COMPARISON:
        return f"Comparison: {routing.item_a} vs {routing.item_b}"
    return query


# ───────────────────────────── Router ─────────────────────────────


def router_node(state: AgentState) -> dict[str, Any]:
    query = state["query"]
    _log(state, f"🧭 Routing query: {query}")

    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=512,
        system=ROUTER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": query}],
    )
    data = _extract_json(response.content[0].text)
    routing = RoutingDecision(**data)

    _log(state, f"   → Classified as: {routing.query_type.value}")
    _log(state, f"   → Reasoning: {routing.reasoning}")

    return {
        "query_type": routing.query_type,
        "routing": routing,
        "logs": state["logs"],
    }


# ───────────────────────────── Refusal ─────────────────────────────


def refusal_node(state: AgentState) -> dict[str, Any]:
    routing = state["routing"]
    _log(state, "⛔ Query is outside the agent's research scope — refusing politely")

    refusal = RefusalReport(
        reason=routing.reasoning or "This query isn't a researchable topic.",
        suggestion=(
            "I'm a research agent specialised in three things: "
            "companies (e.g., 'Razorpay'), topics (e.g., 'carbon capture'), "
            "and comparisons (e.g., 'React vs Vue'). Try rephrasing your query "
            "into one of these shapes."
        ),
    )
    return {"final_report": refusal, "logs": state["logs"]}


# ───────────────────────────── Planner (type-aware) ─────────────────────────────


def planner_node(state: AgentState) -> dict[str, Any]:
    qtype: QueryType = state["query_type"]
    routing: RoutingDecision = state["routing"]
    prompt = PLANNER_PROMPTS[qtype]

    if qtype == QueryType.COMPANY:
        user_msg = f"Plan research for company: {routing.company_name or state['query']}"
    elif qtype == QueryType.TOPIC:
        user_msg = f"Plan research for topic: {routing.topic or state['query']}"
    else:  # COMPARISON
        user_msg = f"Plan research to compare: {routing.item_a} vs {routing.item_b}"

    _log(state, f"🧠 Planning research ({qtype.value})")

    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=1024,
        system=prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    plan_data = _extract_json(response.content[0].text)
    tasks = _parse_tasks(plan_data.get("tasks", []), state)
    if not tasks:
        raise ValueError("Planner produced no valid tasks — cannot proceed")

    _log(state, f"📋 Plan: {len(tasks)} research tasks")
    for t in tasks:
        _log(state, f"   • [{t.tool.value}] {t.query}")

    return {
        "plan": tasks,
        "completed_tasks": [],
        "iteration": 0,
        "max_iterations": 3,
        "logs": state["logs"],
    }


# ───────────────────────────── Executor (shared) ─────────────────────────────


def executor_node(state: AgentState) -> dict[str, Any]:
    plan = state.get("plan", [])
    completed = list(state.get("completed_tasks", []))
    tools = _get_tools()

    for task in plan:
        _log(state, f"🔧 Executing [{task.tool.value}]: {task.query}")
        if task.tool == ToolName.WEB_SEARCH:
            result = tools.web_search(task.query)
        elif task.tool == ToolName.WIKIPEDIA:
            result = tools.wikipedia_search(task.query)
        elif task.tool == ToolName.NEWS_SEARCH:
            result = tools.news_search(task.query)
        else:
            continue

        completed.append({"task": task.model_dump(mode="json"), "result": result})
        n = len(result.get("results", []))
        if "error" in result and result.get("error"):
            _log(state, f"   ⚠️  Error: {result['error']}")
        else:
            _log(state, f"   ✓ Got {n} result(s)")

    return {
        "completed_tasks": completed,
        "iteration": state.get("iteration", 0) + 1,
        "logs": state["logs"],
    }


# ───────────────────────────── Reflector (type-aware) ─────────────────────────────


def reflector_node(state: AgentState) -> dict[str, Any]:
    qtype: QueryType = state["query_type"]
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    if iteration >= max_iter:
        _log(state, f"⏹  Reached max iterations ({max_iter}); synthesizing")
        return {"needs_more_research": False, "plan": [], "logs": state["logs"]}

    _log(state, "🤔 Reflecting on findings...")
    summary = _summarize_results(state.get("completed_tasks", []))
    prompt = REFLECTOR_PROMPTS[qtype]

    user_msg = (
        f"Original query: {state['query']}\n"
        f"Target: {_target_label(state['routing'], state['query'])}\n"
        f"Iteration: {iteration}/{max_iter}\n\n"
        f"Findings so far:\n{summary}"
    )

    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=1024,
        system=prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    reflection = _extract_json(response.content[0].text)

    needs_more = bool(reflection.get("needs_more_research", False))
    _log(state, f"💭 {reflection.get('reasoning', '')}")

    if needs_more:
        follow_ups = _parse_tasks(reflection.get("follow_up_tasks", []), state)
        if not follow_ups:
            # Model said it needed more research but didn't produce valid tasks —
            # we have data, just synthesize what we've got rather than loop forever.
            _log(state, "✅ No valid follow-up tasks returned; proceeding to synthesis")
            return {"needs_more_research": False, "plan": [], "logs": state["logs"]}
        _log(state, f"🔁 Queuing {len(follow_ups)} follow-up task(s)")
        return {"plan": follow_ups, "needs_more_research": True, "logs": state["logs"]}

    _log(state, "✅ Sufficient research — proceeding to synthesis")
    return {"needs_more_research": False, "plan": [], "logs": state["logs"]}


# ───────────────────────────── Synthesizer (type-aware) ─────────────────────────────


def synthesizer_node(state: AgentState) -> dict[str, Any]:
    qtype: QueryType = state["query_type"]
    schema = SCHEMAS[qtype]
    prompt = SYNTHESIZER_PROMPTS[qtype]
    routing: RoutingDecision = state["routing"]

    _log(state, f"📝 Synthesizing final report ({qtype.value})...")

    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    findings = _summarize_results(state.get("completed_tasks", []), full=True)
    target = _target_label(routing, state["query"])

    base_msg = (
        f"{target}\n\n"
        f"Research findings:\n{findings}\n\n"
        f"Match this schema exactly:\n{schema_json}\n\n"
        f"Set research_iterations to {state.get('iteration', 0)}. "
        f"Return ONLY valid JSON, no markdown."
    )

    report = _synthesize_with_retry(prompt, base_msg, schema, state)
    _log(state, f"✨ Report complete (confidence: {report.confidence_score:.0%})")
    return {"final_report": report, "logs": state["logs"]}


def _synthesize_with_retry(
    prompt: str,
    base_msg: str,
    schema: type,
    state: AgentState,
    max_attempts: int = 2,
):
    """Call the synthesizer with up to N attempts, asking for shorter output on retry."""
    last_error: str | None = None
    user_msg = base_msg

    for attempt in range(max_attempts):
        if attempt > 0:
            _log(state, f"   🔁 Retry {attempt}: asking for a more concise output")
            user_msg = (
                base_msg
                + f"\n\nPREVIOUS ATTEMPT FAILED: {last_error}\n"
                "Be MORE CONCISE this time: keep every explanation to 2-3 sentences, "
                "include at most 5 items per list field, and double-check that the JSON "
                "is syntactically valid and complete."
            )

        response = _get_client().messages.create(
            model=MODEL,
            max_tokens=8192,
            system=prompt,
            messages=[{"role": "user", "content": user_msg}],
        )

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "max_tokens":
            _log(state, "   ⚠️  Response hit max_tokens; output may be truncated")

        raw_text = response.content[0].text
        try:
            report_data = _extract_json(raw_text)
            return schema(**report_data)
        except Exception as e:  # noqa: BLE001
            last_error = f"{type(e).__name__}: {str(e)[:200]}"
            _log(state, f"   ⚠️  Synthesis attempt {attempt + 1} failed: {last_error}")
            if attempt == max_attempts - 1:
                raise

    raise RuntimeError("synthesizer exhausted all retries")
