"""LangGraph wiring.

Topology:

   START → router ──► (unsupported) ──► refusal ──► END
                  └─► planner ──► executor ──► reflector ──►(loop)──► executor
                                                          └──► synthesizer ──► END
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import (
    executor_node,
    planner_node,
    reflector_node,
    refusal_node,
    router_node,
    synthesizer_node,
)
from .schemas import AgentState, QueryType


def _route_after_router(state: AgentState) -> str:
    """If the query is unsupported, short-circuit to refusal."""
    if state.get("query_type") == QueryType.UNSUPPORTED:
        return "refusal"
    return "planner"


def _route_after_reflector(state: AgentState) -> str:
    """Loop back for more research, or move on to synthesis."""
    if state.get("needs_more_research", False) and state.get("plan"):
        return "execute"
    return "synthesize"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("refusal", refusal_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reflector", reflector_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {"planner": "planner", "refusal": "refusal"},
    )
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reflector")
    graph.add_conditional_edges(
        "reflector",
        _route_after_reflector,
        {"execute": "executor", "synthesize": "synthesizer"},
    )
    graph.add_edge("synthesizer", END)
    graph.add_edge("refusal", END)

    return graph.compile()


def run_agent(query: str) -> AgentState:
    """Convenience helper for the CLI / tests."""
    app = build_graph()
    return app.invoke({"query": query, "logs": []})
