from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import planner_node, coder_node, tester_node, reviewer_node, should_continue, init_agents


def build_graph(model: str):
    """Compile and return the LangGraph StateGraph for the multi-agent pipeline."""
    init_agents(model=model)

    workflow = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    workflow.add_node("planner",  planner_node)
    workflow.add_node("coder",    coder_node)
    workflow.add_node("tester",   tester_node)
    workflow.add_node("reviewer", reviewer_node)

    # ── Edges ─────────────────────────────────────────────────────────────────
    workflow.set_entry_point("planner")
    workflow.add_edge("planner",  "coder")
    workflow.add_edge("coder",    "tester")
    workflow.add_edge("tester",   "reviewer")

    # feedback loop: reviewer → coder (retry) or END (done)
    workflow.add_conditional_edges(
        "reviewer",
        should_continue,
        {"continue": "coder", "end": END},
    )

    return workflow.compile()
