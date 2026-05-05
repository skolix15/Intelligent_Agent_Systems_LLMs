from loguru import logger
from agents import PlannerAgent, CoderAgent, TesterAgent, ReviewerAgent
from .state import AgentState

# agents are module-level singletons; model is injected at graph compile time
_planner: PlannerAgent | None = None
_coder: CoderAgent | None = None
_tester: TesterAgent | None = None
_reviewer: ReviewerAgent | None = None


def init_agents(model: str) -> None:
    global _planner, _coder, _tester, _reviewer
    _planner = PlannerAgent(model=model)
    _coder = CoderAgent(model=model)
    _tester = TesterAgent(model=model)
    _reviewer = ReviewerAgent(model=model)


# ── Node functions ────────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> AgentState:
    logger.info(f"[{state['problem_id']}] PlannerAgent running")
    result = _planner.run(task=state["problem_description"])
    return {
        **state,
        "plan": result.output,
        "total_tokens": state["total_tokens"] + result.tokens_used,
        "iteration": 0,
    }


def coder_node(state: AgentState) -> AgentState:
    iteration = state["iteration"] + 1
    logger.info(f"[{state['problem_id']}] CoderAgent — iteration {iteration}")
    result = _coder.run(
        task=state["problem_description"],
        context={
            "plan": state["plan"],
            "previous_code": state.get("code", ""),
            "test_feedback": state.get("test_output", ""),
            "review": state.get("review", ""),
        },
    )
    return {
        **state,
        "code": result.output,
        "iteration": iteration,
        "total_tokens": state["total_tokens"] + result.tokens_used,
    }


def tester_node(state: AgentState) -> AgentState:
    logger.info(f"[{state['problem_id']}] TesterAgent running")
    test_result = _tester.execute_tests(code=state["code"], tests=state["tests"])
    total = test_result.passed + test_result.failed + test_result.errors
    logger.info(
        f"[{state['problem_id']}] pass={test_result.passed}/{total} "
        f"fail={test_result.failed} err={test_result.errors}"
    )
    return {
        **state,
        "test_output": test_result.output,
        "test_passed": test_result.passed,
        "test_failed": test_result.failed,
        "test_errors": test_result.errors,
    }


def reviewer_node(state: AgentState) -> AgentState:
    logger.info(f"[{state['problem_id']}] ReviewerAgent running")
    from agents.tester_agent import TestResult

    test_result = TestResult(
        passed=state["test_passed"],
        failed=state["test_failed"],
        errors=state["test_errors"],
        output=state["test_output"],
    )
    result = _reviewer.run(
        task=state["problem_description"],
        context={"code": state["code"], "test_result": test_result, "plan": state["plan"]},
    )
    approved = "APPROVED" in result.output.upper()

    history_entry = {
        "iteration": state["iteration"],
        "code": state["code"],
        "pass_rate": test_result.pass_rate,
        "review": result.output,
        "tokens": result.tokens_used,
    }

    return {
        **state,
        "review": result.output,
        "approved": approved,
        "total_tokens": state["total_tokens"] + result.tokens_used,
        "history": state.get("history", []) + [history_entry],
    }


# ── Conditional edge ──────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    total = state["test_passed"] + state["test_failed"] + state["test_errors"]
    pass_rate = state["test_passed"] / total if total > 0 else 0.0

    if pass_rate >= state["pass_threshold"]:
        logger.success(f"[{state['problem_id']}] Pass threshold reached — stopping.")
        return "end"

    if state["iteration"] >= state["max_iterations"]:
        logger.warning(f"[{state['problem_id']}] Max iterations reached — stopping.")
        return "end"

    return "continue"
