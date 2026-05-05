from dataclasses import dataclass, field
from typing import Any
from loguru import logger

from agents import PlannerAgent, CoderAgent, TesterAgent, ReviewerAgent
from agents.tester_agent import TestResult


@dataclass
class RunConfig:
    max_iterations: int = 5
    pass_threshold: float = 1.0  # fraction of tests that must pass to stop early
    model: str = "gpt-4o"


@dataclass
class OrchestrationResult:
    problem_id: str
    final_code: str
    final_test_result: TestResult | None
    iterations_used: int
    total_tokens: int
    success: bool
    history: list[dict[str, Any]] = field(default_factory=list)


class Orchestrator:
    """
    Drives the Planner → Coder → Tester → Reviewer feedback loop.

    Loop continues until:
      - pass_rate >= pass_threshold, OR
      - max_iterations is reached.
    """

    def __init__(self, config: RunConfig):
        self.config = config
        self.planner = PlannerAgent(model=config.model)
        self.coder = CoderAgent(model=config.model)
        self.tester = TesterAgent(model=config.model)
        self.reviewer = ReviewerAgent(model=config.model)

    def solve(self, problem_id: str, problem_description: str, tests: str) -> OrchestrationResult:
        logger.info(f"Starting orchestration for problem {problem_id!r}")

        plan_result = self.planner.run(task=problem_description)
        logger.debug(f"Plan: {plan_result.output[:200]}")

        code = ""
        test_result: TestResult | None = None
        total_tokens = plan_result.tokens_used
        history: list[dict[str, Any]] = []

        for iteration in range(1, self.config.max_iterations + 1):
            logger.info(f"Iteration {iteration}/{self.config.max_iterations}")

            coder_result = self.coder.run(
                task=problem_description,
                context={
                    "plan": plan_result.output,
                    "previous_code": code,
                    "test_feedback": test_result.output if test_result else "",
                },
            )
            code = coder_result.output
            total_tokens += coder_result.tokens_used

            test_result = self.tester.execute_tests(code=code, tests=tests)
            logger.info(
                f"Tests — passed: {test_result.passed}, failed: {test_result.failed}, "
                f"errors: {test_result.errors}"
            )

            review_result = self.reviewer.run(
                task=problem_description,
                context={"code": code, "test_result": test_result, "plan": plan_result.output},
            )
            total_tokens += review_result.tokens_used

            history.append(
                {
                    "iteration": iteration,
                    "code": code,
                    "pass_rate": test_result.pass_rate,
                    "review": review_result.output,
                    "tokens": coder_result.tokens_used + review_result.tokens_used,
                }
            )

            if test_result.pass_rate >= self.config.pass_threshold:
                logger.success(f"Pass threshold reached at iteration {iteration}.")
                break

        return OrchestrationResult(
            problem_id=problem_id,
            final_code=code,
            final_test_result=test_result,
            iterations_used=len(history),
            total_tokens=total_tokens,
            success=test_result.all_passed if test_result else False,
            history=history,
        )
