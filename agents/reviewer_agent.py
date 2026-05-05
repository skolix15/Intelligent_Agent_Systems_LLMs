from typing import Any
from .base_agent import BaseAgent, AgentResult


class ReviewerAgent(BaseAgent):
    """Reviews generated code for correctness, style, and edge cases."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a senior code reviewer. "
        "Given a programming problem, its solution, and test results, provide concise feedback: "
        "correctness issues, edge cases missed, and suggestions for improvement. "
        "If the code is acceptable, say 'APPROVED'."
    )

    def __init__(self, model: str, system_prompt: str | None = None):
        super().__init__(
            name="ReviewerAgent",
            model=model,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
        )

    def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        # context may include: {"code": str, "test_result": TestResult, "plan": str}
        # TODO: call LLM API and return review feedback
        raise NotImplementedError
