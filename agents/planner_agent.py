from typing import Any
from .base_agent import BaseAgent, AgentResult


class PlannerAgent(BaseAgent):
    """Decomposes a high-level task into a step-by-step implementation plan."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a software architect. Given a programming problem, produce a clear, "
        "step-by-step implementation plan that a developer can follow. "
        "Focus on data structures, algorithm choice, and edge cases."
    )

    def __init__(self, model: str, system_prompt: str | None = None):
        super().__init__(
            name="PlannerAgent",
            model=model,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
        )

    def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        # TODO: call LLM API and return structured plan
        raise NotImplementedError
