from typing import Any
from .base_agent import BaseAgent, AgentResult


class CoderAgent(BaseAgent):
    """Generates or fixes Python code based on a plan and optional test feedback."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert Python developer. "
        "Given a plan (and optionally failing test output), write correct, clean Python code. "
        "Return ONLY the code block, no explanations."
    )

    def __init__(self, model: str, system_prompt: str | None = None):
        super().__init__(
            name="CoderAgent",
            model=model,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
        )

    def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        # context may include: {"plan": str, "previous_code": str, "test_feedback": str}
        # TODO: call LLM API and return generated code
        raise NotImplementedError
