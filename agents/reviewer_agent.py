from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage

from .base_agent import BaseAgent, AgentResult
from utils.llm_client import get_llm, get_tokens_used, llm_retry


class ReviewerAgent(BaseAgent):
    """Reviews generated code and either approves it or lists specific issues to fix."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a senior Python code reviewer.\n"
        "Given a programming problem, its solution, and test results, respond with ONE of:\n"
        "  1. The single word APPROVED — if the code is correct and all tests pass.\n"
        "  2. A short, actionable list of specific bugs or edge cases to fix.\n"
        "Do NOT rewrite the code yourself. Be concise."
    )

    def __init__(self, model: str, system_prompt: str | None = None):
        super().__init__(
            name="ReviewerAgent",
            model=model,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
        )

    def _build_prompt(self, task: str, context: dict[str, Any]) -> str:
        code = context.get("code", "")
        test_result = context.get("test_result")
        plan = context.get("plan", "")

        parts = [f"## Problem\n{task}"]
        if plan:
            parts.append(f"## Plan\n{plan}")
        parts.append(f"## Solution\n```python\n{code}\n```")

        if test_result is not None:
            total = test_result.passed + test_result.failed + test_result.errors
            pass_pct = f"{test_result.passed}/{total} tests passed"
            parts.append(
                f"## Test Results ({pass_pct})\n```\n{test_result.output[:2000]}\n```"
            )

        return "\n\n".join(parts)

    @llm_retry
    def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        llm = get_llm(self.model)
        prompt = self._build_prompt(task, context or {})
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        return AgentResult(
            agent_name=self.name,
            output=response.content,
            success=True,
            tokens_used=get_tokens_used(response),
        )
