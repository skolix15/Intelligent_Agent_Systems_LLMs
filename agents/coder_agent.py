import re
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage

from .base_agent import BaseAgent, AgentResult
from utils.llm_client import get_llm, get_tokens_used, llm_retry


class CoderAgent(BaseAgent):
    """Generates or fixes Python code based on a plan and optional test feedback."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert Python developer.\n"
        "Given a programming problem and an implementation plan, write the complete Python solution.\n"
        "Rules:\n"
        "  - Return ONLY the Python code inside a ```python ... ``` fence.\n"
        "  - Include all necessary imports inside the function or at the top.\n"
        "  - Do NOT add explanations outside the code block.\n"
        "  - Implement the exact function signature from the problem prompt."
    )

    def __init__(self, model: str, system_prompt: str | None = None):
        super().__init__(
            name="CoderAgent",
            model=model,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
        )

    def _build_prompt(self, task: str, context: dict[str, Any] | None) -> str:
        parts = [f"## Problem\n{task}"]
        ctx = context or {}
        if ctx.get("plan"):
            parts.append(f"## Implementation Plan\n{ctx['plan']}")
        if ctx.get("previous_code"):
            parts.append(
                f"## Previous Code (contained failures — fix it)\n"
                f"```python\n{ctx['previous_code']}\n```"
            )
        if ctx.get("test_feedback"):
            parts.append(f"## Failing Test Output\n```\n{ctx['test_feedback']}\n```")
        if ctx.get("review"):
            parts.append(f"## Code Review Feedback\n{ctx['review']}")
        return "\n\n".join(parts)

    @staticmethod
    def _extract_code(text: str) -> str:
        """Strip markdown fences; fall back to raw text if none found."""
        m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()

    @llm_retry
    def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        llm = get_llm(self.model)
        prompt = self._build_prompt(task, context)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        code = self._extract_code(response.content)
        return AgentResult(
            agent_name=self.name,
            output=code,
            success=True,
            tokens_used=get_tokens_used(response),
        )
