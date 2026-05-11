import re
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage

from .base_agent import BaseAgent, AgentResult
from utils.llm_client import get_llm, get_tokens_used, llm_retry


class SingleAgent(BaseAgent):
    """
    Single-agent baseline: one LLM call to solve the problem, no planning or review loop.

    Used to benchmark the benefit of the multi-agent pipeline against the simplest
    possible approach (single-shot code generation).
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert Python developer.\n"
        "Solve the following Python programming problem.\n"
        "Return ONLY the Python code inside a ```python ... ``` fence.\n"
        "Include all necessary imports. Do NOT add any explanation."
    )

    def __init__(self, model: str, system_prompt: str | None = None):
        super().__init__(
            name="SingleAgent",
            model=model,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
        )

    @staticmethod
    def _extract_code(text: str) -> str:
        m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()

    @llm_retry
    def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        llm = get_llm(self.model)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Problem:\n\n{task}"),
        ]
        response = llm.invoke(messages)
        code = self._extract_code(response.content)
        return AgentResult(
            agent_name=self.name,
            output=code,
            success=True,
            tokens_used=get_tokens_used(response),
        )
