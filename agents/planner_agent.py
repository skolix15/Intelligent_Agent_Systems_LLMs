from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage

from .base_agent import BaseAgent, AgentResult
from utils.llm_client import get_llm, get_tokens_used, llm_retry


class PlannerAgent(BaseAgent):
    """Decomposes a programming problem into a step-by-step implementation plan."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a software architect. Given a Python programming problem (a function signature "
        "and docstring), produce a concise, numbered step-by-step implementation plan.\n"
        "Focus on: algorithm choice, data structures, edge cases to handle, and key logic.\n"
        "Do NOT write code — only the plan."
    )

    def __init__(self, model: str, system_prompt: str | None = None):
        super().__init__(
            name="PlannerAgent",
            model=model,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
        )

    @llm_retry
    def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        llm = get_llm(self.model)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Problem to plan:\n\n{task}"),
        ]
        response = llm.invoke(messages)
        return AgentResult(
            agent_name=self.name,
            output=response.content,
            success=True,
            tokens_used=get_tokens_used(response),
        )
