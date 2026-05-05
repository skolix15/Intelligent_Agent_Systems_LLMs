from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from loguru import logger


@dataclass
class AgentMessage:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    agent_name: str
    output: str
    success: bool
    tokens_used: int = 0
    iterations: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    def __init__(self, name: str, model: str, system_prompt: str):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.conversation_history: list[AgentMessage] = []
        self.total_tokens_used: int = 0

    def reset(self) -> None:
        self.conversation_history = []

    def _add_message(self, role: str, content: str, metadata: dict | None = None) -> None:
        self.conversation_history.append(
            AgentMessage(role=role, content=content, metadata=metadata or {})
        )

    @abstractmethod
    def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Execute the agent's task and return a structured result."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.model!r})"
