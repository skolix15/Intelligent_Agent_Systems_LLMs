import json
from dataclasses import dataclass
from pathlib import Path
from graph.state import AgentState


@dataclass
class AggregateMetrics:
    total_problems: int
    solved: int
    pass_at_1: float
    overall_pass_rate: float
    avg_iterations: float
    total_tokens: int
    avg_tokens_per_problem: float


class MetricsCollector:
    def __init__(self):
        self._states: list[AgentState] = []

    def add_from_state(self, state: AgentState) -> None:
        self._states.append(state)

    def _pass_rate(self, s: AgentState) -> float:
        total = s["test_passed"] + s["test_failed"] + s["test_errors"]
        return s["test_passed"] / total if total > 0 else 0.0

    def aggregate(self) -> AggregateMetrics:
        n = len(self._states)
        if n == 0:
            return AggregateMetrics(0, 0, 0.0, 0.0, 0.0, 0, 0.0)

        solved = sum(1 for s in self._states if s["approved"] or self._pass_rate(s) == 1.0)
        pass_at_1 = sum(
            1 for s in self._states if s["iteration"] == 1 and self._pass_rate(s) == 1.0
        ) / n
        overall_pass_rate = sum(self._pass_rate(s) for s in self._states) / n
        avg_iterations = sum(s["iteration"] for s in self._states) / n
        total_tokens = sum(s["total_tokens"] for s in self._states)

        return AggregateMetrics(
            total_problems=n,
            solved=solved,
            pass_at_1=pass_at_1,
            overall_pass_rate=overall_pass_rate,
            avg_iterations=avg_iterations,
            total_tokens=total_tokens,
            avg_tokens_per_problem=total_tokens / n,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self._states, indent=2, default=str))
