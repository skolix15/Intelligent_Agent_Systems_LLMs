import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Problem:
    problem_id: str
    description: str
    canonical_solution: str
    tests: str  # pytest-compatible test string


class HumanEvalBenchmark:
    """
    Loads a subset of HumanEval problems from a local JSONL file.

    Expected JSONL format (one JSON object per line):
      {"task_id": "HumanEval/0", "prompt": "...", "canonical_solution": "...", "test": "..."}
    """

    def __init__(self, path: str | Path, max_problems: int | None = None):
        self.path = Path(path)
        self.problems: list[Problem] = []
        self._load(max_problems)

    def _load(self, max_problems: int | None) -> None:
        with open(self.path) as f:
            for i, line in enumerate(f):
                if max_problems and i >= max_problems:
                    break
                raw = json.loads(line)
                self.problems.append(
                    Problem(
                        problem_id=raw["task_id"],
                        description=raw["prompt"],
                        canonical_solution=raw["canonical_solution"],
                        tests=raw["test"],
                    )
                )

    def __len__(self) -> int:
        return len(self.problems)

    def __iter__(self):
        return iter(self.problems)
