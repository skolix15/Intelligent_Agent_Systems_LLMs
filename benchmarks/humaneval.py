import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Problem:
    problem_id: str
    description: str
    canonical_solution: str
    entry_point: str       # name of the function to implement
    tests: str             # pytest-compatible test string (adapted from HumanEval format)


def _adapt_tests_to_pytest(raw_test: str, entry_point: str) -> str:
    """
    Convert HumanEval's check(candidate) test format into a pytest-runnable file.

    HumanEval tests look like:
        def check(candidate):
            assert candidate(...) == expected

    We wrap them so pytest can discover and run a single test_check() function
    that calls check(entry_point_function_imported_from_solution).
    """
    return (
        f"from solution import {entry_point}\n\n"
        f"{raw_test.strip()}\n\n"
        f"def test_check():\n"
        f"    check({entry_point})\n"
    )


class HumanEvalBenchmark:
    """
    Loads a subset of HumanEval problems from a local JSONL file.

    Expected JSONL format (one JSON object per line):
      {"task_id": "HumanEval/0", "prompt": "...", "entry_point": "...",
       "canonical_solution": "...", "test": "..."}
    """

    def __init__(self, path: str | Path, max_problems: int | None = None):
        self.path = Path(path)
        self.problems: list[Problem] = []
        self._load(max_problems)

    def _load(self, max_problems: int | None) -> None:
        with open(self.path) as f:
            all_lines = f.readlines()
        if max_problems:
            all_lines = random.sample(all_lines, min(max_problems, len(all_lines)))
        for line in all_lines:
            raw = json.loads(line)
            entry_point = raw["entry_point"]
            self.problems.append(
                Problem(
                    problem_id=raw["task_id"],
                    description=raw["prompt"],
                    canonical_solution=raw["canonical_solution"],
                    entry_point=entry_point,
                    tests=_adapt_tests_to_pytest(raw["test"], entry_point),
                )
            )

    def __len__(self) -> int:
        return len(self.problems)

    def __iter__(self):
        return iter(self.problems)
