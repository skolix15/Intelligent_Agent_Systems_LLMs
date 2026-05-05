"""Tests for HumanEval benchmark loader."""
import json
import pytest
from pathlib import Path
from benchmarks.humaneval import HumanEvalBenchmark, Problem


@pytest.fixture
def sample_jsonl(tmp_path):
    data = [
        {
            "task_id": "HumanEval/0",
            "prompt": "def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n",
            "canonical_solution": "    return a + b\n",
            "test": "def check(c):\n    assert c(1,2)==3\ncheck(add)",
            "entry_point": "add",
        },
        {
            "task_id": "HumanEval/1",
            "prompt": "def double(x):\n    \"\"\"Return 2*x.\"\"\"\n",
            "canonical_solution": "    return 2 * x\n",
            "test": "def check(c):\n    assert c(3)==6\ncheck(double)",
            "entry_point": "double",
        },
    ]
    path = tmp_path / "test.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in data))
    return path


def test_loads_all_problems(sample_jsonl):
    bench = HumanEvalBenchmark(path=sample_jsonl)
    assert len(bench) == 2


def test_max_problems_limit(sample_jsonl):
    bench = HumanEvalBenchmark(path=sample_jsonl, max_problems=1)
    assert len(bench) == 1


def test_problem_fields(sample_jsonl):
    bench = HumanEvalBenchmark(path=sample_jsonl)
    p: Problem = bench.problems[0]
    assert p.problem_id == "HumanEval/0"
    assert "add" in p.description
    assert "check" in p.tests
