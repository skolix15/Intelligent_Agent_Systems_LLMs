"""Tests for graph state transitions and conditional edge logic."""
import pytest
from graph.nodes import should_continue


def make_state(**overrides):
    base = {
        "problem_id": "test/0",
        "problem_description": "",
        "tests": "",
        "plan": "",
        "code": "",
        "test_output": "",
        "test_passed": 0,
        "test_failed": 0,
        "test_errors": 0,
        "review": "",
        "approved": False,
        "iteration": 1,
        "max_iterations": 5,
        "pass_threshold": 1.0,
        "total_tokens": 0,
        "history": [],
    }
    return {**base, **overrides}


def test_should_end_when_all_pass():
    state = make_state(test_passed=3, test_failed=0, test_errors=0, pass_threshold=1.0)
    assert should_continue(state) == "end"


def test_should_continue_when_tests_fail():
    state = make_state(test_passed=2, test_failed=1, test_errors=0, iteration=1, max_iterations=5)
    assert should_continue(state) == "continue"


def test_should_end_at_max_iterations():
    state = make_state(test_passed=0, test_failed=3, test_errors=0, iteration=5, max_iterations=5)
    assert should_continue(state) == "end"


def test_partial_threshold():
    state = make_state(
        test_passed=8, test_failed=2, test_errors=0,
        pass_threshold=0.8, iteration=1, max_iterations=5
    )
    assert should_continue(state) == "end"
