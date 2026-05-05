"""Unit tests for TesterAgent.execute_tests — no LLM calls needed."""
import pytest
from agents.tester_agent import TesterAgent


@pytest.fixture
def tester():
    return TesterAgent(model="gpt-4o")


GOOD_CODE = """
def add(a, b):
    return a + b
"""

BAD_CODE = """
def add(a, b):
    return a - b
"""

TESTS = """
from solution import add

def test_add_positive():
    assert add(2, 3) == 5

def test_add_zero():
    assert add(0, 0) == 0
"""


def test_all_pass(tester):
    result = tester.execute_tests(code=GOOD_CODE, tests=TESTS)
    assert result.passed == 2
    assert result.failed == 0
    assert result.all_passed is True
    assert result.pass_rate == 1.0


def test_some_fail(tester):
    result = tester.execute_tests(code=BAD_CODE, tests=TESTS)
    assert result.failed > 0
    assert result.all_passed is False
    assert result.pass_rate < 1.0


def test_syntax_error_code(tester):
    result = tester.execute_tests(code="def add(a b): return a+b", tests=TESTS)
    assert result.all_passed is False
