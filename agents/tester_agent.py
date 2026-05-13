import subprocess
import tempfile
import os
from typing import Any
from dataclasses import dataclass
from .base_agent import BaseAgent, AgentResult


@dataclass
class TestResult:
    passed: int
    failed: int
    errors: int
    output: str

    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.errors == 0

    @property
    def pass_rate(self) -> float:
        total = self.passed + self.failed + self.errors
        return self.passed / total if total > 0 else 0.0


class TesterAgent(BaseAgent):
    """
    Executes pytest tests against generated Python code.

    Writes the generated code and its test suite to a temporary directory,
    runs pytest in a subprocess, and parses pass/fail/error counts from output.
    No LLM call is made — execution is purely local.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a QA engineer. "
        "Given a programming problem and its test cases, run the tests and report which pass or fail."
    )

    def __init__(self, model: str, system_prompt: str | None = None):
        super().__init__(
            name="TesterAgent",
            model=model,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
        )

    def execute_tests(self, code: str, tests: str) -> TestResult:
        """
        Write code + tests to a temp dir, run pytest in a subprocess, return results.

        Args:
            code:  Python source code produced by CoderAgent.
            tests: pytest-compatible test string (uses the generated code via import or exec).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            code_path = os.path.join(tmp_dir, "solution.py")
            test_path = os.path.join(tmp_dir, "test_solution.py")

            with open(code_path, "w") as f:
                f.write(code)
            with open(test_path, "w") as f:
                f.write(tests)

            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", test_path, "-v", "--tb=short"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=tmp_dir,
                )
                output = result.stdout + result.stderr
            except subprocess.TimeoutExpired:
                return TestResult(passed=0, failed=1, errors=0, output="TIMEOUT: test execution exceeded 30s (likely infinite loop)")

            passed = output.count(" PASSED")
            failed = output.count(" FAILED")
            errors = output.count(" ERROR")
            return TestResult(passed=passed, failed=failed, errors=errors, output=output)

    def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        # context must include: {"code": str, "tests": str}
        ctx = context or {}
        test_result = self.execute_tests(code=ctx["code"], tests=ctx["tests"])
        return AgentResult(
            agent_name=self.name,
            output=test_result.output,
            success=test_result.all_passed,
            metadata={
                "passed":    test_result.passed,
                "failed":    test_result.failed,
                "errors":    test_result.errors,
                "pass_rate": test_result.pass_rate,
            },
        )
