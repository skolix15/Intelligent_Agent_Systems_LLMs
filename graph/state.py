from typing import TypedDict


class AgentState(TypedDict):
    # problem input
    problem_id: str
    problem_description: str
    tests: str

    # agent outputs (accumulated across iterations)
    plan: str
    code: str
    test_output: str
    test_passed: int
    test_failed: int
    test_errors: int
    review: str
    approved: bool

    # loop control
    iteration: int
    max_iterations: int
    pass_threshold: float  # 0.0 – 1.0

    # tracking
    total_tokens: int
    history: list[dict]
