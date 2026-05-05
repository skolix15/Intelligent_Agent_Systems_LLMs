import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

from graph import build_graph, AgentState
from benchmarks.humaneval import HumanEvalBenchmark
from utils.metrics import MetricsCollector

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Agent Software Engineering Benchmark")
    parser.add_argument("--benchmark",      type=str,   default="data/humaneval.jsonl")
    parser.add_argument("--max-problems",   type=int,   default=10)
    parser.add_argument("--max-iterations", type=int,   default=5)
    parser.add_argument("--pass-threshold", type=float, default=1.0)
    parser.add_argument("--model",          type=str,   default="gpt-4o")
    parser.add_argument("--output",         type=str,   default="results/run.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    graph = build_graph(model=args.model)
    benchmark = HumanEvalBenchmark(path=args.benchmark, max_problems=args.max_problems)
    collector = MetricsCollector()

    logger.info(f"Running {len(benchmark)} problems · model={args.model!r}")

    for problem in benchmark:
        initial_state: AgentState = {
            "problem_id":          problem.problem_id,
            "problem_description": problem.description,
            "tests":               problem.tests,
            "plan":                "",
            "code":                "",
            "test_output":         "",
            "test_passed":         0,
            "test_failed":         0,
            "test_errors":         0,
            "review":              "",
            "approved":            False,
            "iteration":           0,
            "max_iterations":      args.max_iterations,
            "pass_threshold":      args.pass_threshold,
            "total_tokens":        0,
            "history":             [],
        }

        final_state: AgentState = graph.invoke(initial_state)
        collector.add_from_state(final_state)

        total = final_state["test_passed"] + final_state["test_failed"] + final_state["test_errors"]
        pass_rate = final_state["test_passed"] / total if total > 0 else 0.0
        logger.info(
            f"{problem.problem_id}: pass_rate={pass_rate:.0%} "
            f"iterations={final_state['iteration']} tokens={final_state['total_tokens']}"
        )

    metrics = collector.aggregate()
    logger.info(f"Aggregate: {metrics}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    collector.save(args.output)
    logger.success(f"Results saved to {args.output!r}")


if __name__ == "__main__":
    main()
