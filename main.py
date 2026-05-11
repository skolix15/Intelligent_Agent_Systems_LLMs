import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

from graph import build_graph, AgentState
from benchmarks.humaneval import HumanEvalBenchmark
from utils.metrics import MetricsCollector

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Agent Software Engineering Benchmark")
    parser.add_argument("--benchmark",      type=str,   default="data/HumanEval.jsonl")
    parser.add_argument("--max-problems",   type=int,   default=10)
    parser.add_argument("--max-iterations", type=int,   default=5)
    parser.add_argument("--pass-threshold", type=float, default=1.0)
    parser.add_argument("--model",          type=str,   default="gpt-4o")
    parser.add_argument("--output",         type=str,   default="results/run.json")
    parser.add_argument(
        "--mode",
        type=str,
        default="multi",
        choices=["multi", "single"],
        help="multi: full 4-agent pipeline; single: one-shot baseline (no planning/review loop)",
    )
    return parser.parse_args()


def run_multi_agent(args: argparse.Namespace) -> list[AgentState]:
    graph = build_graph(model=args.model)
    benchmark = HumanEvalBenchmark(path=args.benchmark, max_problems=args.max_problems)
    collector = MetricsCollector()

    logger.info(f"[multi-agent] {len(benchmark)} problems · model={args.model!r}")

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

    return collector


def run_single_agent(args: argparse.Namespace) -> MetricsCollector:
    from agents import SingleAgent, TesterAgent

    single = SingleAgent(model=args.model)
    tester = TesterAgent(model=args.model)
    benchmark = HumanEvalBenchmark(path=args.benchmark, max_problems=args.max_problems)
    collector = MetricsCollector()

    logger.info(f"[single-agent] {len(benchmark)} problems · model={args.model!r}")

    for problem in benchmark:
        result = single.run(task=problem.description)
        test_result = tester.execute_tests(code=result.output, tests=problem.tests)

        total = test_result.passed + test_result.failed + test_result.errors
        pass_rate = test_result.passed / total if total > 0 else 0.0
        approved = test_result.all_passed

        state: AgentState = {
            "problem_id":          problem.problem_id,
            "problem_description": problem.description,
            "tests":               problem.tests,
            "plan":                "",
            "code":                result.output,
            "test_output":         test_result.output,
            "test_passed":         test_result.passed,
            "test_failed":         test_result.failed,
            "test_errors":         test_result.errors,
            "review":              "",
            "approved":            approved,
            "iteration":           1,
            "max_iterations":      1,
            "pass_threshold":      args.pass_threshold,
            "total_tokens":        result.tokens_used,
            "history":             [],
        }
        collector.add_from_state(state)

        logger.info(
            f"{problem.problem_id}: pass_rate={pass_rate:.0%} tokens={result.tokens_used}"
        )

    return collector


def _save_results(collector: MetricsCollector, output_path: str) -> None:
    metrics = collector.aggregate()
    logger.info(f"Aggregate: {metrics}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    collector.save(output_path)
    logger.success(f"Results saved to {output_path!r}")


def _generate_report(
    collector: MetricsCollector,
    mode: str,
    model: str,
    output_path: str,
) -> None:
    from agents import ReporterAgent

    metrics = collector.aggregate()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = str(Path(output_path).parent / f"report_{mode}_{timestamp}.html")

    logger.info("ReporterAgent: generating HTML report …")
    reporter = ReporterAgent(model=model)
    reporter.generate_report(
        states=collector._states,
        metrics=metrics,
        mode=mode,
        model=model,
        output_path=report_path,
    )
    logger.success(f"HTML report saved to {report_path!r}")


def main() -> None:
    args = parse_args()

    try:
        if args.mode == "single":
            collector = run_single_agent(args)
        else:
            collector = run_multi_agent(args)
    except Exception as exc:
        _handle_fatal(exc, args.model)
        return

    _save_results(collector, args.output)
    _generate_report(collector, mode=args.mode, model=args.model, output_path=args.output)


def _handle_fatal(exc: Exception, model: str) -> None:
    msg = str(exc)
    # Permission / model-not-found (OpenAI 403)
    if "PermissionDeniedError" in type(exc).__name__ or "403" in msg or "model_not_found" in msg:
        logger.error(
            f"Model '{model}' is not accessible on your OpenAI project.\n"
            "  Fix: go to https://platform.openai.com → your project → Settings → Model access\n"
            "  and enable the model, or change MODEL= in your .env to one that is allowed\n"
            "  (e.g. gpt-3.5-turbo)."
        )
        return
    # Authentication error (401)
    if "AuthenticationError" in type(exc).__name__ or "401" in msg or "invalid_api_key" in msg:
        logger.error(
            "API key is invalid or missing.\n"
            "  Fix: check OPENAI_API_KEY in your .env file."
        )
        return
    # Unknown — show the raw message without the full traceback
    logger.error(f"Fatal error: {type(exc).__name__}: {msg}")


if __name__ == "__main__":
    main()
