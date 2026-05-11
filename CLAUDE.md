# CLAUDE.md

## Project Overview

This is a research/academic project evaluating multi-agent LLM collaboration for automated code generation. It benchmarks a 4-agent LangGraph pipeline (Planner → Coder → Tester → Reviewer) against a single-agent baseline on the HumanEval dataset.

## Common Commands

```bash
# Install dependencies
pipenv install

# Quick smoke test (3 problems)
pipenv run python run.py smoke

# Single-agent baseline (20 problems)
pipenv run python run.py single

# Full multi-agent pipeline (20 problems)
pipenv run python run.py multi

# Custom run
pipenv run python main.py --max-problems 5 --mode multi --model gpt-4o-mini

# Lint
pipenv run ruff check .

# Format
pipenv run black .

# Tests
pipenv run pytest tests/
```

## Architecture

### State Machine (LangGraph)

The multi-agent graph is built in [graph/pipeline.py](graph/pipeline.py). Nodes are defined in [graph/nodes.py](graph/nodes.py). The shared state is `AgentState` (TypedDict) in [graph/state.py](graph/state.py).

**Control flow:**
- `planner_node` → `coder_node` → `tester_node` → `reviewer_node` → `should_continue()`
- `should_continue()` returns `"coder"` (retry) or `"end"` based on: approval status, pass rate vs threshold, and iteration count.

### Agents

All agents live in [agents/](agents/). They wrap LangChain chat models and store per-run token counts.

| Agent | File | LLM? | Role |
|---|---|---|---|
| PlannerAgent | [agents/planner_agent.py](agents/planner_agent.py) | Yes | Step-by-step plan, no code |
| CoderAgent | [agents/coder_agent.py](agents/coder_agent.py) | Yes | Generate/fix Python code |
| TesterAgent | (in graph/nodes.py) | No | Run pytest in subprocess |
| ReviewerAgent | [agents/reviewer_agent.py](agents/reviewer_agent.py) | Yes | Approve or list bugs |
| ReporterAgent | [agents/reporter_agent.py](agents/reporter_agent.py) | Yes | HTML narrative report |
| SingleAgent | [agents/single_agent.py](agents/single_agent.py) | Yes | Baseline one-shot generation |

### LLM Client

[utils/llm_client.py](utils/llm_client.py) provides:
- `get_llm(model)` — returns `ChatAnthropic` or `ChatOpenAI` based on model name prefix
- `get_tokens_used(response)` — reads `usage_metadata` from LangChain responses
- `@llm_retry` — tenacity decorator: 3 attempts, 4–60s exponential backoff, skips 401/403

### Benchmark Loader

[benchmarks/humaneval.py](benchmarks/humaneval.py) loads `data/HumanEval.jsonl` and adapts HumanEval's `check(candidate)` test format into standalone pytest `test_check()` functions.

### Metrics

[utils/metrics.py](utils/metrics.py) accumulates `AgentState` results into `AggregateMetrics`: solve rate, pass@1, avg iterations, total tokens. Serialized to JSON via `MetricsCollector.save()`.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `MODEL` | No (default: `gpt-4o-mini`) | LLM identifier |
| `OPENAI_API_KEY` | For OpenAI models | OpenAI API key |
| `ANTHROPIC_API_KEY` | For Anthropic models | Anthropic API key |

Put these in a `.env` file at the project root — `python-dotenv` loads them automatically.

## Key Design Decisions

- **TesterAgent has no LLM** — test execution is deterministic subprocess pytest, not LLM-judged.
- **CoderAgent uses regex extraction** — pulls code from ```python...``` fenced blocks in LLM output.
- **ReviewerAgent keyword protocol** — the string `APPROVED` (case-sensitive) in the response triggers graph termination; anything else is treated as a bug list fed back to the Coder.
- **Isolated test execution** — each test run uses a fresh `tempfile.mkdtemp()` to prevent cross-problem contamination.
- **Token tracking** — every LLM agent accumulates tokens into `state["total_tokens"]` for cost analysis across modes.

## Output Files

- `results/{smoke,single,multi}.json` — full run state for all problems
- `results/report_{mode}_{timestamp}.html` — self-contained HTML report

## Adding a New Agent

1. Create `agents/my_agent.py` inheriting from `BaseAgent` (see [agents/planner_agent.py](agents/planner_agent.py) for reference).
2. Add a node function in [graph/nodes.py](graph/nodes.py) that calls the agent and updates `AgentState`.
3. Wire the node into the graph in [graph/pipeline.py](graph/pipeline.py).
4. Export from [agents/__init__.py](agents/__init__.py).
