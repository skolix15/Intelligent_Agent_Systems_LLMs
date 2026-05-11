# Intelligent Agent Systems with LLMs

A multi-agent AI system for automated code generation, benchmarked against a single-agent baseline on the [HumanEval](https://github.com/openai/human-eval) programming benchmark. The system orchestrates four specialized agents (Planner, Coder, Tester, Reviewer) via a LangGraph state machine to iteratively solve Python programming problems.

## Overview

The core research question is whether a collaborative multi-agent pipeline outperforms a single-agent baseline in solving code generation tasks, measured by solve rate, pass@1, iterations needed, and token efficiency.

**Pipeline:**

```
Planner → Coder → Tester → Reviewer → (loop back to Coder or END)
```

**Baseline:**

```
SingleAgent → Tester → END
```

## Project Structure

```
.
├── agents/
│   ├── planner_agent.py     # Generates step-by-step implementation plan
│   ├── coder_agent.py       # Produces/fixes Python code iteratively
│   ├── reviewer_agent.py    # Approves code or lists actionable bugs
│   ├── reporter_agent.py    # Writes HTML narrative reports
│   └── single_agent.py      # One-shot baseline agent
├── graph/
│   ├── pipeline.py          # Builds the LangGraph state machine
│   ├── nodes.py             # Node functions + conditional edge logic
│   └── state.py             # AgentState TypedDict definition
├── benchmarks/
│   └── humaneval.py         # HumanEval loader + pytest adapter
├── utils/
│   ├── llm_client.py        # LLM factory, token counting, retry decorator
│   └── metrics.py           # MetricsCollector + AggregateMetrics
├── data/
│   └── HumanEval.jsonl      # HumanEval benchmark dataset
├── results/                 # JSON results + HTML reports (generated)
├── main.py                  # CLI entry point
├── run.py                   # Preset run configurations
└── Pipfile
```

## Setup

**Requirements:** Python 3.13, [Pipenv](https://pipenv.pypa.io/)

```bash
pipenv install
```

Create a `.env` file in the project root:

```env
MODEL=gpt-4o-mini          # or gpt-4o, claude-3-5-sonnet-20241022, etc.
OPENAI_API_KEY=sk-...      # required for OpenAI models
ANTHROPIC_API_KEY=sk-...   # required for Anthropic/Claude models
```

## Running

### Preset modes (recommended)

```bash
pipenv run python run.py smoke    # 3 problems — quick sanity check
pipenv run python run.py single   # 20 problems, single-agent baseline
pipenv run python run.py multi    # 20 problems, full multi-agent pipeline
```

### Advanced CLI

```bash
pipenv run python main.py \
  --benchmark data/HumanEval.jsonl \
  --max-problems 10 \
  --max-iterations 5 \
  --model gpt-4o \
  --mode multi \
  --output results/run.json
```

| Flag | Description | Default |
|---|---|---|
| `--benchmark` | Path to HumanEval JSONL file | `data/HumanEval.jsonl` |
| `--max-problems` | Number of problems to solve | 20 |
| `--max-iterations` | Max retry iterations per problem | 3 |
| `--pass-threshold` | Pass rate to stop early (0–1) | 1.0 |
| `--model` | LLM identifier | `gpt-4o-mini` |
| `--mode` | `multi` or `single` | `multi` |
| `--output` | JSON results file path | `results/run.json` |

## How It Works

### Multi-Agent Pipeline

1. **PlannerAgent** — given the problem description, produces a numbered step-by-step implementation plan (no code).
2. **CoderAgent** — generates Python code informed by the plan. On subsequent iterations, also receives test failure output and reviewer feedback.
3. **TesterAgent** — writes code to a temp file, runs pytest, and returns pass/fail/error counts. No LLM calls — purely local execution.
4. **ReviewerAgent** — reads the code and test results. Outputs `APPROVED` or a numbered list of bugs.
5. **Decision edge** — if the code is approved or `pass_rate >= pass_threshold` or `iteration >= max_iterations`, the graph terminates; otherwise the Coder retries.
6. **ReporterAgent** — after the graph finishes, generates a narrative summary and an HTML report.

### Single-Agent Baseline

A single LLM call generates code directly from the problem description, then the Tester executes it. No planning, no review loop.

## Output

Results are written to `results/`:

- **`{mode}.json`** — full state snapshots for all problems, including per-iteration history, token counts, and test results.
- **`report_{mode}_{timestamp}.html`** — self-contained HTML report with a metrics dashboard and per-problem cards showing the plan, iteration history, final code, and test output.

### Metrics reported

| Metric | Description |
|---|---|
| Solve rate | % of problems where all tests pass |
| pass@1 | % solved on the first attempt |
| Avg iterations | Mean retries per problem |
| Total / avg tokens | Token usage for cost analysis |

## Supported Models

Any model supported by LangChain's `ChatAnthropic` or `ChatOpenAI` integrations:

- OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, etc.
- Anthropic: `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`, etc.

Set the `MODEL` environment variable or pass `--model` via CLI.

## Development

```bash
# Lint
pipenv run ruff check .

# Format
pipenv run black .

# Tests
pipenv run pytest tests/
```
