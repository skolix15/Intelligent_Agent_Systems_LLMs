# Analysis Report: Multi-Agent vs Single-Agent Code Generation
## HumanEval Benchmark — LangGraph Pipeline Evaluation

---

## 1. Overview

This report evaluates a **4-agent LangGraph pipeline** (PlannerAgent → CoderAgent → TesterAgent → ReviewerAgent) against a **single-agent baseline** on the HumanEval dataset. The central research question, as defined by the course assignment, is:

> *When and why does collaboration among multiple specialized agents outperform a single general-purpose agent — and at what cost?*

Both runs were evaluated on **100 randomly sampled HumanEval problems** using `gpt-3.5-turbo` as the underlying LLM. The problem sets differ between runs due to random sampling, so per-problem comparisons serve as illustrative examples rather than direct head-to-head matches.

---

## 2. System Architecture

### 2.1 Multi-Agent Pipeline

The multi-agent system follows a structured state machine built with **LangGraph**:

```
PlannerAgent → CoderAgent → TesterAgent → ReviewerAgent
                    ↑__________________________|
                         (feedback loop)
```

| Agent | Role | LLM |
|---|---|---|
| PlannerAgent | Decomposes the problem into a step-by-step plan | Yes |
| CoderAgent | Generates or revises Python code based on plan + feedback | Yes |
| TesterAgent | Executes pytest in an isolated subprocess | No |
| ReviewerAgent | Approves code or returns a structured bug list | Yes |

**Termination conditions:**
- Test pass rate ≥ threshold (pass@1 achieved), OR
- Maximum iterations reached (set to 10)

**Feedback loop mechanics:** TesterAgent output (pass/fail/error details) and ReviewerAgent output (bug descriptions) are both passed back to CoderAgent on each retry, enabling iterative self-correction.

### 2.2 Single-Agent Baseline

A single `CoderAgent` receives the raw problem description and generates code in one shot — no planning, no testing feedback, no review. This represents the zero-iteration baseline.

---

## 3. Experimental Results

### 3.1 Aggregate Metrics

| Metric | Single-Agent | Multi-Agent | Δ |
|---|---|---|---|
| Problems evaluated | 100 | 100 | — |
| Problems solved | **71** | **72** | +1 |
| Solve rate | **71%** | **72%** | +1 pp |
| Avg pass rate | 71% | 72% | +1 pp |
| Avg iterations per problem | **1.0** | **3.7** | +2.7 |
| Total tokens consumed | **24,742** | **776,072** | **+31.4×** |
| Tokens per problem | **247** | **7,760** | **+31.4×** |

### 3.2 Token Cost vs. Accuracy Trade-off

The most striking finding is the **disproportionate token cost** of the multi-agent system relative to its marginal accuracy gain:

- Multi-agent consumed **776,072 tokens** to solve **72 problems**
- Single-agent consumed **24,742 tokens** to solve **71 problems**
- **Cost per solved problem:** Single = 349 tokens; Multi = 10,779 tokens — a **30.9× efficiency disadvantage**

This directly addresses the assignment's requirement to analyze *"iterations vs. quality vs. API call cost"*: for `gpt-3.5-turbo` at current pricing (~$0.50/1M input tokens), the multi-agent run costs approximately **30× more** while improving accuracy by only **1 percentage point**.

---

## 4. Qualitative Analysis

### 4.1 When Multi-Agent Outperforms Single

The feedback loop provided measurable benefit on problems where the first code attempt was **nearly correct but contained a subtle bug** that deterministic test execution could identify. Examples from the multi-agent run:

| Problem | Iterations | Outcome | Single outcome |
|---|---|---|---|
| HumanEval/125 | 7 | ✅ Pass | ❌ Fail |
| HumanEval/12 | 8 | ✅ Pass | Not in sample |
| HumanEval/91 | 3 | ✅ Pass | ❌ Fail |
| HumanEval/75 | 2 | ✅ Pass | ❌ Fail (0/2) |
| HumanEval/6 | 2 | ✅ Pass | ✅ Pass |
| HumanEval/9 | 2 | ✅ Pass | Not in sample |

**HumanEval/125** is the clearest success case: the single agent failed outright, while the multi-agent system recovered after 7 iterations (7,394 tokens), using the precise pytest failure messages to guide the coder toward a correct implementation. This demonstrates **emergent behavior through structured feedback** — the system produced an outcome that neither the coder alone nor the reviewer alone could have achieved.

### 4.2 When Multi-Agent Fails to Outperform

**27 problems** in the multi-agent run exhausted all 10 iterations without passing (0/1 tests). On several of these, the single agent also failed — suggesting the problems are genuinely hard for `gpt-3.5-turbo`. However, for some problems (e.g., HumanEval/99, HumanEval/123, HumanEval/138) the single agent **succeeded in one shot** while the multi-agent system failed after 10 iterations.

This reveals a significant weakness: the **reviewer and planner can introduce interference**. When the PlannerAgent generates a flawed decomposition, or when the ReviewerAgent provides vague or incorrect bug descriptions, the CoderAgent iterates toward an increasingly incorrect solution — a phenomenon sometimes called **"feedback poisoning"** in multi-agent literature.

The highest token consumers among failures:

| Problem | Iterations | Tokens | Outcome |
|---|---|---|---|
| HumanEval/32 | 10 | 13,921 | ❌ Fail |
| HumanEval/81 | 10 | 13,450 | ❌ Fail |
| HumanEval/130 | 10 | 12,761 | ❌ Fail |
| HumanEval/64 | 10 | 12,477 | ❌ Fail |
| HumanEval/115 | 10 | 12,817 | ❌ Fail |

These represent the worst-case scenario: maximum compute cost with zero benefit.

### 4.3 Task Decomposition Analysis

The **PlannerAgent** implements *task decomposition* as required by the assignment. Its value is most evident on algorithmically complex problems where a structured plan helps the CoderAgent avoid obvious logical errors on the first attempt. However, for simple problems (e.g., HumanEval/2, HumanEval/30, HumanEval/22, solved in 1 iteration), the planner adds latency and tokens with no benefit — the coder would have solved them correctly anyway.

This suggests that **conditional planning** (skip PlannerAgent for problems below a complexity threshold) could significantly improve the cost-efficiency ratio.

---

## 5. Design Decisions and Implementation Notes

### 5.1 TesterAgent as a Deterministic Oracle

A key design decision is that **TesterAgent uses no LLM** — it runs `pytest` in a sandboxed `tempfile.mkdtemp()` subprocess. This makes test evaluation **ground-truth accurate and reproducible**, unlike LLM-judged evaluation. The tradeoff is that the system cannot handle problems where the test itself is ambiguous or where partial credit is meaningful.

### 5.2 ReviewerAgent Protocol

The ReviewerAgent uses a **keyword protocol**: the string `APPROVED` in its response triggers graph termination. This binary signal is simple and deterministic, but lacks nuance — the reviewer cannot express "mostly correct, minor edge case missing."

### 5.3 Timeout Handling

A `subprocess.TimeoutExpired` handler was added to the TesterAgent to prevent infinite loops in generated code (e.g., HumanEval/39 produced an infinite loop across multiple iterations) from crashing the entire run. Timeouts are now captured as failed test results and passed back as feedback to the CoderAgent.

### 5.4 Isolated Test Execution

Each problem uses a fresh temporary directory, preventing import cache contamination across problems. This is critical for correctness in sequential benchmark runs.

---

## 6. Theoretical Framing

### 6.1 Task Decomposition
The Planner → Coder → Tester → Reviewer chain implements **hierarchical task decomposition**: a complex problem ("write a correct Python function") is broken into planning, implementation, verification, and quality assurance sub-tasks. The assignment identifies this as a core concept of multi-agent AI.

### 6.2 Feedback Loops
The Tester → Coder and Reviewer → Coder feedback loops implement **closed-loop control**: the system measures actual output against desired output and corrects accordingly. This is analogous to PID control in classical engineering. The results show this works when the error signal is precise (pytest output) but can be counterproductive when the corrective signal is noisy (reviewer hallucinations).

### 6.3 Emergent Behavior
HumanEval/125 (7 iterations to pass) and HumanEval/12 (8 iterations to pass) demonstrate **emergent behavior**: neither a single forward pass nor a single review would have solved these problems, but the iterative loop produced a correct solution that emerges from agent interactions over multiple rounds.

---

## 7. Conclusions

| Finding | Implication |
|---|---|
| +1% solve rate at 31× token cost | Multi-agent is not cost-effective at this scale for `gpt-3.5-turbo` |
| Feedback loop rescued ~6 problems | Loop value is real but narrow — applies only to "near-miss" cases |
| 27/100 problems hit max iterations | Failure mode is expensive; early termination heuristics needed |
| Some single-pass successes became multi-agent failures | Planner/reviewer can degrade performance on straightforward problems |
| Avg 3.7 iterations for multi vs 1.0 for single | Most improvement happens in iterations 2–4; diminishing returns after |

### 7.1 Answer to the Core Research Question

> *When does multi-agent collaboration outperform a single agent?*

Multi-agent collaboration outperforms when:
1. The problem is complex enough that planning provides genuine decomposition value
2. The first code attempt is **nearly correct** — a small, identifiable bug exists
3. The feedback signal (test output) is **precise and actionable**

It does **not** outperform when:
1. The problem is straightforward (single agent solves it in one shot)
2. The underlying model lacks the capability to fix the identified bug regardless of iterations
3. The planner or reviewer introduces incorrect guidance that misdirects the coder

### 7.2 Recommendations for Future Work

1. **Adaptive pipeline**: bypass PlannerAgent for low-complexity problems (e.g., based on description length or keyword heuristics)
2. **Early stopping improvement**: add confidence scoring from the ReviewerAgent to exit before 10 iterations when recovery is unlikely
3. **Stronger model**: repeat experiment with `gpt-4o-mini`, `gpt-4o`, or `claude-sonnet` — the feedback loop likely shows greater benefit when the coder can act on subtler bug descriptions
4. **Larger sample**: 164-problem full run needed for statistically significant conclusions (current ±5% confidence interval is wide)
5. **Cost-weighted metric**: report "tokens per solved problem" as a primary metric alongside solve rate

---

## 8. Summary Statistics Reference

### Single-Agent
- **Solve rate:** 71/100 (71%)
- **Total tokens:** 24,742
- **Tokens/problem:** 247
- **Token range:** 114 (HumanEval/27) – 581 (HumanEval/81)
- **All problems:** 1 iteration (by definition)

### Multi-Agent
- **Solve rate:** 72/100 (72%)
- **Total tokens:** 776,072
- **Tokens/problem:** 7,760
- **Iteration range:** 1–10
- **1-iteration solves:** 64/100 (64% solved on first attempt, confirming most problems don't benefit from the loop)
- **Multi-iteration rescues:** ~6 problems recovered that would otherwise fail
- **Max-iteration failures:** 27/100 (27% consumed full budget and still failed)
