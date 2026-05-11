import html
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from .base_agent import BaseAgent, AgentResult
from utils.llm_client import get_llm, get_tokens_used, llm_retry


class ReporterAgent(BaseAgent):
    """
    Generates a self-contained HTML report for a complete benchmark run.

    For every problem it makes one LLM call to write a 2-3 sentence narrative
    (what happened, what broke, how it was fixed, final outcome).
    The rest of the HTML (code, test output, iteration timeline, metrics) is
    assembled directly from the collected AgentState objects — no extra LLM cost.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a technical writer analyzing an AI agent code-generation session.\n"
        "Given the execution trace of a multi-agent system solving a Python programming problem, "
        "write exactly 2-3 sentences that explain:\n"
        "  1. What the problem asked for.\n"
        "  2. What went wrong (if anything) and how it was fixed across iterations.\n"
        "  3. The final outcome (solved / not solved, how many iterations).\n"
        "Be concise and factual. Do not repeat the code."
    )

    def __init__(self, model: str, system_prompt: str | None = None):
        super().__init__(
            name="ReporterAgent",
            model=model,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
        )

    # ── LLM narrative per problem ─────────────────────────────────────────────

    @llm_retry
    def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        ctx = context or {}
        history = ctx.get("history", [])
        approved = ctx.get("approved", False)
        iterations = ctx.get("iterations", 1)
        pass_rate = ctx.get("pass_rate", 0.0)

        history_text = ""
        for entry in history:
            history_text += (
                f"\n  Iteration {entry['iteration']}: "
                f"pass_rate={entry['pass_rate']:.0%}, "
                f"review={entry['review'][:200]}"
            )

        prompt = (
            f"Problem:\n{task[:600]}\n\n"
            f"Iterations used: {iterations}\n"
            f"Final pass rate: {pass_rate:.0%}\n"
            f"Approved: {approved}\n"
            f"Execution trace:{history_text or ' (single-shot, no loop)'}"
        )

        llm = get_llm(self.model)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        return AgentResult(
            agent_name=self.name,
            output=response.content,
            success=True,
            tokens_used=get_tokens_used(response),
        )

    # ── Full HTML report ──────────────────────────────────────────────────────

    def generate_report(
        self,
        states: list[dict],
        metrics,
        mode: str,
        model: str,
        output_path: str,
    ) -> None:
        """
        Build and save a complete HTML report for the run.
        Calls self.run() once per problem to get the LLM narrative.
        """
        problem_sections = []
        for state in states:
            total = state["test_passed"] + state["test_failed"] + state["test_errors"]
            pass_rate = state["test_passed"] / total if total > 0 else 0.0

            narrative_result = self.run(
                task=state["problem_description"],
                context={
                    "history":    state.get("history", []),
                    "approved":   state.get("approved", False),
                    "iterations": state.get("iteration", 1),
                    "pass_rate":  pass_rate,
                },
            )
            self.total_tokens_used += narrative_result.tokens_used
            problem_sections.append(
                _render_problem(state, pass_rate, narrative_result.output)
            )

        html_content = _render_full_page(
            problem_sections=problem_sections,
            metrics=metrics,
            mode=mode,
            model=model,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html_content, encoding="utf-8")


# ── HTML rendering helpers ────────────────────────────────────────────────────

def _e(text: str) -> str:
    """HTML-escape a string."""
    return html.escape(str(text))


def _render_problem(state: dict, pass_rate: float, narrative: str) -> str:
    problem_id = state["problem_id"]
    solved = state.get("approved") or pass_rate == 1.0
    status_cls = "status-pass" if solved else "status-fail"
    status_label = "SOLVED" if solved else "FAILED"

    plan_html = ""
    if state.get("plan"):
        plan_html = f"""
        <div class="section-block">
          <div class="block-title">📋 Plan</div>
          <pre class="pre-block">{_e(state['plan'])}</pre>
        </div>"""

    iter_html = _render_iterations(state.get("history", []), state)

    total = state["test_passed"] + state["test_failed"] + state["test_errors"]
    return f"""
    <div class="problem-card">
      <div class="problem-header">
        <div>
          <span class="problem-id">{_e(problem_id)}</span>
          <span class="status-badge {status_cls}">{status_label}</span>
        </div>
        <div class="problem-meta">
          {state['iteration']} iteration(s) &nbsp;·&nbsp;
          {state['test_passed']}/{total} tests pass &nbsp;·&nbsp;
          {state['total_tokens']:,} tokens
        </div>
      </div>

      <div class="section-block">
        <div class="block-title">📝 Problem</div>
        <pre class="pre-block">{_e(state['problem_description'][:500])}{'…' if len(state['problem_description']) > 500 else ''}</pre>
      </div>

      {plan_html}

      {iter_html}

      <div class="section-block">
        <div class="block-title">💻 Final Code</div>
        <pre class="pre-block code-block">{_e(state.get('code', '(no code generated)'))}</pre>
      </div>

      <div class="section-block">
        <div class="block-title">🧪 Final Test Output</div>
        <pre class="pre-block test-block">{_e(state.get('test_output', '(no output)')[:1500])}</pre>
      </div>

      <div class="narrative-block">
        <div class="block-title">🤖 Analysis</div>
        <p>{_e(narrative)}</p>
      </div>
    </div>"""


def _render_iterations(history: list[dict], state: dict) -> str:
    if not history:
        return ""

    items = ""
    for entry in history:
        pr = entry.get("pass_rate", 0.0)
        pr_cls = "iter-pass" if pr == 1.0 else ("iter-partial" if pr > 0 else "iter-fail")
        review_text = entry.get("review", "")
        approved_marker = " ✓ APPROVED" if "APPROVED" in review_text.upper() else ""
        items += f"""
        <div class="iter-entry">
          <div class="iter-header">
            <span class="iter-num">Iteration {entry['iteration']}</span>
            <span class="iter-rate {pr_cls}">{pr:.0%} pass rate{approved_marker}</span>
            <span class="iter-tokens">{entry.get('tokens', 0):,} tokens</span>
          </div>
          <details>
            <summary>Code</summary>
            <pre class="pre-block code-block">{_e(entry.get('code', ''))}</pre>
          </details>
          <details>
            <summary>Review</summary>
            <pre class="pre-block">{_e(review_text)}</pre>
          </details>
        </div>"""

    return f"""
    <div class="section-block">
      <div class="block-title">🔄 Iteration History</div>
      {items}
    </div>"""


def _render_full_page(
    problem_sections: list[str],
    metrics,
    mode: str,
    model: str,
) -> str:
    solved_pct = (metrics.solved / metrics.total_problems * 100) if metrics.total_problems else 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    problems_html = "\n".join(problem_sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Benchmark Report — {_e(mode)} · {_e(model)}</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh;padding:40px 48px}}
    h1{{font-size:1.4rem;font-weight:700;color:#f8fafc;margin-bottom:4px}}
    .sub{{color:#64748b;font-size:.85rem;margin-bottom:32px}}
    .metrics-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px;margin-bottom:40px}}
    .metric-card{{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:16px 18px}}
    .metric-card .val{{font-size:1.6rem;font-weight:700;color:#38bdf8;margin-bottom:4px}}
    .metric-card .lbl{{font-size:.75rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em}}
    .problem-card{{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:22px 24px;margin-bottom:24px}}
    .problem-header{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px;flex-wrap:wrap;gap:8px}}
    .problem-id{{font-size:1rem;font-weight:700;color:#f1f5f9;margin-right:10px}}
    .status-badge{{font-size:.7rem;font-weight:700;text-transform:uppercase;padding:3px 10px;border-radius:20px}}
    .status-pass{{background:#064e3b;color:#34d399}}
    .status-fail{{background:#450a0a;color:#f87171}}
    .problem-meta{{font-size:.78rem;color:#475569}}
    .section-block{{margin:14px 0}}
    .block-title{{font-size:.78rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}}
    .pre-block{{background:#0f1117;border:1px solid #1e293b;border-radius:6px;padding:12px 14px;font-family:"SF Mono","Fira Code",monospace;font-size:.74rem;line-height:1.7;color:#a5f3fc;overflow-x:auto;white-space:pre-wrap;word-break:break-word}}
    .code-block{{color:#e2e8f0}}
    .test-block{{color:#94a3b8}}
    .narrative-block{{background:#0c1a2e;border:1px solid #1e40af;border-radius:8px;padding:14px 18px;margin-top:14px}}
    .narrative-block p{{font-size:.85rem;color:#93c5fd;line-height:1.8}}
    .iter-entry{{border:1px solid #1e293b;border-radius:8px;padding:12px 14px;margin:8px 0}}
    .iter-header{{display:flex;align-items:center;gap:14px;margin-bottom:8px;flex-wrap:wrap}}
    .iter-num{{font-size:.8rem;font-weight:700;color:#e2e8f0}}
    .iter-rate{{font-size:.75rem;font-weight:600;padding:2px 8px;border-radius:12px}}
    .iter-pass{{background:#064e3b;color:#34d399}}
    .iter-partial{{background:#451a03;color:#fbbf24}}
    .iter-fail{{background:#450a0a;color:#f87171}}
    .iter-tokens{{font-size:.72rem;color:#475569}}
    details{{margin-top:6px}}
    summary{{font-size:.78rem;color:#64748b;cursor:pointer;padding:4px 0}}
    summary:hover{{color:#94a3b8}}
    hr{{border:none;border-top:1px solid #1e293b;margin:32px 0}}
  </style>
</head>
<body>
  <h1>Benchmark Report</h1>
  <p class="sub">
    Mode: <strong style="color:#e2e8f0">{_e(mode)}</strong> &nbsp;·&nbsp;
    Model: <strong style="color:#e2e8f0">{_e(model)}</strong> &nbsp;·&nbsp;
    Generated: {now}
  </p>

  <div class="metrics-grid">
    <div class="metric-card">
      <div class="val">{metrics.total_problems}</div>
      <div class="lbl">Problems</div>
    </div>
    <div class="metric-card">
      <div class="val">{metrics.solved}</div>
      <div class="lbl">Solved</div>
    </div>
    <div class="metric-card">
      <div class="val">{solved_pct:.0f}%</div>
      <div class="lbl">Solve Rate</div>
    </div>
    <div class="metric-card">
      <div class="val">{metrics.overall_pass_rate:.0%}</div>
      <div class="lbl">Avg Pass Rate</div>
    </div>
    <div class="metric-card">
      <div class="val">{metrics.avg_iterations:.1f}</div>
      <div class="lbl">Avg Iterations</div>
    </div>
    <div class="metric-card">
      <div class="val">{metrics.total_tokens:,}</div>
      <div class="lbl">Total Tokens</div>
    </div>
    <div class="metric-card">
      <div class="val">{int(metrics.avg_tokens_per_problem):,}</div>
      <div class="lbl">Tokens / Problem</div>
    </div>
  </div>

  <hr/>

  {problems_html}

</body>
</html>"""
