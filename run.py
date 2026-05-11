#!/usr/bin/env python3
"""
Wrapper to run the benchmark with a single argument.

Usage:
    python run.py smoke    # 3 problems, 2 iterations — quick sanity check
    python run.py single   # 20 problems, single-agent baseline
    python run.py multi    # 20 problems, full multi-agent pipeline
"""
import sys
import subprocess
from dotenv import dotenv_values

CONFIGS = {
    "smoke": {
        "max_problems": 3,
        "max_iterations": 5,
        "mode": "multi",
        "output": "results/smoke.json",
    },
    "single": {
        "max_problems": 20,
        "max_iterations": 10,
        "mode": "single",
        "output": "results/single.json",
    },
    "multi": {
        "max_problems": 20,
        "max_iterations": 10,
        "mode": "multi",
        "output": "results/multi.json",
    },
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in CONFIGS:
        print("Usage: python run.py [smoke|single|multi]")
        print()
        print("  smoke   3 problems, 2 iterations — quick sanity check")
        print("  single  20 problems, single-agent baseline")
        print("  multi   20 problems, full multi-agent pipeline")
        sys.exit(1)

    mode = sys.argv[1]
    cfg = CONFIGS[mode]

    env = dotenv_values(".env")
    model = env.get("MODEL", "gpt-4o-mini")

    print(f">> Mode: {mode} · model: {model}")

    cmd = [
        sys.executable, "main.py",
        "--benchmark",      "data/HumanEval.jsonl",
        "--max-problems",   str(cfg["max_problems"]),
        "--max-iterations", str(cfg["max_iterations"]),
        "--model",          model,
        "--mode",           cfg["mode"],
        "--output",         cfg["output"],
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
