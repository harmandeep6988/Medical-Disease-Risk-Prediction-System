from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_step(command: list[str], name: str) -> None:
    print(f"\n=== {name} ===")
    print(" ".join(command))
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {name} (exit code={result.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full medical risk predictor workflow.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip dataset preparation step.")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip launching Streamlit dashboard.")
    parser.add_argument("--port", type=int, default=8501, help="Dashboard port when launching Streamlit.")
    args = parser.parse_args()

    if not args.skip_prepare:
        run_step([PYTHON, "scripts/prepare_datasets.py"], "Prepare datasets")

    run_step([PYTHON, "scripts/train_and_evaluate.py"], "Train and evaluate models")

    if not args.no_dashboard:
        run_step(
            [
                PYTHON,
                "-m",
                "streamlit",
                "run",
                str(ROOT / "dashboard" / "app.py"),
                "--server.headless",
                "true",
                "--server.port",
                str(args.port),
            ],
            "Launch dashboard",
        )


if __name__ == "__main__":
    main()
