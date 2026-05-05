from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_step(command: list[str], name: str) -> None:
    print(f"\n=== {name} ===")
    print(" ".join(command))
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {name} (exit code={result.returncode})")


def _check_kaggle_setup() -> None:
    kaggle_config_dir = None
    if "KAGGLE_CONFIG_DIR" in os.environ:
        kaggle_config_dir = Path(os.environ["KAGGLE_CONFIG_DIR"])
    else:
        project_kaggle = ROOT / "kaggle" / "kaggle.json"
        raw_kaggle = ROOT / "data" / "raw" / "kaggle" / "kaggle.json"
        home_kaggle = Path.home() / ".kaggle" / "kaggle.json"
        if raw_kaggle.exists():
            kaggle_config_dir = raw_kaggle.parent
            os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_config_dir)
        elif project_kaggle.exists():
            kaggle_config_dir = project_kaggle.parent
            os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_config_dir)
        elif home_kaggle.exists():
            kaggle_config_dir = home_kaggle.parent
            os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_config_dir)

    if kaggle_config_dir and (kaggle_config_dir / "kaggle.json").exists():
        print(f"Kaggle credentials found in: {kaggle_config_dir}")
    else:
        print("Warning: Kaggle credentials not found. Downloads may fail.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full medical risk predictor workflow.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip dataset fetch and merge step.")
    parser.add_argument("--use-local-raw", action="store_true", help="Skip Kaggle fetch and use existing data/raw files.")
    parser.add_argument("--kaggle-only", action="store_true", help="(Legacy) Use Kaggle-only sources during dataset preparation.")
    parser.add_argument("--diabetes-kaggle-dataset", default=None, help="(Legacy) Kaggle dataset slug for diabetes sources.")
    parser.add_argument("--heart-kaggle-dataset", default=None, help="(Legacy) Kaggle dataset slug for heart disease sources.")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip launching Streamlit dashboard.")
    parser.add_argument("--port", type=int, default=8501, help="Dashboard port when launching Streamlit.")
    args = parser.parse_args()

    start_time = time.time()
    _check_kaggle_setup()

    if not args.skip_prepare and not args.use_local_raw:
        run_step([PYTHON, "scripts/fetch_and_merge_datasets.py"], "Fetch and merge datasets")
        diabetes_path = ROOT / "data" / "raw" / "diabetes_merged_10k.csv"
        heart_path = ROOT / "data" / "raw" / "heart_merged_10k.csv"
        if not diabetes_path.exists() or not heart_path.exists():
            raise SystemExit(
                "Merged 10k datasets were not created. Check Kaggle credentials or dataset access permissions."
            )
    elif args.use_local_raw:
        print("Using local raw datasets in data/raw; skipping Kaggle fetch.")

    run_step([PYTHON, "scripts/train_and_evaluate.py", "--skip-overfitting"], "Train and evaluate models")

    diabetes_candidates = [
        ROOT / "data" / "raw" / "diabetes_kaggle.csv",
        ROOT / "data" / "raw" / "diabetes.csv",
        ROOT / "data" / "raw" / "diabetes_merged_10k.csv",
    ]
    heart_candidates = [
        ROOT / "data" / "raw" / "heart_kaggle.csv",
        ROOT / "data" / "raw" / "heart.csv",
        ROOT / "data" / "raw" / "heart_merged_10k.csv",
    ]

    diabetes_path = next((path for path in diabetes_candidates if path.exists()), None)
    heart_path = next((path for path in heart_candidates if path.exists()), None)

    overfit_command = [PYTHON, "scripts/detect_overfitting.py"]
    if diabetes_path:
        overfit_command.extend(["--diabetes-path", str(diabetes_path)])
    if heart_path:
        overfit_command.extend(["--heart-path", str(heart_path)])
    run_step(overfit_command, "Detect overfitting")

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

    elapsed = time.time() - start_time
    summary_path = ROOT / "reports" / "generated" / "overfitting_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        verdicts = [res.get("status", "n/a") for res in summary.get("results", [])]
        print(f"Final overfitting verdicts: {', '.join(verdicts)}")
    print(f"Total execution time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
