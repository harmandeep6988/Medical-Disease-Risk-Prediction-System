from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "dashboard" / "app.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Streamlit dashboard from any working directory.")
    parser.add_argument("--port", type=int, default=8501, help="Port for Streamlit server.")
    parser.add_argument("--address", default="0.0.0.0", help="Bind address for Streamlit server.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not APP_PATH.exists():
        raise SystemExit(f"Dashboard app not found at: {APP_PATH}")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.headless",
        "true" if args.headless else "false",
        "--server.port",
        str(args.port),
        "--server.address",
        args.address,
    ]

    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
