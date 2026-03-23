"""
Run the complete pipeline from scratch: Phase 1 -> Phase 4 + final report.

Usage
-----
    python scripts/run_all.py [--skip-bert] [--epochs N]

After completion, run: pytest tests/ -v
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
PHASES = [
    ("Phase 1 - Data Acquisition", [sys.executable, str(SCRIPTS_DIR / "run_phase1.py")]),
    ("Phase 2 - Preprocessing",    [sys.executable, str(SCRIPTS_DIR / "run_phase2.py")]),
    ("Phase 3 - Embeddings",       [sys.executable, str(SCRIPTS_DIR / "run_phase3.py")]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bert", action="store_true", help="Skip DistilBERT fine-tuning")
    parser.add_argument("--epochs", type=int, default=3, help="DistilBERT training epochs")
    args = parser.parse_args()

    total_t0 = time.time()

    # Phase 1-3
    for name, cmd in PHASES:
        print(f"\n{'#' * 60}")
        print(f"# {name}")
        print(f"{'#' * 60}")
        result = subprocess.run(cmd, cwd=str(SCRIPTS_DIR.parent))
        if result.returncode != 0:
            print(f"\nERROR: {name} failed (exit code {result.returncode})")
            sys.exit(1)

    # Phase 4
    print(f"\n{'#' * 60}")
    print(f"# Phase 4 - Model Training")
    print(f"{'#' * 60}")
    phase4_cmd = [sys.executable, str(SCRIPTS_DIR / "run_phase4.py")]
    if args.skip_bert:
        phase4_cmd.append("--skip-bert")
    else:
        phase4_cmd.extend(["--epochs", str(args.epochs)])
    result = subprocess.run(phase4_cmd, cwd=str(SCRIPTS_DIR.parent))
    if result.returncode != 0:
        print(f"\nERROR: Phase 4 failed (exit code {result.returncode})")
        sys.exit(1)

    # Final report
    print(f"\n{'#' * 60}")
    print(f"# Generating Final Report")
    print(f"{'#' * 60}")
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "generate_final_report.py")],
        cwd=str(SCRIPTS_DIR.parent),
    )
    if result.returncode != 0:
        print(f"\nERROR: Report generation failed (exit code {result.returncode})")
        sys.exit(1)

    elapsed = time.time() - total_t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\n{'=' * 60}")
    print(f"All phases complete in {minutes}m {seconds}s")
    print(f"Now run: pytest tests/ -v")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
