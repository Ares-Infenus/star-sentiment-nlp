"""
Phase 1 -- Download datasets and generate stratified splits.

Usage
-----
    python scripts/run_phase1.py [--max-per-class N]

After this script: run pytest tests/phase1/ to verify.
"""

import argparse
import sys
from pathlib import Path

# Make src importable when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_all
from src.data.splitter import split_and_save


def main(max_per_class: int = 5000) -> None:
    print("=" * 60)
    print("PHASE 1 -- Data Acquisition & Splitting")
    print("=" * 60)

    datasets = load_all(max_per_class=max_per_class)

    print("\nSplitting into train / val / test (70/15/15)...")
    for domain, df in datasets.items():
        print(f"\n[{domain}]  total={len(df):,}  classes={sorted(df['label'].unique())}")
        split_and_save(domain, df)

    print("\nPhase 1 complete. Run: pytest tests/phase1/ -v")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=5000,
        help="Max samples per class per domain (default: 5000)",
    )
    args = parser.parse_args()
    main(args.max_per_class)
