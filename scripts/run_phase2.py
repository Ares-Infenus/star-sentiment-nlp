"""
Phase 2 — Run NLP preprocessing pipeline on all splits.

Usage
-----
    python scripts/run_phase2.py

After this script: run pytest tests/phase2/ to verify.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Download NLTK data if missing
import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

from src.preprocessing.pipeline import run_preprocessing_pipeline


def main() -> None:
    print("=" * 60)
    print("PHASE 2 — NLP Preprocessing")
    print("=" * 60)

    run_preprocessing_pipeline()

    print("\nPhase 2 complete. Run: pytest tests/phase2/ -v")


if __name__ == "__main__":
    main()
