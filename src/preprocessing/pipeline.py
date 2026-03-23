"""
Phase 2 -- Batch preprocessing pipeline.

Reads all splits for every domain, applies full_preprocess,
and writes data/processed/{domain}_processed.csv.

Usage
-----
    from src.preprocessing.pipeline import run_preprocessing_pipeline
    run_preprocessing_pipeline()
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.preprocessing.cleaner import full_preprocess

SPLITS_PATH = Path("data/splits")
PROCESSED_PATH = Path("data/processed")
DOMAINS = ["amazon_reviews", "tweets", "financial_news"]
SPLITS = ["train", "val", "test"]
MIN_WORDS = 3
MAX_WORDS = 512


def _preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    tqdm.pandas(desc="  preprocessing")
    df = df.copy()
    df["text_processed"] = df["text"].progress_apply(
        lambda x: full_preprocess(str(x)) if pd.notna(x) else ""
    )
    # Filter length bounds
    wc = df["text_processed"].str.split().str.len()
    df = df[(wc >= MIN_WORDS) & (wc <= MAX_WORDS)].copy()
    df.dropna(subset=["text_processed"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def run_preprocessing_pipeline(domains: list[str] | None = None) -> Path:
    """Process all domains and save CSVs. Returns the processed/ path."""
    if domains is None:
        domains = DOMAINS

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        frames = []
        for split in SPLITS:
            path = SPLITS_PATH / domain / f"{split}.csv"
            if path.exists():
                frames.append(pd.read_csv(path))

        if not frames:
            print(f"[WARN] No split files found for {domain}, skipping.")
            continue

        combined = pd.concat(frames, ignore_index=True)
        print(f"\n{domain} -- {len(combined):,} total samples")

        processed = _preprocess_df(combined)
        out = PROCESSED_PATH / f"{domain}_processed.csv"
        processed.to_csv(out, index=False)
        print(f"  saved {len(processed):,} rows -> {out}")

    return PROCESSED_PATH
