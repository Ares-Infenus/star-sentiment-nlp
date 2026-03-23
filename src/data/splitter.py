"""
Phase 1 — Stratified train / val / test splitter.

Splits a DataFrame and saves CSVs to data/splits/{domain}/.
Ratios: 70% train / 15% val / 15% test (stratified by label).
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SPLITS_PATH = Path("data/splits")
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def split_dataframe(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df, test_df) with stratified splits."""
    test_size = 1.0 - train_ratio
    train_df, temp_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    # val_ratio relative to the remaining 30 %
    val_fraction = val_ratio / (val_ratio + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df, test_size=1.0 - val_fraction, stratify=temp_df["label"], random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_splits(domain: str, train_df, val_df, test_df) -> None:
    """Save three CSVs to data/splits/{domain}/."""
    out_dir = SPLITS_PATH / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    print(
        f"  {domain}: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}"
    )


def split_and_save(domain: str, df: pd.DataFrame, seed: int = 42) -> None:
    """Convenience: split + save in one call."""
    # Normalize whitespace and strip to prevent near-duplicate leakage
    df["text"] = df["text"].str.strip().str.replace(r'\s+', ' ', regex=True)
    # Drop duplicate texts (keep first) to prevent data leakage
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    train_df, val_df, test_df = split_dataframe(df, seed=seed)
    save_splits(domain, train_df, val_df, test_df)
