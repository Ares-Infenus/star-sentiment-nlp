"""
Phase 1 -- Dataset loader.

Downloads the 3 datasets from HuggingFace and maps all labels
to the unified 5-class scale (0=Very Negative ... 4=Very Positive).

Datasets used
-------------
- amazon_reviews : yelp_review_full  (5 classes, native 0-4)
- tweets         : tweet_eval/sentiment (3 classes -> mapped to 5)
- financial_news : zeroshot/twitter-financial-news-sentiment (3 -> 5)

Label mapping for 3-class datasets
------------------------------------
  original 0 (negative) -> randomly 0 or 1  (split ~50/50)
  original 1 (neutral)  -> 2
  original 2 (positive) -> randomly 3 or 4  (split ~50/50)

This keeps the ordering consistent while populating the
intermediate classes (1 and 3) that don't exist natively.
"""

import numpy as np
import pandas as pd
from datasets import load_dataset
from pathlib import Path

RAW_PATH = Path("data/raw")
DOMAINS = ["amazon_reviews", "tweets", "financial_news"]


# ─── helpers ────────────────────────────────────────────────────────────────

def _map_3_to_5(labels: np.ndarray, texts: list[str], seed: int = 42) -> np.ndarray:
    """
    Spread 3-class labels across 5 classes using text intensity.

    Instead of random assignment, uses word count as a proxy for intensity:
    - Negative + longer text -> 0 (Very Negative, more emphatic)
    - Negative + shorter text -> 1 (Negative, milder)
    - Neutral -> 2
    - Positive + shorter text -> 3 (Positive, milder)
    - Positive + longer text -> 4 (Very Positive, more emphatic)
    """
    result = np.empty(len(labels), dtype=int)

    # Pre-compute word counts
    word_counts = np.array([len(str(t).split()) for t in texts])

    for orig_label in [0, 2]:  # negative and positive
        mask = labels == orig_label
        if not mask.any():
            continue
        wc = word_counts[mask]
        median_wc = np.median(wc)

        if orig_label == 0:  # negative
            # Longer/more emphatic -> Very Negative (0), shorter -> Negative (1)
            result[mask] = np.where(wc >= median_wc, 0, 1)
        else:  # positive
            # Longer/more emphatic -> Very Positive (4), shorter -> Positive (3)
            result[mask] = np.where(wc >= median_wc, 4, 3)

    # Neutral stays neutral
    result[labels == 1] = 2

    return result


def _to_df(texts, labels, domain: str) -> pd.DataFrame:
    df = pd.DataFrame({
        "text": list(texts),
        "label": list(labels),
        "domain": domain,
    })
    df.drop_duplicates(subset=["text"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _cap_per_class(df: pd.DataFrame, max_per_class: int) -> pd.DataFrame:
    """Sample at most max_per_class rows per label."""
    return pd.concat(
        g.sample(min(len(g), max_per_class), random_state=42)
        for _, g in df.groupby("label")
    ).reset_index(drop=True)


# ─── per-domain loaders ─────────────────────────────────────────────────────

def load_amazon_reviews(max_per_class: int = 5000) -> pd.DataFrame:
    """
    Uses yelp_review_full as proxy (5 star classes, labels 0-4).
    Caps samples per class so the dataset stays manageable.
    """
    print("Downloading yelp_review_full (Amazon proxy)...")
    ds = load_dataset("yelp_review_full", split="train")
    df = pd.DataFrame({"text": ds["text"], "label": ds["label"]})
    df["domain"] = "amazon_reviews"

    if max_per_class:
        df = _cap_per_class(df, max_per_class)
    return df


def load_tweets(max_per_class: int = 5000) -> pd.DataFrame:
    """
    Uses tweet_eval/sentiment (neg=0, neu=1, pos=2) -> mapped to 5 classes.
    Combines train + validation splits for more data.
    """
    print("Downloading tweet_eval/sentiment...")
    train = load_dataset("tweet_eval", "sentiment", split="train")
    val = load_dataset("tweet_eval", "sentiment", split="validation")

    texts = list(train["text"]) + list(val["text"])
    raw_labels = np.array(list(train["label"]) + list(val["label"]))
    labels = _map_3_to_5(raw_labels, texts)

    df = _to_df(texts, labels, "tweets")

    if max_per_class:
        df = _cap_per_class(df, max_per_class)
    return df


def load_financial_news(max_per_class: int = 5000) -> pd.DataFrame:
    """
    Uses zeroshot/twitter-financial-news-sentiment (Bearish=0, Neutral=1, Bullish=2).
    Replaces financial_phrasebank which is broken on modern datasets versions.
    3 classes -> mapped to 5 with same strategy as tweets.
    """
    print("Downloading twitter-financial-news-sentiment...")
    train = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    val = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")

    texts = list(train["text"]) + list(val["text"])
    raw_labels = np.array(list(train["label"]) + list(val["label"]))
    labels = _map_3_to_5(raw_labels, texts)
    df = _to_df(texts, labels, "financial_news")

    if max_per_class:
        df = _cap_per_class(df, max_per_class)
    return df


# ─── public API ─────────────────────────────────────────────────────────────

def load_all(max_per_class: int = 5000) -> dict[str, pd.DataFrame]:
    """Return a dict {domain_name: DataFrame} for all 3 domains."""
    return {
        "amazon_reviews": load_amazon_reviews(max_per_class),
        "tweets": load_tweets(max_per_class),
        "financial_news": load_financial_news(max_per_class),
    }
