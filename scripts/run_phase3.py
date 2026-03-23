"""
Phase 3 -- Fit and compare the 3 embedding strategies.

Saves TF-IDF and Word2Vec embedders to models/.
DistilBERT doesn't need saving (loaded on demand from HuggingFace).

Usage
-----
    python scripts/run_phase3.py

After this script: run pytest tests/phase3/ to verify.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.tfidf import TFIDFEmbedder
from src.embeddings.word2vec import Word2VecEmbedder
from src.embeddings.distilbert import DistilBERTEmbedder

MODELS_PATH = Path("models")
MODELS_PATH.mkdir(exist_ok=True)

SAMPLE_SIZE = 2000  # rows used for W2V training (speed)


def load_train_texts() -> list[str]:
    frames = []
    for domain in ["amazon_reviews", "tweets", "financial_news"]:
        p = Path(f"data/processed/{domain}_processed.csv")
        if p.exists():
            df = pd.read_csv(p)
            # Use text_processed if available
            col = "text_processed" if "text_processed" in df.columns else "text"
            frames.append(df[col].dropna())
    if not frames:
        raise FileNotFoundError("No processed data found. Run scripts/run_phase2.py first.")
    combined = pd.concat(frames, ignore_index=True)
    return combined.tolist()


def benchmark(name: str, embedder, texts: list[str]) -> None:
    sample = texts[:100]
    t0 = time.perf_counter()
    embedder.transform(sample)
    ms = (time.perf_counter() - t0) * 1000 / len(sample)
    print(f"  {name}: {ms:.1f} ms/sample")


def main() -> None:
    print("=" * 60)
    print("PHASE 3 -- Embedding Comparison")
    print("=" * 60)

    texts = load_train_texts()
    print(f"Loaded {len(texts):,} training texts.")

    # ── TF-IDF ──────────────────────────────────────────────────────────────
    print("\n[TF-IDF] Fitting...")
    tfidf = TFIDFEmbedder(max_features=50_000, ngram_range=(1, 2))
    tfidf.fit(texts)
    tfidf.save(MODELS_PATH / "tfidf_embedder.joblib")
    print(f"  vocab size: {len(tfidf.vectorizer.vocabulary_):,}")
    benchmark("TF-IDF", tfidf, texts)

    # ── Word2Vec ─────────────────────────────────────────────────────────────
    print("\n[Word2Vec] Training (this may take a few minutes)...")
    w2v = Word2VecEmbedder(vector_size=300, window=5, min_count=2, epochs=10)
    w2v.fit(texts[:SAMPLE_SIZE] if len(texts) > SAMPLE_SIZE else texts)
    w2v.save(MODELS_PATH / "word2vec_embedder.joblib")
    benchmark("Word2Vec", w2v, texts)

    # ── DistilBERT ───────────────────────────────────────────────────────────
    print("\n[DistilBERT] Loading model and benchmarking (no training needed)...")
    bert = DistilBERTEmbedder()
    benchmark("DistilBERT", bert, texts[:10])  # only 10 for speed

    print("\nPhase 3 complete. Run: pytest tests/phase3/ -v")


if __name__ == "__main__":
    main()
