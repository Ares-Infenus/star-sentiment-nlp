"""
Phase 3 — TF-IDF Embedder.

Interface
---------
    emb = TFIDFEmbedder(max_features=10_000, ngram_range=(1, 2))
    emb.fit(texts)
    vectors = emb.transform(texts)   # scipy sparse matrix
    emb.save(path)
    emb2 = TFIDFEmbedder.load(path)
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFEmbedder:
    def __init__(self, max_features: int = 10_000, ngram_range: tuple = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )
        self.is_fitted = False

    def fit(self, texts: list[str]) -> "TFIDFEmbedder":
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts: list[str]):
        """Returns a scipy sparse matrix of shape (n_samples, n_features)."""
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: list[str]):
        self.is_fitted = True
        return self.vectorizer.fit_transform(texts)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TFIDFEmbedder":
        return joblib.load(path)
