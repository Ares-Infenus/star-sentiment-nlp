"""
Phase 3 — Word2Vec Embedder.

Trains a Word2Vec model (gensim) and produces document vectors
by mean-pooling token vectors.

Interface
---------
    emb = Word2VecEmbedder(vector_size=300, window=5, min_count=1)
    emb.fit(texts)
    vectors = emb.transform(texts)   # np.ndarray (n, 300)
    emb.save(path)
    emb2 = Word2VecEmbedder.load(path)
"""

import joblib
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec


class Word2VecEmbedder:
    def __init__(
        self,
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        epochs: int = 10,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model: Word2Vec | None = None

    @staticmethod
    def _tokenize(texts: list[str]) -> list[list[str]]:
        return [t.split() for t in texts]

    def fit(self, texts: list[str]) -> "Word2VecEmbedder":
        sentences = self._tokenize(texts)
        self.model = Word2Vec(
            sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            seed=42,
        )
        return self

    def _text_to_vector(self, text: str) -> np.ndarray:
        tokens = text.split()
        vecs = [self.model.wv[t] for t in tokens if t in self.model.wv]
        if vecs:
            return np.mean(vecs, axis=0)
        return np.zeros(self.vector_size, dtype=np.float32)

    def transform(self, texts: list[str]) -> np.ndarray:
        """Returns ndarray of shape (n_samples, vector_size)."""
        return np.array([self._text_to_vector(t) for t in texts], dtype=np.float32)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "Word2VecEmbedder":
        return joblib.load(path)
