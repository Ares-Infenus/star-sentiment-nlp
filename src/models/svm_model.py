"""
Phase 4 -- SVM Classifier with integrated TF-IDF pipeline.

The model wraps TF-IDF + CalibratedClassifierCV(LinearSVC) in a single
sklearn Pipeline so that predict() accepts raw text strings.

Interface
---------
    clf = SVMClassifier()
    clf.fit(texts, labels)
    preds = clf.predict(texts)          # np.ndarray of int
    probs = clf.predict_proba(texts)    # np.ndarray (n, 5)
    clf.save(path)
    clf2 = SVMClassifier.load(path)
"""

import joblib
import numpy as np
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


class SVMClassifier:
    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: tuple = (1, 2),
        C: float = 1.0,
        max_iter: int = 2000,
    ):
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "svm",
                    CalibratedClassifierCV(
                        LinearSVC(C=C, max_iter=max_iter, random_state=42),
                        cv=3,
                    ),
                ),
            ]
        )

    def fit(self, texts: list[str], labels) -> "SVMClassifier":
        print(f"  Vectorizing {len(texts):,} texts with TF-IDF...", flush=True)
        print(f"  Training SVM (CalibratedClassifierCV, cv=3)...", flush=True)
        self.pipeline.fit(texts, labels)
        print(f"  SVM training done.", flush=True)
        return self

    def predict(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.pipeline.predict(texts)

    def predict_proba(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.pipeline.predict_proba(texts)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "SVMClassifier":
        return joblib.load(path)
