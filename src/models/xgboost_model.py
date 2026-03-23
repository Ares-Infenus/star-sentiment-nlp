"""
Phase 4 -- XGBoost Classifier with integrated TF-IDF.

Interface
---------
    clf = XGBoostClassifier()
    clf.fit(texts, labels)
    preds = clf.predict(texts)
    probs = clf.predict_proba(texts)
    clf.save(path)
    clf2 = XGBoostClassifier.load(path)
"""

import joblib
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier


class XGBoostClassifier:
    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: tuple = (1, 2),
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="multi:softprob",
            num_class=5,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=n_jobs,
            verbosity=0,
        )

    def fit(self, texts: list[str], labels) -> "XGBoostClassifier":
        print("  Vectorizing with TF-IDF...", flush=True)
        X = self.tfidf.fit_transform(texts)
        print(f"  TF-IDF done: {X.shape[0]:,} samples × {X.shape[1]:,} features")
        print(f"  Training XGBoost ({self.n_estimators} rounds)...", flush=True)
        # Use eval_set + verbose_eval for progress
        self.model.set_params(verbosity=1)
        self.model.fit(X, labels, eval_set=[(X, labels)], verbose=50)
        self.model.set_params(verbosity=0)
        return self

    def predict(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        X = self.tfidf.transform(texts)
        return self.model.predict(X)

    def predict_proba(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        X = self.tfidf.transform(texts)
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostClassifier":
        return joblib.load(path)
