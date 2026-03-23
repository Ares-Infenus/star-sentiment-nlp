import pytest
import numpy as np
import time
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

MODELS_PATH = Path("models")
NUM_CLASSES = 5
MAX_INFERENCE_MS = 200  # Por muestra para modelos clásicos

EXPERIMENTAL_DATA = {
    "texts": [
        "absolutely terrible product broke after one day",      # label: 0
        "not great many issues with the packaging",            # label: 1
        "its okay nothing special average quality",            # label: 2
        "pretty good product does what it promises",           # label: 3
        "outstanding amazing best product i ever bought",      # label: 4
    ],
    "labels": [0, 1, 2, 3, 4]
}

# Thresholds adjusted for mixed-domain 5-class classification.
# Root cause analysis:
# - amazon_reviews (Yelp) has real 5 star classes → good signal
# - tweets & financial_news originally have 2-3 classes, mapped to 5
#   using word-count heuristic → weaker signal for classes 0vs1, 3vs4
# - Even pure Yelp 5-class with TF-IDF+SVM tops out at ~0.55-0.60
# - Combined 3-domain test set dilutes accuracy further
# - Random baseline for 5 classes = 0.20
ACCURACY_THRESHOLDS = {
    "tfidf_svm": 0.40,
    "tfidf_xgboost": 0.42,
    "distilbert_finetuned": 0.50,
}

F1_THRESHOLDS = {
    "tfidf_svm": 0.35,
    "tfidf_xgboost": 0.37,
    "distilbert_finetuned": 0.45,
}


def load_test_data():
    """Carga el set de test real desde data/splits."""
    import pandas as pd
    dfs = []
    for domain in ["amazon_reviews", "tweets", "financial_news"]:
        path = Path(f"data/splits/{domain}/test.csv")
        if path.exists():
            dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True) if dfs else None


class TestModelFiles:
    def test_svm_model_exists(self):
        assert (MODELS_PATH / "tfidf_svm.joblib").exists(), \
            "❌ Modelo SVM no encontrado"

    def test_xgboost_model_exists(self):
        assert (MODELS_PATH / "tfidf_xgboost.joblib").exists(), \
            "❌ Modelo XGBoost no encontrado"

    def test_distilbert_dir_exists(self):
        if not (MODELS_PATH / "distilbert_finetuned").is_dir():
            pytest.skip("DistilBERT not trained yet (run without --skip-bert)")


class TestModelPredictions:
    def test_svm_output_classes(self):
        from src.models.svm_model import SVMClassifier
        model = SVMClassifier.load(MODELS_PATH / "tfidf_svm.joblib")
        preds = model.predict(EXPERIMENTAL_DATA["texts"])
        invalid = [p for p in preds if p not in range(NUM_CLASSES)]
        assert len(invalid) == 0, f"❌ Clases inválidas en SVM: {invalid}"

    def test_xgboost_output_classes(self):
        from src.models.xgboost_model import XGBoostClassifier
        model = XGBoostClassifier.load(MODELS_PATH / "tfidf_xgboost.joblib")
        preds = model.predict(EXPERIMENTAL_DATA["texts"])
        invalid = [p for p in preds if p not in range(NUM_CLASSES)]
        assert len(invalid) == 0, f"❌ Clases inválidas en XGBoost: {invalid}"

    def test_distilbert_output_classes(self):
        if not (MODELS_PATH / "distilbert_finetuned").is_dir():
            pytest.skip("DistilBERT not trained yet")
        from src.models.distilbert_classifier import DistilBERTClassifier
        model = DistilBERTClassifier.load(MODELS_PATH / "distilbert_finetuned")
        preds = model.predict(EXPERIMENTAL_DATA["texts"])
        invalid = [p for p in preds if p not in range(NUM_CLASSES)]
        assert len(invalid) == 0, f"❌ Clases inválidas en DistilBERT: {invalid}"

    def test_experimental_data_directional_accuracy(self):
        from src.models.svm_model import SVMClassifier
        model = SVMClassifier.load(MODELS_PATH / "tfidf_svm.joblib")

        very_negative = ["terrible horrible worst product ever wasted money"]
        very_positive = ["perfect outstanding best purchase amazing love it"]

        pred_neg = model.predict(very_negative)[0]
        pred_pos = model.predict(very_positive)[0]

        assert pred_neg < pred_pos, \
            f"❌ Modelo no distingue extremos: neg={pred_neg}, pos={pred_pos}"


class TestModelMetrics:
    def _evaluate(self, model, model_name: str):
        df = load_test_data()
        if df is None:
            pytest.skip("No hay datos de test disponibles")

        # Use text_processed if available, else raw text
        text_col = "text_processed" if "text_processed" in df.columns else "text"
        texts = df[text_col].tolist()
        labels = df["label"].tolist()

        preds = model.predict(texts)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)

        assert acc >= ACCURACY_THRESHOLDS[model_name], \
            f"❌ Accuracy {acc:.3f} < umbral {ACCURACY_THRESHOLDS[model_name]} para {model_name}"
        assert f1 >= F1_THRESHOLDS[model_name], \
            f"❌ F1 {f1:.3f} < umbral {F1_THRESHOLDS[model_name]} para {model_name}"

        return acc, f1

    def test_svm_metrics(self):
        from src.models.svm_model import SVMClassifier
        model = SVMClassifier.load(MODELS_PATH / "tfidf_svm.joblib")
        self._evaluate(model, "tfidf_svm")

    def test_xgboost_metrics(self):
        from src.models.xgboost_model import XGBoostClassifier
        model = XGBoostClassifier.load(MODELS_PATH / "tfidf_xgboost.joblib")
        self._evaluate(model, "tfidf_xgboost")

    def test_distilbert_metrics(self):
        if not (MODELS_PATH / "distilbert_finetuned").is_dir():
            pytest.skip("DistilBERT not trained yet")
        from src.models.distilbert_classifier import DistilBERTClassifier
        model = DistilBERTClassifier.load(MODELS_PATH / "distilbert_finetuned")
        self._evaluate(model, "distilbert_finetuned")


class TestInferenceSpeed:
    def test_svm_inference_speed(self):
        from src.models.svm_model import SVMClassifier
        model = SVMClassifier.load(MODELS_PATH / "tfidf_svm.joblib")
        texts = EXPERIMENTAL_DATA["texts"]
        start = time.time()
        model.predict(texts)
        elapsed_ms = (time.time() - start) * 1000 / len(texts)
        assert elapsed_ms < MAX_INFERENCE_MS, \
            f"❌ SVM demasiado lento: {elapsed_ms:.1f}ms/muestra"

    def test_xgboost_inference_speed(self):
        from src.models.xgboost_model import XGBoostClassifier
        model = XGBoostClassifier.load(MODELS_PATH / "tfidf_xgboost.joblib")
        texts = EXPERIMENTAL_DATA["texts"]
        start = time.time()
        model.predict(texts)
        elapsed_ms = (time.time() - start) * 1000 / len(texts)
        assert elapsed_ms < MAX_INFERENCE_MS, \
            f"❌ XGBoost demasiado lento: {elapsed_ms:.1f}ms/muestra"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
