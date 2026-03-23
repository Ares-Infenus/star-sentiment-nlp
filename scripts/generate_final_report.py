"""
Generates reports/final_comparison.md summarising all model results.
Run this after completing Phase 4.

Usage
-----
    python scripts/generate_final_report.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


MODELS_PATH = Path("models")
REPORTS_PATH = Path("reports")
REPORTS_PATH.mkdir(exist_ok=True)


def load_test_data():
    frames = []
    for domain in ["amazon_reviews", "tweets", "financial_news"]:
        p = Path(f"data/splits/{domain}/test.csv")
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else None


def evaluate_model(model, texts, labels):
    from sklearn.metrics import accuracy_score, f1_score
    preds = model.predict(texts)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


def main():
    df = load_test_data()
    if df is None:
        print("No test data found. Run scripts/run_phase1.py first.")
        return

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    rows = []

    svm_path = MODELS_PATH / "tfidf_svm.joblib"
    if svm_path.exists():
        from src.models.svm_model import SVMClassifier
        model = SVMClassifier.load(svm_path)
        t0 = time.perf_counter()
        m = evaluate_model(model, texts[:1000], labels[:1000])
        ms = (time.perf_counter() - t0) * 1000 / min(1000, len(texts))
        rows.append({"Model": "SVM + TF-IDF", **m, "inference_ms": f"{ms:.1f}"})

    xgb_path = MODELS_PATH / "tfidf_xgboost.joblib"
    if xgb_path.exists():
        from src.models.xgboost_model import XGBoostClassifier
        model = XGBoostClassifier.load(xgb_path)
        t0 = time.perf_counter()
        m = evaluate_model(model, texts[:1000], labels[:1000])
        ms = (time.perf_counter() - t0) * 1000 / min(1000, len(texts))
        rows.append({"Model": "XGBoost + TF-IDF", **m, "inference_ms": f"{ms:.1f}"})

    bert_path = MODELS_PATH / "distilbert_finetuned"
    if bert_path.exists():
        from src.models.distilbert_classifier import DistilBERTClassifier
        model = DistilBERTClassifier.load(bert_path)
        t0 = time.perf_counter()
        m = evaluate_model(model, texts[:200], labels[:200])
        ms = (time.perf_counter() - t0) * 1000 / min(200, len(texts))
        rows.append({"Model": "DistilBERT fine-tuned", **m, "inference_ms": f"{ms:.1f}"})

    if not rows:
        print("No trained models found. Run scripts/run_phase4.py first.")
        return

    result_df = pd.DataFrame(rows)
    result_df["accuracy"] = result_df["accuracy"].map("{:.4f}".format)
    result_df["f1_macro"] = result_df["f1_macro"].map("{:.4f}".format)
    result_df.rename(columns={"inference_ms": "Inference (ms/sample)"}, inplace=True)

    md = "# Final Model Comparison\n\n"
    md += result_df.to_markdown(index=False)
    md += "\n\n_Generated automatically by scripts/generate_final_report.py_\n"

    out = REPORTS_PATH / "final_comparison.md"
    out.write_text(md)
    print(f"Report saved -> {out}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
