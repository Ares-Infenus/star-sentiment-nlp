"""
Phase 4 — Train and evaluate all models.

Trains:
  - SVM + TF-IDF
  - XGBoost + TF-IDF
  - DistilBERT fine-tuned (optional, set --skip-bert to skip)

Saves models to models/ and generates reports/phase_reports/phase4_report.md.

Usage
-----
    python scripts/run_phase4.py [--skip-bert] [--epochs N]

After this script: run pytest tests/phase4/ to verify.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.svm_model import SVMClassifier
from src.models.xgboost_model import XGBoostClassifier
from src.models.distilbert_classifier import DistilBERTClassifier
from src.evaluation.metrics import compute_metrics, print_metrics, plot_confusion_matrix

MODELS_PATH = Path("models")
REPORTS_PATH = Path("reports/phase_reports")
MODELS_PATH.mkdir(exist_ok=True)
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

DOMAINS = ["amazon_reviews", "tweets", "financial_news"]


def load_data(split: str) -> tuple[list[str], list[int]]:
    """Load raw texts and labels from data/splits/.

    Uses raw text so that train and test use the same format.
    The TF-IDF vectorizer inside each model handles its own tokenization.
    """
    frames = []
    for domain in DOMAINS:
        p = Path(f"data/splits/{domain}/{split}.csv")
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"No {split} data found. Run scripts/run_phase1.py first.")

    df = pd.concat(frames, ignore_index=True)
    # Drop any rows with null text
    df = df.dropna(subset=["text"])
    return df["text"].tolist(), df["label"].tolist()


def train_and_eval(model, model_key: str, train_texts, train_labels, test_texts, test_labels):
    print(f"\n{'='*50}")
    print(f"[{model_key}] Training…")
    print(f"{'='*50}")
    t0 = time.time()
    model.fit(train_texts, train_labels)
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.1f}s")

    print(f"  Evaluating on {len(test_texts):,} test samples…")
    preds = model.predict(test_texts)
    metrics = compute_metrics(test_labels, preds)
    print_metrics(metrics, model_key)

    save_path = MODELS_PATH / f"{model_key}.joblib"
    model.save(save_path)
    print(f"  Saved -> {save_path}")

    return metrics


def main(skip_bert: bool = False, bert_epochs: int = 3) -> None:
    print("=" * 60)
    print("PHASE 4 — Model Training & Evaluation")
    print("=" * 60)

    train_texts, train_labels = load_data("train")
    test_texts, test_labels = load_data("test")
    print(f"\nTrain: {len(train_texts):,} samples | Test: {len(test_texts):,} samples")

    report_lines = ["# Phase 4 — Model Evaluation Report\n"]

    # ── SVM ─────────────────────────────────────────────────────────────────
    svm = SVMClassifier(max_features=50_000, ngram_range=(1, 2), C=1.0)
    m = train_and_eval(svm, "tfidf_svm", train_texts, train_labels, test_texts, test_labels)
    report_lines.append(
        f"## SVM + TF-IDF\n- Accuracy: {m['accuracy']:.4f}\n- F1 Macro: {m['f1_macro']:.4f}\n\n{m['report']}\n"
    )

    # ── XGBoost ──────────────────────────────────────────────────────────────
    xgb = XGBoostClassifier(max_features=50_000, ngram_range=(1, 2), n_estimators=300)
    m = train_and_eval(xgb, "tfidf_xgboost", train_texts, train_labels, test_texts, test_labels)
    report_lines.append(
        f"## XGBoost + TF-IDF\n- Accuracy: {m['accuracy']:.4f}\n- F1 Macro: {m['f1_macro']:.4f}\n\n{m['report']}\n"
    )

    # ── DistilBERT ────────────────────────────────────────────────────────────
    if not skip_bert:
        val_texts, val_labels = load_data("val")
        bert = DistilBERTClassifier(max_length=256)
        print(f"\n{'='*50}")
        print(f"[distilbert_finetuned] Fine-tuning ({bert_epochs} epochs)…")
        print(f"{'='*50}")
        t0 = time.time()
        bert.fit(train_texts, train_labels, val_texts, val_labels, epochs=bert_epochs)
        print(f"  Trained in {time.time()-t0:.1f}s")

        preds = bert.predict(test_texts)
        m = compute_metrics(test_labels, preds)
        print_metrics(m, "DistilBERT")

        bert.save(MODELS_PATH / "distilbert_finetuned")
        report_lines.append(
            f"## DistilBERT fine-tuned\n- Accuracy: {m['accuracy']:.4f}\n- F1 Macro: {m['f1_macro']:.4f}\n\n{m['report']}\n"
        )
    else:
        print("\n[DistilBERT] Skipped (--skip-bert)")

    # ── Write report ─────────────────────────────────────────────────────────
    report_path = REPORTS_PATH / "phase4_report.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\nReport saved -> {report_path}")

    print("\nPhase 4 complete. Run: pytest tests/phase4/ -v")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bert", action="store_true", help="Skip DistilBERT fine-tuning")
    parser.add_argument("--epochs", type=int, default=3, help="DistilBERT training epochs")
    args = parser.parse_args()
    main(skip_bert=args.skip_bert, bert_epochs=args.epochs)
