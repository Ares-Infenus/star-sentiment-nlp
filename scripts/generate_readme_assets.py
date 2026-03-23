"""Generate all visualizations for README.md and save to assets/images/."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

OUT = Path("assets/images")
OUT.mkdir(parents=True, exist_ok=True)

DOMAINS = ["amazon_reviews", "tweets", "financial_news"]
DOMAIN_LABELS = {"amazon_reviews": "Amazon Reviews", "tweets": "Tweets", "financial_news": "Financial News"}
LABEL_NAMES = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
PALETTE = ["#d32f2f", "#f57c00", "#fbc02d", "#66bb6a", "#2e7d32"]


def plot_class_distribution():
    """Bar chart: class distribution per domain."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, domain in zip(axes, DOMAINS):
        p = Path(f"data/splits/{domain}/train.csv")
        if not p.exists():
            continue
        df = pd.read_csv(p)
        counts = df["label"].value_counts().sort_index()
        bars = ax.bar(range(5), [counts.get(i, 0) for i in range(5)], color=PALETTE, edgecolor="white", linewidth=0.5)
        ax.set_title(DOMAIN_LABELS[domain], fontsize=14, fontweight="bold")
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"{i}" for i in range(5)], fontsize=10)
        ax.set_xlabel("Sentiment Class", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("Number of Samples", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 50, f"{int(h):,}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Training Set Class Distribution by Domain", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "class_distribution.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  class_distribution.png")


def plot_text_length():
    """Histogram: text length (word count) per domain."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, domain in zip(axes, DOMAINS):
        p = Path(f"data/splits/{domain}/train.csv")
        if not p.exists():
            continue
        df = pd.read_csv(p)
        wc = df["text"].str.split().str.len()
        ax.hist(wc.clip(upper=200), bins=50, color="#1976d2", alpha=0.85, edgecolor="white", linewidth=0.5)
        median = wc.median()
        ax.axvline(median, color="#d32f2f", linestyle="--", linewidth=2, label=f"Median: {median:.0f}")
        ax.set_title(DOMAIN_LABELS[domain], fontsize=14, fontweight="bold")
        ax.set_xlabel("Word Count", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("Frequency", fontsize=11)
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Text Length Distribution by Domain (Training Set)", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "text_length_distribution.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  text_length_distribution.png")


def plot_model_comparison():
    """Grouped bar chart: model accuracy & F1 comparison."""
    report_path = Path("reports/final_comparison.md")
    if not report_path.exists():
        print("  SKIP model_comparison.png (no report)")
        return

    content = report_path.read_text()
    models, accuracies, f1s = [], [], []
    for line in content.strip().split("\n"):
        if "|" in line and "Model" not in line and "---" not in line and "Generated" not in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 3:
                models.append(parts[0])
                accuracies.append(float(parts[1]))
                f1s.append(float(parts[2]))

    if not models:
        return

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy", color="#1976d2", edgecolor="white")
    bars2 = ax.bar(x + width/2, f1s, width, label="F1 Macro", color="#f57c00", edgecolor="white")

    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Model Performance Comparison", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  model_comparison.png")


def plot_confusion_matrices():
    """Confusion matrices for all available models."""
    # Load test data
    frames = []
    for domain in DOMAINS:
        p = Path(f"data/splits/{domain}/test.csv")
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        print("  SKIP confusion matrices (no test data)")
        return
    df = pd.concat(frames, ignore_index=True).dropna(subset=["text"])
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    model_configs = []

    svm_path = Path("models/tfidf_svm.joblib")
    if svm_path.exists():
        from src.models.svm_model import SVMClassifier
        model_configs.append(("SVM + TF-IDF", SVMClassifier.load(svm_path)))

    xgb_path = Path("models/tfidf_xgboost.joblib")
    if xgb_path.exists():
        from src.models.xgboost_model import XGBoostClassifier
        model_configs.append(("XGBoost + TF-IDF", XGBoostClassifier.load(xgb_path)))

    bert_path = Path("models/distilbert_finetuned")
    if bert_path.exists():
        from src.models.distilbert_classifier import DistilBERTClassifier
        model_configs.append(("DistilBERT Fine-tuned", DistilBERTClassifier.load(bert_path)))

    if not model_configs:
        print("  SKIP confusion matrices (no models)")
        return

    n = len(model_configs)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, model_configs):
        preds = model.predict(texts)
        cm = confusion_matrix(labels, preds)
        # Normalize by row
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            ax=ax, vmin=0, vmax=1, linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.tick_params(axis="both", labelsize=9)

    fig.suptitle("Normalized Confusion Matrices", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrices.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  confusion_matrices.png")


def plot_per_class_f1():
    """Per-class F1 heatmap across all models."""
    frames = []
    for domain in DOMAINS:
        p = Path(f"data/splits/{domain}/test.csv")
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True).dropna(subset=["text"])
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    results = {}

    svm_path = Path("models/tfidf_svm.joblib")
    if svm_path.exists():
        from src.models.svm_model import SVMClassifier
        preds = SVMClassifier.load(svm_path).predict(texts)
        report = classification_report(labels, preds, target_names=LABEL_NAMES, output_dict=True, zero_division=0)
        results["SVM + TF-IDF"] = [report[ln]["f1-score"] for ln in LABEL_NAMES]

    xgb_path = Path("models/tfidf_xgboost.joblib")
    if xgb_path.exists():
        from src.models.xgboost_model import XGBoostClassifier
        preds = XGBoostClassifier.load(xgb_path).predict(texts)
        report = classification_report(labels, preds, target_names=LABEL_NAMES, output_dict=True, zero_division=0)
        results["XGBoost + TF-IDF"] = [report[ln]["f1-score"] for ln in LABEL_NAMES]

    bert_path = Path("models/distilbert_finetuned")
    if bert_path.exists():
        from src.models.distilbert_classifier import DistilBERTClassifier
        preds = DistilBERTClassifier.load(bert_path).predict(texts)
        report = classification_report(labels, preds, target_names=LABEL_NAMES, output_dict=True, zero_division=0)
        results["DistilBERT"] = [report[ln]["f1-score"] for ln in LABEL_NAMES]

    if not results:
        return

    data = pd.DataFrame(results, index=LABEL_NAMES).T
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, vmin=0, vmax=1,
                linewidths=1, linecolor="white", cbar_kws={"shrink": 0.8})
    ax.set_title("Per-Class F1 Score by Model", fontsize=16, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "per_class_f1_heatmap.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  per_class_f1_heatmap.png")


def plot_pipeline_diagram():
    """Simple pipeline flow diagram."""
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis("off")

    phases = [
        ("Data\nAcquisition", "#1565c0"),
        ("NLP\nPreprocessing", "#2e7d32"),
        ("Embedding\nGeneration", "#f57c00"),
        ("Model\nTraining", "#d32f2f"),
        ("Interactive\nDemo", "#7b1fa2"),
    ]

    for i, (label, color) in enumerate(phases):
        x = 1.2 + i * 2.6
        rect = plt.Rectangle((x, 0.6), 2.0, 1.8, facecolor=color, edgecolor="white",
                              linewidth=2, alpha=0.9, zorder=2, transform=ax.transData)
        ax.add_patch(rect)
        ax.text(x + 1.0, 1.5, label, ha="center", va="center", fontsize=12,
                fontweight="bold", color="white", zorder=3)
        ax.text(x + 1.0, 0.25, f"Phase {i+1}", ha="center", va="center",
                fontsize=10, color="#555555")
        if i < len(phases) - 1:
            ax.annotate("", xy=(x + 2.2, 1.5), xytext=(x + 2.0, 1.5),
                        arrowprops=dict(arrowstyle="->", color="#333333", lw=2.5), zorder=1)

    fig.savefig(OUT / "pipeline_overview.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  pipeline_overview.png")


if __name__ == "__main__":
    print("Generating README assets...")
    plot_pipeline_diagram()
    plot_class_distribution()
    plot_text_length()
    plot_model_comparison()
    plot_confusion_matrices()
    plot_per_class_f1()
    print("Done!")
