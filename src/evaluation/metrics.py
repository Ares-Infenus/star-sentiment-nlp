"""Evaluation utilities — metrics and confusion matrix plots."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

LABEL_NAMES = [
    "Very Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Very Positive",
]


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "report": classification_report(
            y_true, y_pred, target_names=LABEL_NAMES, zero_division=0
        ),
    }


def print_metrics(metrics: dict, model_name: str = "") -> None:
    prefix = f"[{model_name}] " if model_name else ""
    print(f"{prefix}Accuracy  : {metrics['accuracy']:.4f}")
    print(f"{prefix}F1 Macro  : {metrics['f1_macro']:.4f}")
    print(f"{prefix}F1 Weighted: {metrics['f1_weighted']:.4f}")
    print(metrics["report"])


def plot_confusion_matrix(
    y_true,
    y_pred,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
