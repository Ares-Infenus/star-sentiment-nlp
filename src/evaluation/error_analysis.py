"""Error analysis utilities."""

import pandas as pd
import numpy as np

LABEL_NAMES = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]


def find_errors(
    texts: list[str], y_true, y_pred, top_n: int = 20
) -> pd.DataFrame:
    """Return the top-N highest-severity mispredictions."""
    df = pd.DataFrame({
        "text": texts,
        "true": y_true,
        "pred": y_pred,
        "true_label": [LABEL_NAMES[t] for t in y_true],
        "pred_label": [LABEL_NAMES[p] for p in y_pred],
    })
    df["error_magnitude"] = (df["true"] - df["pred"]).abs()
    errors = df[df["true"] != df["pred"]].sort_values(
        "error_magnitude", ascending=False
    )
    return errors.head(top_n).reset_index(drop=True)


def error_summary(y_true, y_pred) -> dict:
    df = pd.DataFrame({"true": list(y_true), "pred": list(y_pred)})
    df["correct"] = df["true"] == df["pred"]
    df["within_1"] = (df["true"] - df["pred"]).abs() <= 1
    top_confusions = (
        df[~df["correct"]]
        .groupby(["true", "pred"])
        .size()
        .sort_values(ascending=False)
        .head(10)
        .to_dict()
    )
    return {
        "accuracy": df["correct"].mean(),
        "within_1_accuracy": df["within_1"].mean(),
        "top_confusions": top_confusions,
    }
