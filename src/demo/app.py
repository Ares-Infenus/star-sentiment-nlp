"""
Phase 5 -- Gradio demo.

Public API (used by tests)
--------------------------
    models = load_models()           -> dict of loaded classifiers
    label, conf = predict_sentiment(text)

Run locally
-----------
    python src/demo/app.py
"""

from pathlib import Path

import gradio as gr
import numpy as np

MODELS_PATH = Path("models")

LABEL_NAMES = {
    0: "⭐ Very Negative",
    1: "⭐⭐ Negative",
    2: "⭐⭐⭐ Neutral",
    3: "⭐⭐⭐⭐ Positive",
    4: "⭐⭐⭐⭐⭐ Very Positive",
}

_models: dict = {}


def load_models() -> dict:
    """Load all available trained models from models/."""
    global _models
    if _models:
        return _models

    from src.models.svm_model import SVMClassifier
    from src.models.xgboost_model import XGBoostClassifier
    from src.models.distilbert_classifier import DistilBERTClassifier

    svm_path = MODELS_PATH / "tfidf_svm.joblib"
    xgb_path = MODELS_PATH / "tfidf_xgboost.joblib"
    bert_path = MODELS_PATH / "distilbert_finetuned"

    if svm_path.exists():
        print("Loading SVM...")
        _models["svm"] = SVMClassifier.load(svm_path)

    if xgb_path.exists():
        print("Loading XGBoost...")
        _models["xgboost"] = XGBoostClassifier.load(xgb_path)

    if bert_path.exists():
        print("Loading DistilBERT...")
        _models["distilbert"] = DistilBERTClassifier.load(bert_path)

    if not _models:
        print("[WARN] No trained models found in models/. Run scripts/run_phase4.py first.")

    return _models


def predict_sentiment(text: str, model_name: str | None = None) -> tuple[int, float]:
    """
    Returns (label: int 0-4, confidence: float 0-1).
    Falls back to (2, 0.0) if no model is available or text is empty.
    """
    if not text or not text.strip():
        return (2, 0.0)

    models = load_models()
    if not models:
        return (2, 0.0)

    # Select model
    if model_name and model_name in models:
        model = models[model_name]
    elif "distilbert" in models:
        model = models["distilbert"]
    elif "xgboost" in models:
        model = models["xgboost"]
    else:
        model = next(iter(models.values()))

    # Truncate very long texts
    words = text.split()
    if len(words) > 400:
        text = " ".join(words[:400])

    label = int(model.predict([text])[0])
    proba = model.predict_proba([text])[0]
    confidence = float(proba[label])

    return (label, confidence)


# ─── Gradio UI ───────────────────────────────────────────────────────────────

def _gradio_predict(text: str, model_choice: str):
    label, confidence = predict_sentiment(text, model_choice)
    return LABEL_NAMES[label], f"{confidence:.1%}", label


def create_app() -> gr.Blocks:
    load_models()
    model_choices = list(_models.keys()) if _models else ["(no models loaded)"]

    with gr.Blocks(title="Sentiment Classifier -- 5-Star Scale") as demo:
        gr.Markdown(
            "# 🌟 Sentiment Classifier\n"
            "Classify any text on a 1-to-5 star sentiment scale."
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Input text",
                    lines=5,
                    placeholder="Enter a product review, tweet, or financial headline...",
                )
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[0],
                    label="Model",
                )
                btn = gr.Button("Analyze", variant="primary")

            with gr.Column(scale=1):
                label_out = gr.Textbox(label="Sentiment")
                conf_out = gr.Textbox(label="Confidence")
                score_slider = gr.Slider(
                    minimum=0, maximum=4, step=1, label="Score (0 = Very Neg, 4 = Very Pos)"
                )

        btn.click(
            _gradio_predict,
            inputs=[text_input, model_dropdown],
            outputs=[label_out, conf_out, score_slider],
        )

        gr.Examples(
            examples=[
                ["This product is absolutely incredible, best I ever used!", model_choices[0]],
                ["Terrible quality, broke after one day, total waste of money.", model_choices[0]],
                ["It is okay, nothing special, average product.", model_choices[0]],
                ["Stocks rose sharply after better-than-expected earnings.", model_choices[0]],
            ],
            inputs=[text_input, model_dropdown],
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False, server_port=7860)
