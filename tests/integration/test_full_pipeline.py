import pytest
import pandas as pd
import numpy as np
import time
from pathlib import Path

GROUND_TRUTH_SAMPLES = [
    # (text, true_label, description)
    ("horrible product complete garbage broke immediately total waste", 0, "clear_very_negative"),
    ("worst purchase ever made absolutely terrible quality avoid", 0, "clear_very_negative_2"),
    ("not satisfied with this product had several issues disappointed", 1, "clear_negative"),
    ("below average quality not worth the price had problems", 1, "clear_negative_2"),
    ("average product nothing special does the job nothing more", 2, "clear_neutral"),
    ("okay quality mediocre performance meets basic expectations", 2, "clear_neutral_2"),
    ("good product works well happy with purchase recommend", 3, "clear_positive"),
    ("solid quality good value satisfied with the results", 3, "clear_positive_2"),
    ("absolutely incredible outstanding best product ever purchased love it", 4, "clear_very_positive"),
    ("perfect exceptional quality exceeded all expectations highly recommend", 4, "clear_very_positive_2"),
]

PHASE_CHECKLIST = {
    "phase1_data": Path("data/splits/amazon_reviews/train.csv"),
    "phase2_preprocessing": Path("data/processed/amazon_reviews_processed.csv"),
    "phase3_embeddings": Path("models/tfidf_svm.joblib"),
    "phase4_models_svm": Path("models/tfidf_svm.joblib"),
    "phase4_models_xgboost": Path("models/tfidf_xgboost.joblib"),
    "phase5_demo": Path("src/demo/app.py"),
}


class TestPhaseCompletion:
    def test_all_phases_have_artifacts(self):
        missing = []
        for phase, path in PHASE_CHECKLIST.items():
            if not path.exists():
                missing.append(f"{phase}: {path}")
        assert len(missing) == 0, \
            "❌ Artefactos faltantes:\n" + "\n".join(missing)


class TestEndToEndPipeline:
    def test_pipeline_runs_without_errors(self):
        from src.preprocessing.cleaner import full_preprocess
        from src.demo.app import predict_sentiment

        for text, true_label, desc in GROUND_TRUTH_SAMPLES:
            try:
                processed = full_preprocess(text)
                assert len(processed) > 0, f"❌ Texto vacío tras preprocesar: {desc}"
                label, conf = predict_sentiment(text)
                assert 0 <= label <= 4
                assert 0.0 <= conf <= 1.0
            except Exception as e:
                pytest.fail(f"❌ Pipeline falló en '{desc}': {e}")

    def test_extreme_sentiment_discrimination(self):
        from src.demo.app import predict_sentiment

        very_neg_texts = [t for t, l, _ in GROUND_TRUTH_SAMPLES if l == 0]
        very_pos_texts = [t for t, l, _ in GROUND_TRUTH_SAMPLES if l == 4]

        neg_preds = [predict_sentiment(t)[0] for t in very_neg_texts]
        pos_preds = [predict_sentiment(t)[0] for t in very_pos_texts]

        avg_neg = np.mean(neg_preds)
        avg_pos = np.mean(pos_preds)

        assert avg_neg < avg_pos, \
            f"❌ CRÍTICO: Modelo no discrimina extremos. " \
            f"avg_neg={avg_neg:.2f}, avg_pos={avg_pos:.2f}"

    def test_directional_accuracy_on_ground_truth(self):
        from src.demo.app import predict_sentiment

        correct_within_1 = 0
        for text, true_label, desc in GROUND_TRUTH_SAMPLES:
            pred_label, _ = predict_sentiment(text)
            if abs(pred_label - true_label) <= 1:
                correct_within_1 += 1

        accuracy_within_1 = correct_within_1 / len(GROUND_TRUTH_SAMPLES)
        assert accuracy_within_1 >= 0.70, \
            f"❌ Solo {accuracy_within_1:.0%} dentro de ±1 clase. Mínimo: 70%"


class TestDataIntegrity:
    def test_no_data_leakage(self):
        for domain in ["amazon_reviews", "tweets", "financial_news"]:
            train_path = Path(f"data/splits/{domain}/train.csv")
            test_path = Path(f"data/splits/{domain}/test.csv")

            if not train_path.exists() or not test_path.exists():
                pytest.skip(f"Archivos de {domain} no disponibles")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_texts = set(train_df["text"].str.strip())
            test_texts = set(test_df["text"].str.strip())

            overlap = train_texts & test_texts
            assert len(overlap) == 0, \
                f"❌ Data leakage en {domain}: {len(overlap)} textos en train y test"

    def test_label_consistency_across_phases(self):
        for domain in ["amazon_reviews"]:
            raw_path = Path(f"data/splits/{domain}/train.csv")
            processed_path = Path(f"data/processed/{domain}_processed.csv")

            if not raw_path.exists() or not processed_path.exists():
                pytest.skip("Archivos no disponibles")

            raw_df = pd.read_csv(raw_path)
            proc_df = pd.read_csv(processed_path)

            merged = raw_df.merge(proc_df, on="label", how="inner")
            assert len(merged) > 0, \
                f"❌ No se pudo hacer merge por label en {domain}"


class TestPerformanceSummary:
    def test_generate_comparison_report(self):
        report_path = Path("reports/final_comparison.md")
        assert report_path.exists(), \
            "❌ Reporte final de comparación no encontrado"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short",
                 "--cov=src", "--cov-report=term-missing"])
