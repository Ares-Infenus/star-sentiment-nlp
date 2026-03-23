import pytest
import time
from src.demo.app import predict_sentiment, load_models

MAX_DEMO_RESPONSE_MS = 3000  # 3 segundos máx para la demo

EXPERIMENTAL_CASES = [
    # (texto, clase_esperada_min, clase_esperada_max, descripcion)
    ("This product is absolutely incredible, best I ever used!", 3, 4, "Very positive"),
    ("Terrible quality, broke after one day, waste of money", 0, 1, "Very negative"),
    ("It is okay, not great not terrible, average product", 1, 3, "Neutral range"),
    ("Pretty good overall, minor issues but satisfied", 2, 4, "Positive range"),
    ("Not great, had some issues but manageable", 0, 3, "Negative-neutral range"),
]


class TestDemoLoads:
    def test_models_load_without_error(self):
        """Verifica que los modelos cargan correctamente."""
        models = load_models()
        assert models is not None, "❌ load_models() retornó None"
        assert len(models) > 0, "❌ No se cargó ningún modelo"


class TestPredictFunction:
    def test_returns_tuple(self):
        result = predict_sentiment("Great product!")
        assert isinstance(result, tuple), \
            f"❌ Retornó {type(result)}, esperado tuple"
        assert len(result) == 2, \
            f"❌ Tuple tiene {len(result)} elementos, esperado 2"

    def test_label_in_valid_range(self):
        for text, _, _, desc in EXPERIMENTAL_CASES:
            label, _ = predict_sentiment(text)
            assert 0 <= label <= 4, \
                f"❌ Label {label} fuera de rango para: '{desc}'"

    def test_confidence_is_probability(self):
        label, confidence = predict_sentiment("Good product")
        assert 0.0 <= confidence <= 1.0, \
            f"❌ Confianza {confidence} fuera de [0,1]"

    def test_experimental_cases_within_expected_range(self):
        for text, min_cls, max_cls, desc in EXPERIMENTAL_CASES:
            label, _ = predict_sentiment(text)
            assert min_cls <= label <= max_cls, \
                f"❌ '{desc}': predicho={label}, esperado en [{min_cls},{max_cls}]"

    def test_empty_text_handled(self):
        try:
            label, conf = predict_sentiment("")
            assert label is not None
        except Exception as e:
            pytest.fail(f"❌ Crash con texto vacío: {e}")

    def test_very_long_text_handled(self):
        long_text = "great product " * 200
        try:
            label, conf = predict_sentiment(long_text)
            assert 0 <= label <= 4
        except Exception as e:
            pytest.fail(f"❌ Crash con texto largo: {e}")


class TestResponseTime:
    def test_prediction_response_time(self):
        texts = [t for t, _, _, _ in EXPERIMENTAL_CASES]
        start = time.time()
        for text in texts:
            predict_sentiment(text)
        elapsed_ms = (time.time() - start) * 1000 / len(texts)
        assert elapsed_ms < MAX_DEMO_RESPONSE_MS, \
            f"❌ Demo demasiado lenta: {elapsed_ms:.0f}ms/predicción"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
