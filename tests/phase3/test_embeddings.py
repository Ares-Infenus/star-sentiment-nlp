import pytest
import numpy as np
import time
from pathlib import Path
from src.embeddings.tfidf import TFIDFEmbedder
from src.embeddings.word2vec import Word2VecEmbedder
from src.embeddings.distilbert import DistilBERTEmbedder

SAMPLE_TEXTS = [
    "amazing product highly recommend",
    "terrible quality waste of money",
    "neutral average nothing special",
    "good value for the price",
    "absolutely love this purchase"
]
SAMPLE_LABELS = [4, 0, 2, 3, 4]

MAX_INFERENCE_MS_PER_SAMPLE = 500  # Umbral de tiempo de inferencia


class TestTFIDFEmbedder:
    @pytest.fixture
    def embedder(self):
        emb = TFIDFEmbedder(max_features=5000, ngram_range=(1, 2))
        emb.fit(SAMPLE_TEXTS)
        return emb

    def test_fit_transform_shape(self, embedder):
        vectors = embedder.transform(SAMPLE_TEXTS)
        assert vectors.shape[0] == len(SAMPLE_TEXTS), \
            "❌ Número de vectores no coincide con muestras"
        assert vectors.shape[1] > 0, "❌ Dimensión de embedding es 0"

    def test_no_nan_values(self, embedder):
        vectors = embedder.transform(SAMPLE_TEXTS)
        dense = vectors.toarray() if hasattr(vectors, "toarray") else vectors
        assert not np.isnan(dense).any(), "❌ NaN encontrado en embeddings TF-IDF"

    def test_inference_time(self, embedder):
        start = time.time()
        embedder.transform(SAMPLE_TEXTS)
        elapsed_ms = (time.time() - start) * 1000 / len(SAMPLE_TEXTS)
        assert elapsed_ms < MAX_INFERENCE_MS_PER_SAMPLE, \
            f"❌ TF-IDF demasiado lento: {elapsed_ms:.1f}ms/muestra"

    def test_model_can_be_saved_and_loaded(self, embedder, tmp_path):
        path = tmp_path / "tfidf.joblib"
        embedder.save(path)
        assert path.exists(), "❌ Modelo TF-IDF no se guardó"
        loaded = TFIDFEmbedder.load(path)
        v1 = embedder.transform(SAMPLE_TEXTS).toarray()
        v2 = loaded.transform(SAMPLE_TEXTS).toarray()
        np.testing.assert_array_almost_equal(v1, v2,
            err_msg="❌ Vectores difieren tras guardar/cargar")


class TestWord2VecEmbedder:
    @pytest.fixture
    def embedder(self):
        emb = Word2VecEmbedder(vector_size=300, window=5, min_count=1)
        emb.fit(SAMPLE_TEXTS)
        return emb

    def test_embedding_dimension(self, embedder):
        vectors = embedder.transform(SAMPLE_TEXTS)
        assert vectors.shape == (len(SAMPLE_TEXTS), 300), \
            f"❌ Dimensión incorrecta: {vectors.shape}, esperado ({len(SAMPLE_TEXTS)}, 300)"

    def test_no_nan_values(self, embedder):
        vectors = embedder.transform(SAMPLE_TEXTS)
        assert not np.isnan(vectors).any(), "❌ NaN en embeddings Word2Vec"

    def test_inference_time(self, embedder):
        start = time.time()
        embedder.transform(SAMPLE_TEXTS)
        elapsed_ms = (time.time() - start) * 1000 / len(SAMPLE_TEXTS)
        assert elapsed_ms < MAX_INFERENCE_MS_PER_SAMPLE, \
            f"❌ Word2Vec demasiado lento: {elapsed_ms:.1f}ms/muestra"


class TestDistilBERTEmbedder:
    @pytest.fixture
    def embedder(self):
        return DistilBERTEmbedder(model_name="distilbert-base-uncased")

    def test_embedding_dimension(self, embedder):
        vectors = embedder.transform(SAMPLE_TEXTS[:2])  # Solo 2 para rapidez
        assert vectors.shape == (2, 768), \
            f"❌ Dimensión incorrecta: {vectors.shape}, esperado (2, 768)"

    def test_no_nan_values(self, embedder):
        vectors = embedder.transform(SAMPLE_TEXTS[:2])
        assert not np.isnan(vectors).any(), "❌ NaN en embeddings DistilBERT"

    def test_different_texts_different_vectors(self, embedder):
        v1 = embedder.transform([SAMPLE_TEXTS[0]])
        v2 = embedder.transform([SAMPLE_TEXTS[1]])
        assert not np.allclose(v1, v2), \
            "❌ Textos diferentes producen vectores idénticos"


class TestEmbeddingComparison:
    def test_all_embeddings_produce_valid_shapes(self):
        """Test de integración: los 3 embeddings funcionan con los mismos textos."""
        tfidf = TFIDFEmbedder(max_features=1000)
        tfidf.fit(SAMPLE_TEXTS)
        v_tfidf = tfidf.transform(SAMPLE_TEXTS)

        w2v = Word2VecEmbedder(vector_size=100, min_count=1)
        w2v.fit(SAMPLE_TEXTS)
        v_w2v = w2v.transform(SAMPLE_TEXTS)

        bert = DistilBERTEmbedder()
        v_bert = bert.transform(SAMPLE_TEXTS)

        assert v_tfidf.shape[0] == len(SAMPLE_TEXTS)
        assert v_w2v.shape[0] == len(SAMPLE_TEXTS)
        assert v_bert.shape[0] == len(SAMPLE_TEXTS)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
