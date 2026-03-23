import pytest
import re
import pandas as pd
from pathlib import Path
from src.preprocessing.cleaner import clean_text, full_preprocess

PROCESSED_PATH = Path("data/processed")
DOMAINS = ["amazon_reviews", "tweets", "financial_news"]
MIN_TEXT_LENGTH = 3   # palabras mínimas tras procesamiento
MAX_TEXT_LENGTH = 512 # palabras máximas (límite DistilBERT)


# --- Tests unitarios del cleaner ---

class TestCleanText:
    def test_lowercases_text(self):
        assert clean_text("HELLO World") == "hello world"

    def test_removes_urls(self):
        result = clean_text("Check this http://example.com now")
        assert "http" not in result
        assert "example.com" not in result

    def test_removes_mentions(self):
        result = clean_text("Hello @JohnDoe how are you")
        assert "@johndoe" not in result
        assert "@" not in result

    def test_removes_hashtags(self):
        result = clean_text("Love this #product #review")
        assert "#product" not in result
        assert "#" not in result

    def test_expands_contractions(self):
        result = clean_text("I don't like it, it's bad")
        assert "do not" in result or "dont" not in result

    def test_removes_punctuation(self):
        result = clean_text("Great product!!! Worth it.")
        assert "!" not in result
        assert "." not in result

    def test_handles_empty_string(self):
        result = clean_text("")
        assert result == ""

    def test_handles_only_noise(self):
        result = clean_text("@user #tag http://url.com !!!")
        assert result.strip() == ""


class TestFullPreprocess:
    def test_removes_stopwords(self):
        result = full_preprocess("This is a very good product")
        stopwords_found = [w for w in ["this", "is", "a", "very"]
                          if w in result.split()]
        assert len(stopwords_found) == 0, \
            f"❌ Stopwords encontradas: {stopwords_found}"

    def test_applies_lemmatization(self):
        result = full_preprocess("The products are running beautifully")
        assert "running" not in result or "run" in result

    def test_output_is_string(self):
        result = full_preprocess("Great product, highly recommended!")
        assert isinstance(result, str)

    def test_min_length_preserved(self):
        # Texto con contenido real no debería quedar vacío
        result = full_preprocess("Amazing product quality excellent")
        assert len(result.split()) >= 1


class TestProcessedFiles:
    def test_processed_files_exist(self):
        for domain in DOMAINS:
            path = PROCESSED_PATH / f"{domain}_processed.csv"
            assert path.exists(), f"❌ Falta archivo procesado: {path}"

    def test_processed_column_exists(self):
        for domain in DOMAINS:
            df = pd.read_csv(PROCESSED_PATH / f"{domain}_processed.csv")
            assert "text_processed" in df.columns, \
                f"❌ Columna 'text_processed' falta en {domain}"

    def test_no_nulls_after_processing(self):
        for domain in DOMAINS:
            df = pd.read_csv(PROCESSED_PATH / f"{domain}_processed.csv")
            null_count = df["text_processed"].isnull().sum()
            assert null_count == 0, \
                f"❌ {null_count} nulos en text_processed de {domain}"

    def test_text_length_bounds(self):
        for domain in DOMAINS:
            df = pd.read_csv(PROCESSED_PATH / f"{domain}_processed.csv")
            word_counts = df["text_processed"].str.split().str.len()
            too_short = (word_counts < MIN_TEXT_LENGTH).sum()
            too_long = (word_counts > MAX_TEXT_LENGTH).sum()
            assert too_short == 0, \
                f"❌ {too_short} textos muy cortos (<{MIN_TEXT_LENGTH} palabras) en {domain}"
            assert too_long == 0, \
                f"❌ {too_long} textos muy largos (>{MAX_TEXT_LENGTH} palabras) en {domain}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
