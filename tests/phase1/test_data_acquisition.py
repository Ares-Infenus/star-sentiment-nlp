import pytest
import pandas as pd
import os
from pathlib import Path

DATA_SPLITS_PATH = Path("data/splits")
DOMAINS = ["amazon_reviews", "tweets", "financial_news"]
SPLITS = ["train", "val", "test"]
REQUIRED_COLUMNS = ["text", "label", "domain"]
NUM_CLASSES = 5
MIN_SAMPLES_PER_CLASS = 100
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
TOLERANCE = 0.03  # 3% de tolerancia en splits


def load_split(domain: str, split: str) -> pd.DataFrame:
    path = DATA_SPLITS_PATH / domain / f"{split}.csv"
    assert path.exists(), f"❌ Archivo no encontrado: {path}"
    return pd.read_csv(path)


class TestDataFiles:
    def test_all_files_exist(self):
        """Verifica que todos los archivos de splits existan."""
        for domain in DOMAINS:
            for split in SPLITS:
                path = DATA_SPLITS_PATH / domain / f"{split}.csv"
                assert path.exists(), f"❌ Falta: {path}"

    def test_required_columns(self):
        """Verifica que todas las columnas requeridas estén presentes."""
        for domain in DOMAINS:
            for split in SPLITS:
                df = load_split(domain, split)
                for col in REQUIRED_COLUMNS:
                    assert col in df.columns, \
                        f"❌ Columna '{col}' falta en {domain}/{split}"


class TestDataQuality:
    def test_no_nulls_in_text(self):
        """Verifica que no haya textos nulos."""
        for domain in DOMAINS:
            for split in SPLITS:
                df = load_split(domain, split)
                null_count = df["text"].isnull().sum()
                assert null_count == 0, \
                    f"❌ {null_count} textos nulos en {domain}/{split}"

    def test_no_empty_strings(self):
        """Verifica que no haya textos vacíos."""
        for domain in DOMAINS:
            for split in SPLITS:
                df = load_split(domain, split)
                empty_count = (df["text"].str.strip() == "").sum()
                assert empty_count == 0, \
                    f"❌ {empty_count} textos vacíos en {domain}/{split}"

    def test_no_duplicates_in_train(self):
        """Verifica que no haya duplicados exactos en el set de entrenamiento."""
        for domain in DOMAINS:
            df = load_split(domain, "train")
            dup_count = df["text"].duplicated().sum()
            assert dup_count == 0, \
                f"❌ {dup_count} duplicados en train de {domain}"


class TestLabelSchema:
    def test_labels_are_0_to_4(self):
        """Verifica que todas las etiquetas estén en el rango [0, 4]."""
        for domain in DOMAINS:
            for split in SPLITS:
                df = load_split(domain, split)
                invalid = df[~df["label"].isin(range(NUM_CLASSES))]
                assert len(invalid) == 0, \
                    f"❌ Etiquetas inválidas en {domain}/{split}: {invalid['label'].unique()}"

    def test_all_classes_present_in_train(self):
        """Verifica que las 5 clases estén representadas en train."""
        for domain in DOMAINS:
            df = load_split(domain, "train")
            present = set(df["label"].unique())
            missing = set(range(NUM_CLASSES)) - present
            assert len(missing) == 0, \
                f"❌ Clases faltantes en train de {domain}: {missing}"

    def test_min_samples_per_class(self):
        """Verifica mínimo de muestras por clase en train."""
        for domain in DOMAINS:
            df = load_split(domain, "train")
            counts = df["label"].value_counts()
            for cls in range(NUM_CLASSES):
                count = counts.get(cls, 0)
                assert count >= MIN_SAMPLES_PER_CLASS, \
                    f"❌ Clase {cls} tiene solo {count} muestras en {domain}/train"


class TestSplitRatios:
    def test_split_proportions(self):
        """Verifica que los ratios de split sean correctos (±3%)."""
        for domain in DOMAINS:
            dfs = {s: load_split(domain, s) for s in SPLITS}
            total = sum(len(df) for df in dfs.values())
            for split, expected_ratio in SPLIT_RATIOS.items():
                actual_ratio = len(dfs[split]) / total
                assert abs(actual_ratio - expected_ratio) <= TOLERANCE, \
                    f"❌ Ratio de {split} en {domain}: esperado {expected_ratio:.0%}, " \
                    f"obtenido {actual_ratio:.2%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
