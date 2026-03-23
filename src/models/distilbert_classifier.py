"""
Phase 4 — DistilBERT fine-tuned classifier (5-class).

Fine-tunes AutoModelForSequenceClassification on the combined dataset.

Interface
---------
    clf = DistilBERTClassifier()
    clf.fit(texts, labels, val_texts, val_labels)
    preds = clf.predict(texts)
    probs = clf.predict_proba(texts)
    clf.save(path)
    clf2 = DistilBERTClassifier.load(path)
"""

import json
import numpy as np
import torch
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class _TextDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class DistilBERTClassifier:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 5,
        max_length: int = 256,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForSequenceClassification | None = None

    def _init_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        self.model.to(self.device)

    def _encode(self, texts: list[str]):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def fit(
        self,
        texts: list[str],
        labels: list[int],
        val_texts: list[str] | None = None,
        val_labels: list[int] | None = None,
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
    ) -> "DistilBERTClassifier":
        self._init_model()
        self.model.train()

        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        dataset = _TextDataset(enc, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = len(loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            print(f"  Epoch {epoch+1}/{epochs} — loss: {total_loss/len(loader):.4f}")

        self.model.eval()
        return self

    def predict(self, texts, batch_size: int = 16) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call fit() or load() first.")
        if isinstance(texts, str):
            texts = [texts]

        all_preds: list[int] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        return np.array(all_preds)

    def predict_proba(self, texts, batch_size: int = 16) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call fit() or load() first.")
        if isinstance(texts, str):
            texts = [texts]

        all_probs: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)

    def save(self, path: str | Path) -> None:
        path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        config = {
            "num_labels": self.num_labels,
            "max_length": self.max_length,
        }
        with open(path / "classifier_config.json", "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path: str | Path) -> "DistilBERTClassifier":
        path = Path(path).resolve()
        config_file = path / "classifier_config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            obj = cls(
                model_name=str(path),
                num_labels=config.get("num_labels", 5),
                max_length=config.get("max_length", 256),
            )
        else:
            obj = cls(model_name=str(path))
        obj._init_model()
        return obj
