"""
Phase 3 — DistilBERT Embedder (CLS token pooling).

Interface
---------
    emb = DistilBERTEmbedder()
    vectors = emb.transform(texts)   # np.ndarray (n, 768)
"""

import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel


class DistilBERTEmbedder:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def transform(self, texts: list[str]) -> np.ndarray:
        """
        Returns CLS-token embeddings: ndarray of shape (n_samples, 768).
        Processes in batches to avoid OOM on long lists.
        """
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                output = self.model(**encoded)

            # CLS token is the first token of last_hidden_state
            cls_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

        return np.vstack(all_embeddings)
