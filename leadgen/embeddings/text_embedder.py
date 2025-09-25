from __future__ import annotations

from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer

from leadgen.config import TEXT_MODEL_NAME_PRIMARY, TEXT_MODEL_NAME_FALLBACK


class TextEmbedder:
    def __init__(self, model_name: str | None = None, hashing_dim: int = 384) -> None:
        name = model_name or TEXT_MODEL_NAME_PRIMARY
        self._fallback = False
        try:
            self.model = SentenceTransformer(name)
        except Exception:
            try:
                self.model = SentenceTransformer(TEXT_MODEL_NAME_FALLBACK)
            except Exception:
                # Offline fallback: hashing vectorizer
                self._fallback = True
                self.vectorizer = HashingVectorizer(n_features=hashing_dim, norm=None, alternate_sign=False)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = list(texts)
        if not self._fallback:
            embeddings = self.model.encode(texts_list, normalize_embeddings=True)
            return np.asarray(embeddings, dtype=np.float32)
        # Hashing fallback
        X = self.vectorizer.transform(texts_list)
        X = X.astype(np.float32)
        # Convert to dense; for small batches this is fine
        dense = X.toarray()
        # L2 normalize
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        dense = dense / norms
        return dense.astype(np.float32)

