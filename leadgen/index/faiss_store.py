from __future__ import annotations

from typing import Tuple

import faiss  # type: ignore
import numpy as np


class FaissIPIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def add(self, X: np.ndarray) -> None:
        assert X.dtype == np.float32
        self.index.add(X)

    def topk(self, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        assert Q.dtype == np.float32
        scores, idx = self.index.search(Q, k)
        return scores, idx

