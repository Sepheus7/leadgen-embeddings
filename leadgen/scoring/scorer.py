from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def cosine_mean_topk(query: np.ndarray, index, k: int = 20) -> float:
    sims, _ = index.topk(query.astype(np.float32), k)
    return float(np.mean(sims[0])) if sims.size else 0.0


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (X / norms).astype(np.float32)


def score_lead(lead_emb: np.ndarray, idx_all, idx_high, k: int = 20) -> Dict[str, float]:
    s_all, nn_all = idx_all.topk(lead_emb.astype(np.float32), k)
    s_high, nn_high = idx_high.topk(lead_emb.astype(np.float32), k)

    s_look = float(np.mean(s_high[0])) if s_high.size else 0.0
    s_novel = 1.0 - float(np.mean(s_all[0])) if s_all.size else 1.0
    contrast = s_look - (1.0 - s_novel)
    return {
        "S_look": s_look,
        "S_novel": s_novel,
        "contrast": contrast,
        "nn_all_ids": nn_all[0].tolist() if nn_all.size else [],
        "nn_high_ids": nn_high[0].tolist() if nn_high.size else [],
    }

