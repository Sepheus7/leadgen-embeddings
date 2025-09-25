from __future__ import annotations

import numpy as np

from leadgen.index.faiss_store import FaissIPIndex
from leadgen.scoring.scorer import l2_normalize, score_lead


def test_score_lead_math():
    # Two clusters in 2D, unit normalized
    base = np.array([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.98, 0.02],
        [0.0, 1.0],
        [0.01, 0.99],
    ], dtype=np.float32)
    base = l2_normalize(base)

    idx_all = FaissIPIndex(2)
    idx_all.add(base)

    idx_high = FaissIPIndex(2)
    idx_high.add(base[:3])  # first cluster high-value

    q = l2_normalize(np.array([[1.0, 0.0]], dtype=np.float32))
    scores = score_lead(q, idx_all, idx_high, k=3)

    assert scores["S_look"] > 0.95
    assert 0.0 <= scores["S_novel"] <= 1.0
    assert isinstance(scores["contrast"], float)

