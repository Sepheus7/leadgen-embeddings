from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from leadgen.config import ARTIFACTS_DIR
from leadgen.features.normalize import normalize_email
from leadgen.embeddings.tabular_embedder import TabularEmbedder
from leadgen.embeddings.text_embedder import TextEmbedder
from leadgen.features.preprocess import preprocess_dataframe
from leadgen.index.faiss_store import FaissIPIndex
from leadgen.scoring.scorer import l2_normalize, score_lead


class Components:
    def __init__(self, text_model: TextEmbedder, tabular: TabularEmbedder, idx_all: FaissIPIndex, idx_high: FaissIPIndex, feature_meta: Dict) -> None:
        self.text_model = text_model
        self.tabular = tabular
        self.idx_all = idx_all
        self.idx_high = idx_high
        self.feature_meta = feature_meta


def load_components() -> Components:
    text_model = TextEmbedder()

    tabular = TabularEmbedder()
    scaler = joblib.load(ARTIFACTS_DIR / "tabular" / "scaler.pkl")
    pca = joblib.load(ARTIFACTS_DIR / "tabular" / "pca.pkl")
    tabular.scaler = scaler
    tabular.pca = pca

    feature_meta = json.loads((ARTIFACTS_DIR / "feature_meta.json").read_text())

    faiss_dir = ARTIFACTS_DIR / "faiss"
    idx_all = FaissIPIndex(feature_meta["embedding_dim"])  # placeholder; will be overwritten below
    idx_high = FaissIPIndex(feature_meta["embedding_dim"])  # placeholder

    # Load FAISS from file
    import faiss  # type: ignore

    idx_all.index = faiss.read_index(str(faiss_dir / "all.index"))
    idx_high.index = faiss.read_index(str(faiss_dir / "high.index"))
    idx_all.dim = idx_all.index.d
    idx_high.dim = idx_high.index.d

    return Components(text_model, tabular, idx_all, idx_high, feature_meta)


def embed_one(lead: Dict, components: Components) -> np.ndarray:
    df = pd.DataFrame([lead])
    text_series, X_tab, _ = preprocess_dataframe(df)
    E_text = components.text_model.encode(text_series.tolist())
    E_tab = components.tabular.transform(X_tab)
    E = np.concatenate([E_text, E_tab], axis=1)
    E = l2_normalize(E)
    return E.astype(np.float32)


def score_one(emb: np.ndarray, components: Components) -> Dict:
    return score_lead(emb, components.idx_all, components.idx_high)


def is_duplicate_email(lead: Dict, crm_emails: set[str]) -> bool:
    email = normalize_email(lead.get("email"))
    if not email:
        return False
    return email in crm_emails

