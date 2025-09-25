from __future__ import annotations

import json
from pathlib import Path
import argparse
import os

import joblib
import numpy as np
import pandas as pd

from leadgen.config import ARTIFACTS_DIR, DATA_DIR, TOPK_DEFAULT
from leadgen.features.normalize import normalize_email
from leadgen.embeddings.tabular_embedder import TabularEmbedder
from leadgen.embeddings.text_embedder import TextEmbedder
from leadgen.features.preprocess import preprocess_dataframe
from leadgen.index.faiss_store import FaissIPIndex
from leadgen.scoring.scorer import l2_normalize


def main() -> None:
    parser = argparse.ArgumentParser(description="Build embeddings and FAISS indices")
    parser.add_argument("--input-path", default=str(DATA_DIR / "crm.parquet"), help="Local path or s3:// URL to a parquet file")
    parser.add_argument("--text-cols", default=",".join(["job_title","bio"]), help="Comma-separated text columns")
    parser.add_argument("--cat-cols", default=",".join(["industry","country"]), help="Comma-separated categorical columns")
    parser.add_argument("--num-cols", default=",".join(["company_size","web_activity_score","email_engagement_score"]), help="Comma-separated numeric columns")
    args = parser.parse_args()

    input_path = os.environ.get("LEADGEN_INPUT_PATH", args.input_path)
    text_cols = os.environ.get("LEADGEN_TEXT_COLS", args.text_cols).split(",") if os.environ.get("LEADGEN_TEXT_COLS", args.text_cols) else []
    cat_cols = os.environ.get("LEADGEN_CAT_COLS", args.cat_cols).split(",") if os.environ.get("LEADGEN_CAT_COLS", args.cat_cols) else []
    num_cols = os.environ.get("LEADGEN_NUM_COLS", args.num_cols).split(",") if os.environ.get("LEADGEN_NUM_COLS", args.num_cols) else []
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "text_model").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "tabular").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "faiss").mkdir(parents=True, exist_ok=True)

    # Accept local or s3 path (requires s3fs installed)
    crm = pd.read_parquet(input_path)
    # Precompute normalized email set for duplicate checks at service time
    emails = set(crm.get("email", pd.Series([], dtype=str)).map(normalize_email).tolist())

    text_series, X_tab, encoders = preprocess_dataframe(crm, text_cols=text_cols, categorical_cols=cat_cols, numeric_cols=num_cols)

    text_model = TextEmbedder()
    E_text = text_model.encode(text_series.tolist())

    tabular = TabularEmbedder()
    tabular.fit(X_tab)
    E_tab = tabular.transform(X_tab)

    E = np.concatenate([E_text, E_tab], axis=1)
    E = l2_normalize(E)

    # Build indices
    dim = E.shape[1]
    idx_all = FaissIPIndex(dim)
    idx_all.add(E.astype(np.float32))

    high_mask = crm["is_high_value"].astype(bool).to_numpy()
    E_high = E[high_mask]
    idx_high = FaissIPIndex(dim)
    idx_high.add(E_high.astype(np.float32))

    # Save artifacts
    joblib.dump(tabular.scaler, ARTIFACTS_DIR / "tabular" / "scaler.pkl")
    joblib.dump(tabular.pca, ARTIFACTS_DIR / "tabular" / "pca.pkl")

    # Save FAISS indices
    import faiss  # type: ignore

    faiss.write_index(idx_all.index, str(ARTIFACTS_DIR / "faiss" / "all.index"))
    faiss.write_index(idx_high.index, str(ARTIFACTS_DIR / "faiss" / "high.index"))

    feature_meta = {
        "embedding_dim": int(dim),
        "encoders": {k: list(v.keys()) for k, v in encoders.items()},
        "topk": TOPK_DEFAULT,
        "has_email": "email" in crm.columns,
    }
    (ARTIFACTS_DIR / "feature_meta.json").write_text(json.dumps(feature_meta, indent=2))
    # Persist normalized email set
    (ARTIFACTS_DIR / "emails.txt").write_text("\n".join(sorted(emails)))


if __name__ == "__main__":
    main()

