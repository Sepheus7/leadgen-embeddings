from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


TEXT_COLS = ["job_title", "bio"]
CATEGORICAL_COLS = ["industry", "country"]
NUMERIC_COLS = ["company_size", "web_activity_score", "email_engagement_score"]


def make_text_blob(job_title: str, bio: str) -> str:
    job_title = job_title or ""
    bio = bio or ""
    text = f"{job_title}. {bio}".strip()
    return text


def frequency_encode(series: pd.Series) -> Dict[str, float]:
    counts = Counter(series.dropna().astype(str).tolist())
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def apply_frequency_encoding(series: pd.Series, mapping: Dict[str, float]) -> np.ndarray:
    return series.fillna("").astype(str).map(mapping).fillna(0.0).to_numpy(dtype=np.float32)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
) -> Tuple[pd.Series, np.ndarray, Dict[str, Dict[str, float]]]:
    text_cols = text_cols or TEXT_COLS
    categorical_cols = categorical_cols or CATEGORICAL_COLS
    numeric_cols = numeric_cols or NUMERIC_COLS

    # Text blob from arbitrary columns (fallback to job_title+bio style)
    if set(["job_title", "bio"]).issubset(text_cols):
        text_series = df.apply(lambda r: make_text_blob(r.get("job_title", ""), r.get("bio", "")), axis=1)
    else:
        def _join_text(row: pd.Series) -> str:
            parts = [str(row.get(c, "")) for c in text_cols]
            return ". ".join([p for p in parts if p]).strip()
        text_series = df.apply(_join_text, axis=1)

    encoders: Dict[str, Dict[str, float]] = {}
    features: List[np.ndarray] = []
    for col in categorical_cols:
        series = df[col] if col in df.columns else pd.Series([""] * len(df))
        mapping = frequency_encode(series)
        encoders[col] = mapping
        features.append(apply_frequency_encoding(series, mapping))

    if len(numeric_cols) > 0:
        if set(numeric_cols).issubset(df.columns):
            numeric = df[numeric_cols].astype(np.float32).to_numpy(copy=False)
        else:
            numeric = np.zeros((len(df), len(numeric_cols)), dtype=np.float32)
        X = np.column_stack(features + [numeric]).astype(np.float32) if features else numeric.astype(np.float32)
    else:
        X = np.column_stack(features).astype(np.float32) if features else np.zeros((len(df), 0), dtype=np.float32)

    return text_series, X, encoders

