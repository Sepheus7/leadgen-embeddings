from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

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


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.Series, np.ndarray, Dict[str, Dict[str, float]]]:
    text_series = df.apply(lambda r: make_text_blob(r.get("job_title", ""), r.get("bio", "")), axis=1)

    encoders: Dict[str, Dict[str, float]] = {}
    features: List[np.ndarray] = []
    for col in CATEGORICAL_COLS:
        mapping = frequency_encode(df[col] if col in df.columns else pd.Series([""] * len(df)))
        encoders[col] = mapping
        features.append(apply_frequency_encoding(df[col] if col in df.columns else pd.Series([""] * len(df)), mapping))

    numeric = df[NUMERIC_COLS].astype(np.float32).to_numpy(copy=False) if set(NUMERIC_COLS).issubset(df.columns) else np.zeros((len(df), len(NUMERIC_COLS)), dtype=np.float32)

    X = np.column_stack(features + [numeric]).astype(np.float32)
    return text_series, X, encoders

