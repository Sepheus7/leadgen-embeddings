from __future__ import annotations

import pandas as pd

from leadgen.features.preprocess import make_text_blob, preprocess_dataframe


def test_make_text_blob_nonempty():
    blob = make_text_blob("Portfolio Manager", "builds trading strategies")
    assert isinstance(blob, str) and len(blob) > 0


def test_frequency_encoder_keys_present():
    df = pd.DataFrame({
        "industry": ["Finance", "Finance", "SaaS"],
        "country": ["US", "US", "UK"],
        "job_title": ["x", "y", "z"],
        "bio": ["a", "b", "c"],
        "company_size": [10, 20, 30],
        "web_activity_score": [0.1, 0.2, 0.3],
        "email_engagement_score": [0.4, 0.5, 0.6],
    })
    _, _, encoders = preprocess_dataframe(df)
    assert "industry" in encoders and "country" in encoders
    assert set(encoders["industry"].keys()) >= {"Finance", "SaaS"}
    assert set(encoders["country"].keys()) >= {"US", "UK"}

