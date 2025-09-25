from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


INDUSTRIES = [
    ("Finance", 0.22),
    ("WealthTech", 0.08),
    ("SaaS", 0.18),
    ("Healthcare", 0.15),
    ("Ecommerce", 0.12),
    ("Manufacturing", 0.1),
    ("Energy", 0.07),
    ("Education", 0.08),
]

COUNTRIES = [
    ("US", 0.45),
    ("UK", 0.15),
    ("DE", 0.1),
    ("FR", 0.08),
    ("CA", 0.07),
    ("IN", 0.1),
    ("SG", 0.05),
]

TITLES = [
    "Portfolio Manager",
    "Quant Researcher",
    "DevOps Engineer",
    "Cloud Architect",
    "Data Scientist",
    "Head of Compliance",
    "Regulatory Affairs Manager",
    "Ecommerce Growth Lead",
    "Manufacturing Operations Manager",
    "Energy Trading Analyst",
]

BIO_FRAGMENTS = {
    "fin": [
        "focuses on portfolio optimization",
        "builds trading strategies",
        "works with allocator relationships",
        "manages risk and execution",
    ],
    "cloud": [
        "deploys cloud infrastructure",
        "implements CI/CD pipelines",
        "optimizes Kubernetes clusters",
        "improves developer productivity",
    ],
    "health": [
        "works on regulatory reporting",
        "builds data pipelines for EHR",
        "ensures HIPAA compliance",
        "models patient outcomes",
    ],
}


KEYWORDS = ["portfolio", "allocator", "trading", "execution", "Kubernetes", "compliance", "regulatory"]


def sample_from_weighted(items: List[Tuple[str, float]], n: int) -> List[str]:
    labels, weights = zip(*items)
    probs = np.array(weights) / np.sum(weights)
    idx = np.random.choice(len(labels), size=n, p=probs)
    return [labels[i] for i in idx]


def synthesize(n_customers: int = 20_000, n_leads: int = 2_000) -> None:
    # CRM dataset
    industries = sample_from_weighted(INDUSTRIES, n_customers)
    countries = sample_from_weighted(COUNTRIES, n_customers)
    company_sizes = np.random.lognormal(mean=5.5, sigma=0.8, size=n_customers).astype(int)
    web_scores = np.clip(np.random.beta(2, 5, size=n_customers) * 1.2, 0, 1)
    email_scores = np.clip(np.random.beta(2, 4, size=n_customers) * 1.3, 0, 1)

    titles = np.random.choice(TITLES, size=n_customers)

    bios = []
    for t in titles:
        if "Portfolio" in t or "Quant" in t or "Energy" in t:
            domain = "fin"
        elif "DevOps" in t or "Cloud" in t:
            domain = "cloud"
        elif "Compliance" in t or "Regulatory" in t:
            domain = "health"
        else:
            domain = np.random.choice(list(BIO_FRAGMENTS.keys()))
        fragments = np.random.choice(BIO_FRAGMENTS[domain], size=2, replace=False)
        bio = f"{fragments[0]} and {fragments[1]}"
        bios.append(bio)

    # simple emails based on name + domain
    def make_email(name: str, domain: str) -> str:
        base = name.lower().replace(" ", ".")
        return f"{base}@{domain}"

    domains = [f"{c.lower()}.com" for c in ["acme", "globex", "initech", "umbrella", "wayne", "stark", "wonka"]]

    crm = pd.DataFrame({
        "customer_id": np.arange(n_customers),
        "name": [f"Customer {i}" for i in range(n_customers)],
        "industry": industries,
        "company_size": company_sizes,
        "country": countries,
        "job_title": titles,
        "bio": bios,
        "web_activity_score": web_scores,
        "email_engagement_score": email_scores,
        "email": [make_email(f"user{i}", np.random.choice(domains)) for i in range(n_customers)],
    })

    # Latent logistic for high value
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    industry_bias = crm["industry"].map({
        "Finance": 1.5,
        "WealthTech": 1.2,
    }).fillna(0.0).to_numpy()
    size_term = np.log1p(crm["company_size"]) / 10.0
    engagement = 1.2 * crm["web_activity_score"].to_numpy() + 1.0 * crm["email_engagement_score"].to_numpy()
    keyword_hit = crm["bio"].str.contains("|".join(KEYWORDS), case=False).astype(float).to_numpy()
    noise = np.random.normal(0, 0.5, size=len(crm))

    latent = -1.0 + industry_bias + size_term + engagement + 0.8 * keyword_hit + noise
    prob = sigmoid(latent)
    is_high_value = (np.random.rand(len(crm)) < prob).astype(int)
    crm["is_high_value"] = is_high_value

    # Leads as a recent sample with slight shifts
    leads_idx = np.random.choice(len(crm), size=n_leads, replace=False)
    leads = crm.loc[leads_idx].copy().reset_index(drop=True)
    leads["customer_id"] = np.arange(n_leads) + 10_000_000

    # save
    crm.to_parquet(DATA_DIR / "crm.parquet", index=False)
    leads.to_parquet(DATA_DIR / "leads.parquet", index=False)


if __name__ == "__main__":
    synthesize()

