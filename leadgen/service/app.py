from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from leadgen.service.bootstrap import Components, embed_one, load_components, score_one, is_duplicate_email


app = FastAPI()
components: Components | None = None


class Lead(BaseModel):
    customer_id: int | None = None
    name: str | None = None
    industry: str
    company_size: int
    country: str
    job_title: str
    bio: str
    web_activity_score: float
    email_engagement_score: float
    email: str | None = None


@app.on_event("startup")
def _startup() -> None:
    global components
    components = load_components()
    # Load normalized email set
    from leadgen.config import ARTIFACTS_DIR
    emails_file = ARTIFACTS_DIR / "emails.txt"
    if emails_file.exists():
        app.state.crm_emails = set(emails_file.read_text().splitlines())
    else:
        app.state.crm_emails = set()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/score_lead")
def score_lead_endpoint(lead: Lead) -> Dict[str, Any]:
    assert components is not None, "Components not loaded"
    # Duplicate check by email (short-circuit)
    if is_duplicate_email(lead.dict(), getattr(app.state, "crm_emails", set())):
        return {"is_duplicate": True, "reason": "email_exact_match"}
    emb = embed_one(lead.dict(), components)
    scores = score_one(emb, components)
    scores.update({"is_duplicate": False})
    return scores

