from __future__ import annotations

import re


_WS_RE = re.compile(r"\s+")


def normalize_email(email: str | None) -> str:
    if not email:
        return ""
    email = email.strip().lower()
    return email


def normalize_name(name: str | None) -> str:
    if not name:
        return ""
    name = name.strip().lower()
    name = _WS_RE.sub(" ", name)
    return name


def normalize_company(name: str | None) -> str:
    if not name:
        return ""
    name = name.strip().lower()
    name = name.replace(",", " ")
    name = _WS_RE.sub(" ", name)
    # remove common suffixes
    for suf in [" inc", " llc", " ltd", " limited", " gmbh", " s.a."]:
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    return name


