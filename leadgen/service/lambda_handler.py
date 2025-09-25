from __future__ import annotations

from mangum import Mangum

from leadgen.service.app import app


# AWS Lambda entrypoint (ASGI adapter for FastAPI)
handler = Mangum(app)


