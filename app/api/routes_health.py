"""[M7] Health check routes"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/v1", tags=["Health"])


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "rag-api"}
