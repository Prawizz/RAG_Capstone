"""GET /v1/jobs/{job_id} — stub (501)

สัญญาเมื่อเปิดใช้: เห็นได้เฉพาะผู้ส่งหรือ owner/admin ของกล่อง — ไม่ใช่ / ไม่มีงานนี้ -> 404 (ตอบเหมือนกัน)
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path

from app.api.deps import CurrentPrincipal, ServiceAuth
from app.api.errors import error_docs, not_implemented
from app.api.schemas import ApiErrorCode, JobResponse

router = APIRouter(prefix="/v1", tags=["jobs"], dependencies=[ServiceAuth])


@router.get("/jobs/{job_id}", response_model=JobResponse, summary="สถานะงาน ingest รายไฟล์และราย unit",
            responses=error_docs(ApiErrorCode.NOT_FOUND))
def get_job(principal: CurrentPrincipal, job_id: Annotated[str, Path(min_length=1, max_length=128)]) -> JobResponse:
    raise not_implemented()
