"""GET /v1/jobs/{job_id} — สถานะงาน ingest

เห็นได้เฉพาะ (ก) ผู้ส่งงานที่ยังเป็นสมาชิกกล่อง หรือ (ข) owner/admin ของกล่อง — นอกนั้น / ไม่มีงานนี้ -> 404 (ตอบเหมือนกัน)
งานของไฟล์ที่ถูกลบ/กำลังลบ -> 404 (งานยังอยู่ใน SQLite เป็นประวัติ แต่ไม่เปิดทาง API)
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request

from app.access import MANAGER_ROLES
from app.api.deps import CurrentPrincipal, ServiceAuth, get_services
from app.api.errors import error_docs, not_found, trace_id_of
from app.api.schemas import ApiErrorCode, JobError, JobResponse, Progress, UnitCounts
from app.api.services import Services
from app.storage import HIDDEN_FILE_STATUSES, JobRecord

router = APIRouter(prefix="/v1", tags=["jobs"], dependencies=[ServiceAuth])


def job_view(job: JobRecord, filename: str, trace_id: str) -> JobResponse:
    error = None if job.error_code is None else JobError(code=job.error_code, message=job.error_message or "")
    return JobResponse(
        job_id=job.job_id, file_id=job.file_id, box_id=job.box_id, filename=filename, status=job.status,
        progress=Progress(pages_total=job.pages_total, pages_done=job.pages_done),
        document_status=job.document_status, units=UnitCounts(**job.units), gaps=job.gaps, error=error,
        created_at=job.created_at, updated_at=job.updated_at, trace_id=trace_id,
    )


@router.get("/jobs/{job_id}", response_model=JobResponse, summary="สถานะงาน ingest รายไฟล์และราย unit",
            responses=error_docs(ApiErrorCode.NOT_FOUND))
def get_job(
    request: Request,
    principal: CurrentPrincipal,
    services: Annotated[Services, Depends(get_services)],
    job_id: Annotated[str, Path(min_length=1, max_length=128)],
) -> JobResponse:
    job = services.storage.get_job(job_id)
    if job is None:
        raise not_found()
    role = services.acl.role_of(principal.user_id, job.box_id)
    if not (role in MANAGER_ROLES or (role is not None and job.submitted_by == principal.user_id)):
        raise not_found()
    file = services.storage.get_file(job.file_id)
    if file is None or file.status in HIDDEN_FILE_STATUSES:
        raise not_found()
    return job_view(job, file.filename, trace_id_of(request))
