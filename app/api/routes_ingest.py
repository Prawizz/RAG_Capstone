"""POST /v1/ingest (multipart) — รับไฟล์เข้ากล่องแล้วเข้าคิว (``app/ingest_service.py``) worker ทำต่อแบบ async

202 งานใหม่ · 200 ไฟล์ซ้ำ (content_hash เดียวกันในกล่องเดียวกัน, duplicate true — งานเดิมที่ล้มชั่วคราว หรือยังมีหน้ารอ OCR เพราะ OCR ใช้ไม่ได้ เข้าคิวใหม่)
ผู้ส่งไม่ใช่ owner/admin หรือไม่มีกล่อง -> 404 · file_acl กว้างกว่ากล่อง -> 422 acl_widening · เกิน 50 MB -> 413 ·
ชนิดไฟล์ (magic bytes) ไม่รองรับ -> 415 · ลำดับตรวจ: รูปแบบฟอร์ม/file_acl (422) -> ขนาด (413) -> สิทธิ์ (404) ->
acl_widening (422) -> ชนิดไฟล์ (415) -> ไฟล์ซ้ำ
"""

from __future__ import annotations

import json
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.access import AclWideningError
from app.api.deps import CurrentPrincipal, ServiceAuth, get_services, get_settings
from app.api.errors import TRACE_HEADER, ApiError, error_docs, not_found, trace_id_of
from app.api.schemas import BOX_ID_PATTERN, ApiErrorCode, FileAclInput, IngestResponse
from app.api.services import Services
from app.config import Settings
from app.ingest_service import IngestNotFound, UnsupportedFile

router = APIRouter(prefix="/v1", tags=["ingest"], dependencies=[ServiceAuth])

DISPLAY_NAME_MAX = 255


def parse_file_acl(raw: Optional[str]) -> Optional[FileAclInput]:
    """ฟอร์มฟิลด์ file_acl (JSON) -> FileAclInput; ไม่ส่ง/ว่าง = None (สืบทอดจากกล่อง); ผิดรูป -> 422 invalid_request"""
    if raw is None or not raw.strip():
        return None
    try:
        return FileAclInput.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        raise ApiError(ApiErrorCode.INVALID_REQUEST, f"file_acl ไม่ใช่ JSON ตาม FileAclInput: {e}") from None


@router.post(
    "/ingest", status_code=202, response_model=IngestResponse, summary="ส่งไฟล์เข้ากล่อง (ประมวลผลแบบ async)",
    responses={
        200: {"model": IngestResponse, "description": "ไฟล์ซ้ำในกล่องเดียวกัน (duplicate true): สถานะเดิม หรือ queued (job_id ใหม่) เมื่องานเดิมล้มชั่วคราว / ยังมีหน้ารอ OCR"},
        **error_docs(ApiErrorCode.NOT_FOUND, ApiErrorCode.ACL_WIDENING, ApiErrorCode.TOO_LARGE,
                     ApiErrorCode.UNSUPPORTED_TYPE, ApiErrorCode.UNAVAILABLE),
    },
)
async def ingest(
    request: Request,
    principal: CurrentPrincipal,
    settings: Annotated[Settings, Depends(get_settings)],
    services: Annotated[Services, Depends(get_services)],
    file: Annotated[UploadFile, File(description="เอกสาร (ชนิดตรวจจาก magic bytes ไม่เชื่อนามสกุล) ไม่เกิน 50 MB")],
    box_id: Annotated[str, Form(pattern=BOX_ID_PATTERN)],
    file_acl: Annotated[Optional[str], Form(description="JSON ของ FileAclInput (ไม่บังคับ)")] = None,
    display_name: Annotated[Optional[str], Form(max_length=DISPLAY_NAME_MAX)] = None,
):
    acl_input = parse_file_acl(file_acl)
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise ApiError(ApiErrorCode.TOO_LARGE, f"ไฟล์ใหญ่เกิน {settings.max_upload_bytes // (1024 * 1024)} MB")
    try:
        result = services.ingest.submit(principal, box_id, data, upload_name=file.filename,
                                        display_name=display_name, file_acl=acl_input)
    except IngestNotFound:
        raise not_found() from None
    except AclWideningError as e:
        raise ApiError(ApiErrorCode.ACL_WIDENING, str(e)) from None
    except UnsupportedFile as e:
        raise ApiError(ApiErrorCode.UNSUPPORTED_TYPE, str(e)) from None
    tid = trace_id_of(request)
    body = IngestResponse(job_id=result.job.job_id, file_id=result.file.file_id, status=result.job.status,
                          duplicate=result.duplicate, trace_id=tid)
    return JSONResponse(body.model_dump(mode="json"), status_code=200 if result.duplicate else 202,
                        headers={TRACE_HEADER: tid})
