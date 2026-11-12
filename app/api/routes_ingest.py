"""POST /v1/ingest (multipart) — stub: ตรวจ auth, header ผู้ใช้, ฟอร์ม และขนาดจริง แล้วคืน 501

สัญญาเมื่อเปิดใช้: 202 งานใหม่ · 200 ไฟล์ซ้ำ (content_hash เดียวกันในกล่องเดียวกัน, duplicate true, สถานะเดิม)
ผู้ส่งต้องเป็น owner/admin ของกล่อง ไม่ใช่ (หรือกล่องไม่มี) -> 404 · file_acl กว้างกว่ากล่อง -> 422 acl_widening
เกินเพดานขนาด -> 413 · ชนิดไฟล์ (magic bytes) ไม่รองรับ -> 415
"""

from __future__ import annotations

import json
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import ValidationError

from app.api.deps import CurrentPrincipal, ServiceAuth, get_settings
from app.api.errors import ApiError, error_docs, not_implemented
from app.api.schemas import BOX_ID_PATTERN, ApiErrorCode, FileAclInput, IngestResponse
from app.config import Settings

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
        200: {"model": IngestResponse, "description": "ไฟล์ซ้ำในกล่องเดียวกัน (duplicate true, สถานะเดิม)"},
        **error_docs(ApiErrorCode.NOT_FOUND, ApiErrorCode.ACL_WIDENING, ApiErrorCode.TOO_LARGE,
                     ApiErrorCode.UNSUPPORTED_TYPE, ApiErrorCode.UNAVAILABLE),
    },
)
async def ingest(
    principal: CurrentPrincipal,
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File(description="เอกสาร (ชนิดตรวจจาก magic bytes ไม่เชื่อนามสกุล) ไม่เกิน 50 MB")],
    box_id: Annotated[str, Form(pattern=BOX_ID_PATTERN)],
    file_acl: Annotated[Optional[str], Form(description="JSON ของ FileAclInput (ไม่บังคับ)")] = None,
    display_name: Annotated[Optional[str], Form(max_length=DISPLAY_NAME_MAX)] = None,
) -> IngestResponse:
    parse_file_acl(file_acl)
    size = file.size
    if size is None:
        size = len(await file.read())
    if size > settings.max_upload_bytes:
        raise ApiError(ApiErrorCode.TOO_LARGE, f"ไฟล์ใหญ่เกิน {settings.max_upload_bytes // (1024 * 1024)} MB")
    raise not_implemented()
