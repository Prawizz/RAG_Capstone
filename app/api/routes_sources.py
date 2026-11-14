"""GET /v1/sources · GET /v1/sources/{box_id}/files

- sources: เฉพาะกล่องที่ผู้เรียกเป็นสมาชิก (เรียงตาม box_id) · ``file_count`` / ``updated_at`` นับเฉพาะไฟล์ที่ผู้เรียกเห็น (ACL)
- files: ไม่เป็นสมาชิก / ไม่มีกล่อง -> 404 (เหมือนกัน) · ไฟล์กรองตาม ACL ทุกชั้น (ไฟล์ที่ไม่มีสิทธิ์ไม่ปรากฏและไม่นับ)
  เรียงตามเวลาที่ส่งเข้า (created_at, file_id) หน้าละ ``FILES_PAGE_SIZE`` · ``cursor`` ทึบ (base64 ของตำแหน่งสุดท้าย) —
  cursor ผิดรูป -> 422
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Path, Query, Request

from app.access import Principal
from app.api.deps import CurrentPrincipal, ServiceAuth, get_services
from app.api.errors import ApiError, error_docs, not_found, trace_id_of
from app.api.schemas import BOX_ID_PATTERN, ApiErrorCode, FileItem, FilesResponse, SourceBox, SourcesResponse
from app.api.services import Services
from app.storage import FileRecord

router = APIRouter(prefix="/v1", tags=["sources"], dependencies=[ServiceAuth])

FILES_PAGE_SIZE = 50
_SCAN_BATCH = 200


def encode_cursor(rec: FileRecord) -> str:
    raw = json.dumps([rec.created_at, rec.file_id]).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[str, str]:
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        created_at, file_id = json.loads(raw)
        if not isinstance(created_at, str) or not isinstance(file_id, str):
            raise ValueError
        return created_at, file_id
    except (binascii.Error, ValueError, TypeError, UnicodeDecodeError):
        raise ApiError(ApiErrorCode.INVALID_REQUEST, "cursor ไม่ถูกต้อง") from None


def visible_files(services: Services, principal: Principal, box_id: str) -> list[FileRecord]:
    return [f for f in services.storage.files_in_box(box_id) if services.acl.can_access(principal, f.file_id)]


@router.get("/sources", response_model=SourcesResponse, summary="กล่องที่ผู้ใช้เป็นสมาชิก", responses=error_docs())
def list_sources(
    request: Request, principal: CurrentPrincipal, services: Annotated[Services, Depends(get_services)],
) -> SourcesResponse:
    out = []
    for box_id in sorted(principal.box_ids):
        box = services.acl.get_box(box_id)
        role = services.acl.role_of(principal.user_id, box_id)
        if box is None or role is None:  # ถูกถอดระหว่าง request
            continue
        files = visible_files(services, principal, box_id)
        out.append(SourceBox(box_id=box_id, name=box.name or box_id, visibility=box.visibility, role=role,
                             file_count=len(files), updated_at=max((f.updated_at for f in files), default=None)))
    return SourcesResponse(sources=out, trace_id=trace_id_of(request))


@router.get("/sources/{box_id}/files", response_model=FilesResponse, summary="ไฟล์ในกล่อง (กรองตาม ACL)",
            responses=error_docs(ApiErrorCode.NOT_FOUND))
def list_files(
    request: Request,
    principal: CurrentPrincipal,
    services: Annotated[Services, Depends(get_services)],
    box_id: Annotated[str, Path(pattern=BOX_ID_PATTERN)],
    cursor: Annotated[Optional[str], Query(max_length=512, description="next_cursor จากหน้าก่อน (opaque)")] = None,
) -> FilesResponse:
    if services.acl.role_of(principal.user_id, box_id) is None:
        raise not_found()
    after = decode_cursor(cursor) if cursor else None
    page: list[FileRecord] = []
    more = False
    while True:
        batch = services.storage.files_page(box_id, after, _SCAN_BATCH)
        for rec in batch:
            if not services.acl.can_access(principal, rec.file_id):
                continue
            if len(page) == FILES_PAGE_SIZE:
                more = True
                break
            page.append(rec)
        if more or len(batch) < _SCAN_BATCH:
            break
        after = (batch[-1].created_at, batch[-1].file_id)
    files = [FileItem(file_id=f.file_id, filename=f.filename, detected_type=f.detected_type, status=f.status,
                      units_total=f.units_total, gaps=f.gaps, ingested_at=f.ingested_at) for f in page]
    return FilesResponse(files=files, next_cursor=encode_cursor(page[-1]) if more else None,
                         trace_id=trace_id_of(request))
