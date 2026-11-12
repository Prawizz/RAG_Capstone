"""GET /v1/sources · GET /v1/sources/{box_id}/files — stub (501)

สัญญาเมื่อเปิดใช้: เฉพาะกล่องที่เป็นสมาชิก · ไฟล์กรองตาม ACL · ไม่เป็นสมาชิก / ไม่มีกล่อง -> 404 (ตอบเหมือนกัน)
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Path, Query

from app.api.deps import CurrentPrincipal, ServiceAuth
from app.api.errors import error_docs, not_implemented
from app.api.schemas import BOX_ID_PATTERN, ApiErrorCode, FilesResponse, SourcesResponse

router = APIRouter(prefix="/v1", tags=["sources"], dependencies=[ServiceAuth])


@router.get("/sources", response_model=SourcesResponse, summary="กล่องที่ผู้ใช้เป็นสมาชิก",
            responses=error_docs())
def list_sources(principal: CurrentPrincipal) -> SourcesResponse:
    raise not_implemented()


@router.get("/sources/{box_id}/files", response_model=FilesResponse, summary="ไฟล์ในกล่อง (กรองตาม ACL)",
            responses=error_docs(ApiErrorCode.NOT_FOUND))
def list_files(
    principal: CurrentPrincipal,
    box_id: Annotated[str, Path(pattern=BOX_ID_PATTERN)],
    cursor: Annotated[Optional[str], Query(max_length=512, description="next_cursor จากหน้าก่อน (opaque)")] = None,
) -> FilesResponse:
    raise not_implemented()
