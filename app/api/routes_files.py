"""DELETE /v1/files/{file_id} — ลบไฟล์และทุกอย่างที่ได้จากไฟล์ (``app/file_deleter.py``)

200 ลบแล้ว (บอกว่าลบอะไรไปบ้าง) · ไม่มีไฟล์ / ไม่เห็นไฟล์ / ไม่ใช่ owner-admin ของกล่อง / ลบไปแล้ว / กำลังลบ -> 404 (เหมือนกัน)
· งาน ingest ของไฟล์กำลังทำอยู่ (extracting/ocr/indexing) -> 409 conflict (งานที่ยัง queued ถูกยกเลิกแล้วลบได้เลย)
· vector store ใช้ไม่ได้ -> 503 (ไฟล์ถูกซ่อนแล้วและจะถูกลบต่อจนครบเมื่อกลับมา — ลบซ้ำจะได้ 404)
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request

from app.api.deps import CurrentPrincipal, ServiceAuth, get_services
from app.api.errors import ApiError, error_docs, not_found, trace_id_of
from app.api.schemas import ApiErrorCode, DeleteFileResponse
from app.api.services import Services
from app.file_deleter import DeleteConflict, DeleteNotFound, DeleteUnavailable

log = logging.getLogger("app.api.files")

router = APIRouter(prefix="/v1", tags=["files"], dependencies=[ServiceAuth])


@router.delete("/files/{file_id}", response_model=DeleteFileResponse, summary="ลบไฟล์ (owner/admin ของกล่อง)",
               responses=error_docs(ApiErrorCode.NOT_FOUND, ApiErrorCode.CONFLICT, ApiErrorCode.UNAVAILABLE))
def delete_file(
    request: Request,
    principal: CurrentPrincipal,
    services: Annotated[Services, Depends(get_services)],
    file_id: Annotated[str, Path(min_length=1, max_length=128)],
) -> DeleteFileResponse:
    tid = trace_id_of(request)
    try:
        report = services.deleter.delete(principal, file_id)
    except DeleteNotFound:
        raise not_found() from None
    except DeleteConflict as e:
        raise ApiError(ApiErrorCode.CONFLICT,
                       f"งาน ingest ของไฟล์นี้กำลังทำอยู่ ({e.status}) ลบได้หลังงานจบ — ดูสถานะที่ /v1/jobs") from None
    except DeleteUnavailable:
        raise ApiError(ApiErrorCode.UNAVAILABLE,
                       "vector store ใช้ไม่ได้ชั่วคราว — ไฟล์ถูกซ่อนแล้วและจะถูกลบต่อจนครบเมื่อกลับมา") from None
    # ผลกับแคช OCR ลง log เท่านั้น (ไม่ส่งออก: kept_shared บอกว่ามีไฟล์เนื้อหาเดียวกันในกล่องอื่น — กฎเหล็กข้อ 9)
    log.info("delete trace_id=%s file_id=%s box_id=%s by=%s points=%d ocr_cache=%s ocr_cache_entries=%d", tid,
             report.file_id, report.box_id, principal.user_id, report.points_deleted, report.ocr_cache,
             report.ocr_cache_entries)
    return DeleteFileResponse(
        file_id=report.file_id, box_id=report.box_id, points_deleted=report.points_deleted,
        original_deleted=report.original_deleted, document_deleted=report.document_deleted,
        cancelled_jobs=report.cancelled_jobs, trace_id=tid,
    )
