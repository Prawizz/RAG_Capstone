"""GET /v1/health/live (ไม่ต้อง auth — สำหรับ orchestrator probe) · GET /v1/health/ready

ready ต้องผ่าน service auth (เผยเวอร์ชัน สถานะส่วนประกอบ และอุปกรณ์ที่ใช้) แต่ไม่ต้องมี header ผู้ใช้
ตรวจจริงตาม ``app/api/readiness.py`` (ไม่โหลดโมเดล) · 200 เมื่อ ready ไม่งั้น 503 (body ReadyResponse ทั้งสองกรณี)
รวมสถานะ query embedder ของ ``/v1/retrieve`` ด้วย (``readiness.with_query_embedder``)
"""

from __future__ import annotations

from typing import Annotated, Callable

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.api import readiness
from app.api.deps import ServiceAuth, get_services, get_settings
from app.api.errors import TRACE_HEADER, error_docs, trace_id_of
from app.api.schemas import API_VERSION, ApiErrorCode, LiveResponse, ReadyResponse, ReadyStatus
from app.api.services import Services
from app.config import Settings

router = APIRouter(prefix="/v1/health", tags=["health"])


def get_probe() -> Callable[[Settings], readiness.Readiness]:
    """override ใน test (ไม่ต้องมี Qdrant / GPU จริง)"""
    return readiness.probe


@router.get("/live", response_model=LiveResponse, summary="process ยังทำงานอยู่",
            responses=error_docs(ApiErrorCode.INTERNAL, common=False))
def live(request: Request) -> LiveResponse:
    return LiveResponse(status="ok", trace_id=trace_id_of(request))


@router.get(
    "/ready", response_model=ReadyResponse,
    summary="พร้อมรับงานไหม + อุปกรณ์/OCR backend ที่ใช้ (200 = ready, 503 = degraded/not_ready)",
    dependencies=[ServiceAuth],
    responses={
        503: {"model": ReadyResponse, "description": "degraded / not_ready (body เป็น ReadyResponse)"},
        **error_docs(ApiErrorCode.UNAUTHORIZED, ApiErrorCode.INTERNAL, common=False),
    },
)
def ready(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    probe: Annotated[Callable[[Settings], readiness.Readiness], Depends(get_probe)],
    services: Annotated[Services, Depends(get_services)],
):
    result = readiness.with_reranker(readiness.with_query_embedder(probe(settings), services.query_embedder),
                                   services.reranker)
    tid = trace_id_of(request)
    body = ReadyResponse(status=result.status, version=API_VERSION, components=result.components,
                         runtime=result.runtime, trace_id=tid)
    code = 200 if result.status == ReadyStatus.READY else 503
    return JSONResponse(body.model_dump(mode="json"), status_code=code, headers={TRACE_HEADER: tid})
