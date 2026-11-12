"""POST /v1/retrieve — stub (501)

สัญญาเมื่อเปิดใช้ (ดู ``RetrieveResponse``): ไม่ส่ง box_id -> source_required · กล่องที่ไม่รู้จัก/ไม่เป็นสมาชิก ->
response เหมือนกล่องที่เป็นสมาชิกแต่ไม่มีผลทุกฟิลด์ยกเว้น trace_id (200 ไม่ใช่ 404) · top_score < RETRIEVE_MIN_SCORE ->
no_relevant_context แต่ยังคืน results · visibility มาจาก Box ใน AclStore

ของเดิม (commit 4d00cb4: rerank / abstention gate / citation pack / consistency) ยังอยู่ใน app/rerank.py,
app/citations.py, app/consistency.py, app/query_expansion.py — รอต่อในขั้น retrieve จริง
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import CurrentPrincipal, ServiceAuth
from app.api.errors import error_docs, not_implemented
from app.api.schemas import ApiErrorCode, RetrieveRequest, RetrieveResponse

router = APIRouter(prefix="/v1", tags=["retrieve"], dependencies=[ServiceAuth])


@router.post("/retrieve", response_model=RetrieveResponse, summary="ค้นหลักฐานในกล่อง (ไม่ generate คำตอบ)",
             responses=error_docs(ApiErrorCode.UNAVAILABLE))
def retrieve(principal: CurrentPrincipal, body: RetrieveRequest) -> RetrieveResponse:
    raise not_implemented()
