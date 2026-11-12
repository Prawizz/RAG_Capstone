"""admin (สำหรับคนดูแลกล่อง ไม่ใช่ Gateway ทั่วไป) — stub (501)

- POST /v1/boxes: ผู้เรียกต้องอยู่ใน SYSTEM_ADMIN_USERS (ไม่ใช่ -> 404) และเป็น owner คนแรก · box_id ซ้ำ -> 422
- PUT /v1/boxes/{box_id}/members · DELETE /v1/boxes/{box_id}/members/{user_id}: ผู้เรียกต้องเป็น owner/admin ของกล่อง
  (ไม่ใช่ / ไม่มีกล่อง -> 404) · ให้/ถอด role owner ได้เฉพาะ owner · owner คนสุดท้ายถอด/ลด role ไม่ได้ -> 422
- ไม่มี endpoint แก้ visibility (กฎเหล็กข้อ 9)
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path

from app.api.deps import CurrentPrincipal, ServiceAuth
from app.api.errors import error_docs, not_implemented
from app.api.schemas import (
    BOX_ID_PATTERN,
    ApiErrorCode,
    BoxResponse,
    CreateBoxRequest,
    MemberRemovedResponse,
    MemberResponse,
    PutMemberRequest,
)

router = APIRouter(prefix="/v1/boxes", tags=["admin"], dependencies=[ServiceAuth])

BoxId = Annotated[str, Path(pattern=BOX_ID_PATTERN)]


@router.post("", status_code=201, response_model=BoxResponse, summary="สร้างกล่อง (admin ระดับระบบ)",
             responses=error_docs(ApiErrorCode.NOT_FOUND))
def create_box(principal: CurrentPrincipal, body: CreateBoxRequest) -> BoxResponse:
    raise not_implemented()


@router.put("/{box_id}/members", response_model=MemberResponse, summary="เพิ่มสมาชิก / เปลี่ยน role",
            responses=error_docs(ApiErrorCode.NOT_FOUND))
def put_member(principal: CurrentPrincipal, box_id: BoxId, body: PutMemberRequest) -> MemberResponse:
    raise not_implemented()


@router.delete("/{box_id}/members/{user_id}", response_model=MemberRemovedResponse, summary="ถอดสมาชิก",
               responses=error_docs(ApiErrorCode.NOT_FOUND))
def delete_member(
    principal: CurrentPrincipal, box_id: BoxId, user_id: Annotated[str, Path(min_length=1, max_length=128)],
) -> MemberRemovedResponse:
    raise not_implemented()
