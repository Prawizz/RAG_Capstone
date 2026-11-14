"""admin (สำหรับคนดูแลกล่อง ไม่ใช่ Gateway ทั่วไป)

- POST /v1/boxes: ผู้เรียกต้องอยู่ใน SYSTEM_ADMIN_USERS (ไม่ใช่ -> 404) และเป็น owner คนแรก · box_id ซ้ำ -> 422
- PUT /v1/boxes/{box_id}/members · DELETE /v1/boxes/{box_id}/members/{user_id}: ผู้เรียกต้องเป็น owner/admin ของกล่อง
  (ไม่ใช่ / ไม่มีกล่อง -> 404) · ให้/ถอด/เปลี่ยน role ของ owner ได้เฉพาะ owner (admin ทำ -> 404) ·
  owner คนสุดท้ายถอด/ลด role ไม่ได้ -> 422 · ถอดคนที่ไม่ได้เป็นสมาชิก -> 404
- ไม่มี endpoint แก้ visibility (กฎเหล็กข้อ 9) · ถอดสมาชิกมีผลทันที (ด่าน 2 ของการค้นถามสมาชิกภาพสด)
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request
from fastapi.responses import JSONResponse

from app.access import MANAGER_ROLES, AclBoxExistsError, AclLastOwnerError, Box, BoxRole
from app.api.deps import CurrentPrincipal, ServiceAuth, get_services
from app.api.errors import TRACE_HEADER, ApiError, error_docs, not_found, trace_id_of
from app.api.schemas import (
    BOX_ID_PATTERN,
    ApiErrorCode,
    BoxResponse,
    CreateBoxRequest,
    MemberRemovedResponse,
    MemberResponse,
    PutMemberRequest,
)
from app.api.services import Services

router = APIRouter(prefix="/v1/boxes", tags=["admin"], dependencies=[ServiceAuth])

BoxId = Annotated[str, Path(pattern=BOX_ID_PATTERN)]
Svc = Annotated[Services, Depends(get_services)]


def _manager_role(services: Services, user_id: str, box_id: str) -> BoxRole:
    role = services.acl.role_of(user_id, box_id)
    if role not in MANAGER_ROLES:
        raise not_found()
    return role


@router.post("", status_code=201, response_model=BoxResponse, summary="สร้างกล่อง (admin ระดับระบบ)",
             responses=error_docs(ApiErrorCode.NOT_FOUND))
def create_box(request: Request, principal: CurrentPrincipal, services: Svc, body: CreateBoxRequest):
    if principal.user_id not in services.settings.system_admin_users:
        raise not_found()
    box = Box(box_id=body.box_id, name=body.name, visibility=body.visibility,
              departments=None if body.departments is None else frozenset(body.departments),
              min_security_level=body.min_security_level)
    try:
        services.acl.create_box(box, principal.user_id)
    except AclBoxExistsError:
        raise ApiError(ApiErrorCode.INVALID_REQUEST, f"มีกล่อง {body.box_id} อยู่แล้ว") from None
    tid = trace_id_of(request)
    out = BoxResponse(box_id=box.box_id, name=body.name, visibility=box.visibility,
                      departments=None if box.departments is None else sorted(box.departments),
                      min_security_level=box.min_security_level, owner=principal.user_id, trace_id=tid)
    return JSONResponse(out.model_dump(mode="json"), status_code=201, headers={TRACE_HEADER: tid})


@router.put("/{box_id}/members", response_model=MemberResponse, summary="เพิ่มสมาชิก / เปลี่ยน role",
            responses=error_docs(ApiErrorCode.NOT_FOUND))
def put_member(request: Request, principal: CurrentPrincipal, services: Svc, box_id: BoxId,
               body: PutMemberRequest) -> MemberResponse:
    caller = _manager_role(services, principal.user_id, box_id)
    current = services.acl.role_of(body.user_id, box_id)
    if BoxRole.OWNER in (body.role, current) and caller != BoxRole.OWNER:
        raise not_found()
    try:
        services.acl.set_member(box_id, body.user_id, body.role)
    except AclLastOwnerError as e:
        raise ApiError(ApiErrorCode.INVALID_REQUEST, str(e)) from None
    return MemberResponse(box_id=box_id, user_id=body.user_id, role=body.role, trace_id=trace_id_of(request))


@router.delete("/{box_id}/members/{user_id}", response_model=MemberRemovedResponse, summary="ถอดสมาชิก",
               responses=error_docs(ApiErrorCode.NOT_FOUND))
def delete_member(
    request: Request, principal: CurrentPrincipal, services: Svc, box_id: BoxId,
    user_id: Annotated[str, Path(min_length=1, max_length=128)],
) -> MemberRemovedResponse:
    caller = _manager_role(services, principal.user_id, box_id)
    target = services.acl.role_of(user_id, box_id)
    if target is None or (target == BoxRole.OWNER and caller != BoxRole.OWNER):
        raise not_found()
    try:
        removed = services.acl.remove_member(box_id, user_id)
    except AclLastOwnerError as e:
        raise ApiError(ApiErrorCode.INVALID_REQUEST, str(e)) from None
    if not removed:  # ถูกถอดไปแล้วระหว่างทาง
        raise not_found()
    return MemberRemovedResponse(box_id=box_id, user_id=user_id, trace_id=trace_id_of(request))
