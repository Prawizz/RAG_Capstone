"""dependency กลางของทุก route: ค่าตั้ง, AclStore, service auth, header -> Principal

ลำดับ: route ผูก ``require_service`` เป็น dependency แรก (ระดับ router) จึงตอบ 401 ก่อนตรวจ header ผู้ใช้/body
(ผู้ที่ไม่ผ่าน service auth ไม่ได้รายละเอียด validation) — แต่ multipart ถูก framework อ่านก่อน dependency ทุกตัว
ขนาด body จึงกันด้วย ``UploadLimitMiddleware`` อีกชั้น

header ผู้ใช้ปลายทาง (ทุก endpoint ยกเว้น health) — Gateway ยืนยันตัวตนแล้ว เราไม่ยืนยันเอง:
- ``X-User-Id``: บังคับ ไม่ว่าง (หลัง strip) ยาว <= 128
- ``X-User-Departments``: บังคับ (ต้องส่ง header) คั่นด้วยจุลภาค ตัดช่องว่างหัวท้าย ทิ้งช่องว่าง · ค่าว่าง = ไม่สังกัดแผนกใด
  (fail closed: เห็นเฉพาะไฟล์/กล่องที่ไม่จำกัดแผนก) — ไม่ส่ง header เลย = 422
- ``X-User-Security-Level``: บังคับ เลขฐานสิบล้วน (0, 1, 2 …) ไม่มีเครื่องหมาย/ช่องว่าง/ทศนิยม -> int >= 0
``box_ids`` ของ Principal หาเองจากสมาชิกภาพใน ``AclStore`` (``principal_for``) ห้ามรับจาก Gateway
"""

from __future__ import annotations

import hmac
import re
from typing import Annotated, Optional

from fastapi import Depends, Header, Request
from fastapi.security import APIKeyHeader

from app.access import AclStore, Principal
from app.api.errors import ApiError
from app.api.schemas import ApiErrorCode
from app.config import Settings, get_settings as _load_settings

SERVICE_KEY_HEADER = "X-Service-Key"
USER_ID_MAX = 128
_LEVEL = re.compile(r"^[0-9]{1,6}$")

_service_key_scheme = APIKeyHeader(
    name=SERVICE_KEY_HEADER,
    auto_error=False,
    scheme_name="ServiceKey",
    description="service credential ของ Gateway (placeholder — ของจริงจะเป็น mTLS หรือ signed token) ผิด/ไม่ส่ง -> 401",
)

_acl_store = AclStore()


def get_settings() -> Settings:
    """override ใน test ด้วย ``app.dependency_overrides[get_settings]``"""
    return _load_settings()


def get_acl_store() -> AclStore:
    """AclStore ของ process (ยังอยู่ในหน่วยความจำ) — override ใน test ได้"""
    return _acl_store


def require_service(
    key: Annotated[Optional[str], Depends(_service_key_scheme)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    expected = settings.service_key
    if not expected or not key or not hmac.compare_digest(key.encode(), expected.encode()):
        raise ApiError(ApiErrorCode.UNAUTHORIZED, "service credential ไม่ถูกต้อง")


def _bad(header: str, why: str) -> ApiError:
    return ApiError(ApiErrorCode.INVALID_REQUEST, f"header {header}: {why}")


def parse_departments(raw: str) -> frozenset[str]:
    return frozenset(d.strip() for d in raw.split(",") if d.strip())


def get_principal(
    request: Request,
    acl: Annotated[AclStore, Depends(get_acl_store)],
    # ชื่อพารามิเตอร์ขึ้นต้น hdr_ กันชนกับ path param ของ route (FastAPI จับคู่ด้วยชื่อ เช่น /members/{user_id})
    hdr_user_id: Annotated[str, Header(alias="X-User-Id", description="ผู้ใช้ปลายทาง (Gateway ยืนยันตัวตนแล้ว)")],
    hdr_departments: Annotated[str, Header(alias="X-User-Departments",
                                           description="แผนก คั่นด้วยจุลภาค; ค่าว่าง = ไม่สังกัดแผนกใด (ต้องส่ง header)")],
    hdr_security_level: Annotated[str, Header(alias="X-User-Security-Level",
                                              description="ระดับความลับ จำนวนเต็ม >= 0")],
) -> Principal:
    uid = hdr_user_id.strip()
    if not uid or len(uid) > USER_ID_MAX:
        raise _bad("X-User-Id", f"ต้องไม่ว่างและยาวไม่เกิน {USER_ID_MAX}")
    level = hdr_security_level.strip()
    if not _LEVEL.match(level):
        raise _bad("X-User-Security-Level", "ต้องเป็นจำนวนเต็ม >= 0")
    principal = acl.principal_for(uid, parse_departments(hdr_departments), int(level))
    request.state.principal = principal
    return principal


ServiceAuth = Depends(require_service)
CurrentPrincipal = Annotated[Principal, Depends(get_principal)]
