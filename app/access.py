"""ACL 4 ชั้น: box membership · แผนก · ระดับความปลอดภัย · สิทธิ์ระดับไฟล์

แยกจาก ``app/models.py`` ตามที่ contract ตั้งใจไว้ (``Document`` ไม่มีฟิลด์สิทธิ์) เชื่อมกันด้วย ``file_id``

**กฎเหล็กข้อ 9 ที่โค้ดนี้บังคับ**
1. ``Visibility`` (Public/Private) เป็นคุณสมบัติของ **box** และเปลี่ยนไม่ได้ (``Box`` เป็น frozen model)
   ไฟล์ไม่มี visibility ของตัวเอง — ย้าย box อย่างเดียวที่เปลี่ยนได้ และนั่นคือการ re-ingest
2. สิทธิ์ระดับไฟล์ทำให้ **แคบลง** จาก box ได้อย่างเดียว การตั้งค่าที่ทำให้กว้างขึ้นถูก **ปฏิเสธตอนลงทะเบียน**
   (``AclWideningError``) ไม่ใช่ปรับให้เข้ากันเงียบ ๆ — ACL ที่ตั้งผิดต้องดังตั้งแต่ตอนตั้ง ไม่ใช่ตอนมีคนค้นเจอ
3. กรณีไม่มีสิทธิ์ต้องตอบเหมือนกรณีหาไม่เจอ **ทุกประการ** — ที่นี่จึงคืนแค่ ``bool``
   ส่วน ``AccessDecision.denied_layer`` มีไว้ให้ test และ log ภายในเท่านั้น **ห้ามส่งออกทาง API**

สมาชิกกล่อง (owner/admin/viewer) อยู่ใน ``AclStore`` (``MembershipBackend``) — Gateway ส่งแค่ตัวตน ไม่ส่ง box_ids
ดูรายละเอียดที่ docstring ของ ``AclStore``

**fail closed ทุกจุด** — ไฟล์หรือ box ที่ไม่รู้จัก, แผนกที่เป็น set ว่าง, principal ที่ไม่ได้เป็นสมาชิก box
ทั้งหมดคือ "ไม่มีสิทธิ์" ไม่ใช่ "ไม่จำกัด"

ความหมายของ ``departments``
- ``None``   = ไม่จำกัดแผนก (ทุกแผนกที่เป็นสมาชิก box เข้าได้)
- ``set()``  = **ไม่มีแผนกไหนเข้าได้** (ไม่ใช่ "ทุกแผนก") เพื่อให้ config ที่ถูกลบค่าไปโดยบังเอิญปิดกั้น ไม่ใช่เปิดกว้าง
- ของ ``FileAcl`` ``None`` = สืบทอดจาก box; ถ้าระบุต้องเป็น subset ของ box
  (ไฟล์จึงไม่มีทางขยายกลับไปเป็น "ทุกแผนก" ได้ เพราะ "ทุกแผนก" เขียนได้แค่ด้วย ``None`` = สืบทอด)

``security_level`` = ตัวเลข ยิ่งสูงยิ่งเข้าถึงได้มาก; ``min_security_level`` ของไฟล์ต้อง >= ของ box (แคบลง)

ยังรออนุมัติ: PLAN.md ข้อ 8 ระบุว่า ACL 2 ด่าน (pre-filter ใน Qdrant + re-check กับ ACL store) ยัง "รออาจารย์"
และ Public/Private แยก collection ยังเป็น "เสนอ" โมดูลนี้ทำตามข้อเสนอนั้น ถ้าแบบจำลองสิทธิ์เปลี่ยน
จุดที่ต้องแก้คือ ``effective_acl`` และ ``evaluate`` (ตัวสร้าง filter ของ Qdrant อยู่ใน ``app/vectorstore.py``)
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Optional, Protocol

from pydantic import BaseModel, Field


class Visibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"


class BoxRole(str, Enum):
    """บทบาทในกล่อง — owner/admin ดูแลกล่องได้ (ส่งไฟล์, จัดการสมาชิก); viewer ค้นได้อย่างเดียว"""

    OWNER = "owner"
    ADMIN = "admin"
    VIEWER = "viewer"


MANAGER_ROLES = frozenset({BoxRole.OWNER, BoxRole.ADMIN})


class AclError(ValueError):
    """ความผิดพลาดของการตั้งค่าสิทธิ์ (ไม่ใช่การปฏิเสธการเข้าถึง)"""


class AclWideningError(AclError):
    """สิทธิ์ระดับไฟล์พยายามทำให้กว้างกว่า box (กฎเหล็กข้อ 9)"""


class AclUnknownBoxError(AclError):
    """ลงทะเบียนไฟล์เข้า box ที่ยังไม่รู้จัก"""


class AclLastOwnerError(AclError):
    """ถอดหรือลด role ของ owner คนสุดท้ายของกล่อง (กล่องต้องมี owner อย่างน้อย 1 คนเสมอ)"""


class AclBoxExistsError(AclError):
    """สร้างกล่องด้วย box_id ที่มีอยู่แล้ว (สร้างทับ = ช่องทางเปลี่ยน visibility/เจ้าของ จึงปฏิเสธ)"""


class Box(BaseModel):
    """กล่องเอกสาร — หน่วยที่ผู้ใช้เลือกตอนถาม และเป็นเจ้าของ Public/Private

    frozen: visibility เปลี่ยนไม่ได้หลังสร้าง (กฎเหล็กข้อ 9) การเปลี่ยนกล่องคือการสร้างกล่องใหม่
    """

    model_config = {"frozen": True}

    box_id: str = Field(min_length=1)
    visibility: Visibility
    departments: Optional[frozenset[str]] = None  # None = ไม่จำกัด; set() = ไม่มีใคร
    min_security_level: int = Field(default=0, ge=0)
    name: Optional[str] = None  # ชื่อที่คนอ่าน (ไม่มีผลต่อสิทธิ์)


class FileAcl(BaseModel):
    """สิทธิ์ระดับไฟล์ — ทำให้แคบลงจาก box ได้อย่างเดียว (None = สืบทอดจาก box)"""

    model_config = {"frozen": True}

    file_id: str = Field(min_length=1)
    box_id: str = Field(min_length=1)
    departments: Optional[frozenset[str]] = None
    min_security_level: Optional[int] = Field(default=None, ge=0)


class Principal(BaseModel):
    """ผู้ถาม (ตัวตนมาจาก Gateway — เราไม่ยืนยันตัวตนเอง)"""

    model_config = {"frozen": True}

    user_id: str = Field(min_length=1)
    departments: frozenset[str] = frozenset()
    security_level: int = Field(default=0, ge=0)
    box_ids: frozenset[str] = frozenset()  # box ที่เป็นสมาชิก (snapshot ตอนสร้าง — ใช้กับด่าน 1; ด่าน 2 ถาม AclStore สด)


class EffectiveAcl(BaseModel):
    """สิทธิ์ที่ใช้จริงของไฟล์หนึ่ง = box ผนวกกับการทำให้แคบลงระดับไฟล์"""

    model_config = {"frozen": True}

    file_id: str
    box_id: str
    visibility: Visibility
    departments: Optional[frozenset[str]]
    min_security_level: int

    @property
    def department_list(self) -> Optional[list[str]]:
        """เรียงแล้วเพื่อให้ payload ที่เขียนลง vector store เหมือนกันทุกครั้ง"""
        return None if self.departments is None else sorted(self.departments)


class AccessDecision(BaseModel):
    """ผลการตัดสิน — ``denied_layer`` สำหรับ test/log ภายในเท่านั้น ห้ามส่งออกทาง API"""

    allowed: bool
    denied_layer: Optional[str] = None  # "box" | "file_box_mismatch" | "department" | "security_level" | "unknown"


LAYER_BOX = "box"
LAYER_FILE_MISMATCH = "file_box_mismatch"
LAYER_DEPARTMENT = "department"
LAYER_SECURITY_LEVEL = "security_level"
LAYER_UNKNOWN = "unknown"

ALLOWED = AccessDecision(allowed=True)


def effective_acl(box: Box, file: FileAcl) -> EffectiveAcl:
    """ผนวก box กับสิทธิ์ระดับไฟล์ — ปฏิเสธถ้าไฟล์ทำให้กว้างขึ้น

    Raises:
        AclWideningError: ไฟล์ขยายแผนกออกนอก box หรือลดระดับความปลอดภัยให้ต่ำกว่า box
        AclError: ``file.box_id`` ไม่ตรงกับ box ที่ส่งมา
    """
    if file.box_id != box.box_id:
        raise AclError(f"FileAcl ของ {file.file_id} อยู่ box {file.box_id} ไม่ใช่ {box.box_id}")

    if file.departments is None:
        departments = box.departments
    elif box.departments is not None and not file.departments <= box.departments:
        extra = sorted(file.departments - box.departments)
        raise AclWideningError(
            f"ไฟล์ {file.file_id} ขยายสิทธิ์แผนกออกนอก box {box.box_id}: {extra} "
            "(สิทธิ์ระดับไฟล์ทำให้แคบลงได้อย่างเดียว)"
        )
    else:
        departments = file.departments

    if file.min_security_level is None:
        min_level = box.min_security_level
    elif file.min_security_level < box.min_security_level:
        raise AclWideningError(
            f"ไฟล์ {file.file_id} ลดระดับความปลอดภัยต่ำกว่า box {box.box_id}: "
            f"{file.min_security_level} < {box.min_security_level}"
        )
    else:
        min_level = file.min_security_level

    return EffectiveAcl(
        file_id=file.file_id,
        box_id=box.box_id,
        visibility=box.visibility,
        departments=departments,
        min_security_level=min_level,
    )


def evaluate(principal: Principal, acl: EffectiveAcl) -> AccessDecision:
    """ตัดสินสิทธิ์ 4 ชั้นตามลำดับ — ชั้นแรกที่ไม่ผ่านคือเหตุผล (ใช้ภายในเท่านั้น)"""
    if acl.box_id not in principal.box_ids:
        return AccessDecision(allowed=False, denied_layer=LAYER_BOX)
    if acl.departments is not None and not (principal.departments & acl.departments):
        return AccessDecision(allowed=False, denied_layer=LAYER_DEPARTMENT)
    if principal.security_level < acl.min_security_level:
        return AccessDecision(allowed=False, denied_layer=LAYER_SECURITY_LEVEL)
    return ALLOWED


class MembershipBackend(Protocol):
    """ที่เก็บสมาชิกกล่อง (user_id, box_id) -> role — สลับเป็นฐานข้อมูลจริงได้โดย implement interface นี้

    backend เก็บข้อมูลอย่างเดียว กติกา (owner คนสุดท้าย ฯลฯ) อยู่ที่ ``AclStore`` ให้ทุก backend ได้กติกาเดียวกัน
    ของจริงต้องทำ ``set_role``/``remove`` ของกล่องเดียวกันแบบ serializable (กันถอด owner สองคนพร้อมกันจนไม่เหลือ)
    """

    def get_role(self, box_id: str, user_id: str) -> Optional[BoxRole]: ...

    def set_role(self, box_id: str, user_id: str, role: BoxRole) -> None: ...

    def remove(self, box_id: str, user_id: str) -> None: ...

    def members(self, box_id: str) -> dict[str, BoxRole]: ...

    def boxes_of(self, user_id: str) -> dict[str, BoxRole]: ...


class InMemoryMemberships:
    """``MembershipBackend`` ในหน่วยความจำ (ค่าเริ่มต้น — หายเมื่อ process จบ)"""

    def __init__(self) -> None:
        self._roles: dict[str, dict[str, BoxRole]] = {}  # box_id -> {user_id: role}

    def get_role(self, box_id: str, user_id: str) -> Optional[BoxRole]:
        return self._roles.get(box_id, {}).get(user_id)

    def set_role(self, box_id: str, user_id: str, role: BoxRole) -> None:
        self._roles.setdefault(box_id, {})[user_id] = role

    def remove(self, box_id: str, user_id: str) -> None:
        self._roles.get(box_id, {}).pop(user_id, None)

    def members(self, box_id: str) -> dict[str, BoxRole]:
        return dict(self._roles.get(box_id, {}))

    def boxes_of(self, user_id: str) -> dict[str, BoxRole]:
        return {bid: m[user_id] for bid, m in self._roles.items() if user_id in m}


class AclStore:
    """แหล่งความจริงของสิทธิ์ (ด่านที่ 2 ของ PLAN.md ข้อ 8)

    vector store ใช้ payload เป็นด่านที่ 1 (pre-filter) แต่ payload อาจค้างเก่าได้หลังสิทธิ์เปลี่ยน
    ด่านที่ 2 จึงถามที่นี่ทุกครั้งก่อนคืนผล — ``app/vectorstore.py::VectorStore.search``

    ยังเก็บในหน่วยความจำ: ของจริงจะเป็นฐานข้อมูลหรือ API ขององค์กร (คำถามค้างใน PLAN.md ข้อ 9)
    ``register_file`` ตรวจการขยายสิทธิ์ทันทีเพื่อให้ config ที่ผิดดังตอนตั้ง ไม่ใช่ตอนค้น

    สมาชิกกล่อง (``MembershipBackend``) เป็นความจริงของชั้น box:
    - ``principal_for`` สร้าง Principal จาก header ของ Gateway โดยหา ``box_ids`` เอง (ไม่รับ box_ids จากภายนอก)
    - ``decide`` (ด่าน 2) ถามสมาชิกภาพ **สด** จาก backend ทุกครั้ง นอกเหนือจาก ``principal.box_ids`` — ถอดสมาชิกแล้ว
      ผลค้นของ request ถัดไป (และ request ที่ถือ Principal เก่าอยู่) ถูกตัดทันที
    - การเป็นสมาชิกคือชั้นแรกเท่านั้น ยังต้องผ่านแผนกและระดับความลับตามเดิม (role ไม่ได้ข้ามชั้นอื่น)
    - กล่องที่สร้างผ่าน ``create_box`` มี owner >= 1 คนเสมอ: ถอด/ลด role ของ owner คนสุดท้าย -> ``AclLastOwnerError``
      (กล่องที่ลงทะเบียนด้วย ``register_box`` ตรงๆ ไม่มีสมาชิก = ไม่มีใครเห็น — fail closed)
    """

    def __init__(self, memberships: Optional[MembershipBackend] = None) -> None:
        self._boxes: dict[str, Box] = {}
        self._files: dict[str, FileAcl] = {}
        self._effective: dict[str, EffectiveAcl] = {}
        self._members: MembershipBackend = memberships if memberships is not None else InMemoryMemberships()

    # ---------- ลงทะเบียน ----------

    def register_box(self, box: Box) -> list[str]:
        """ลงทะเบียน/แก้ box — **atomic**: คำนวณ effective ของทุกไฟล์ใน box ให้เสร็จก่อนแล้วค่อยเขียน state

        ถ้า box ใหม่ทำให้ไฟล์ใดไฟล์หนึ่งผิดกฎ (``AclWideningError``) หรือพยายามเปลี่ยน visibility จะปฏิเสธทั้งหมด
        โดยไม่แตะ ``_boxes`` / ``_effective`` เลย (ไม่งั้น box แคบลงไปแล้วแต่ไฟล์ยังถือสิทธิ์เก่า)

        Returns:
            file_id (เรียงแล้ว) ของไฟล์ที่ effective ACL เปลี่ยนไปจากเดิม — ผู้เรียกใช้ไปสั่ง
            ``VectorStore.refresh_acl_payload`` ให้ payload ใน index ตาม
        """
        existing = self._boxes.get(box.box_id)
        if existing is not None and existing.visibility != box.visibility:
            raise AclError(
                f"box {box.box_id} เปลี่ยน visibility จาก {existing.visibility.value} "
                f"เป็น {box.visibility.value} ไม่ได้ (กฎเหล็กข้อ 9)"
            )
        recomputed = {  # ไฟล์ใน box นั้นต้องคำนวณใหม่ทั้งหมดก่อน; error ใดๆ หยุดที่นี่ ก่อนเขียน state
            file.file_id: effective_acl(box, file)
            for file in self._files.values()
            if file.box_id == box.box_id
        }
        changed = sorted(fid for fid, acl in recomputed.items() if self._effective.get(fid) != acl)
        self._boxes[box.box_id] = box
        self._effective.update(recomputed)
        return changed

    def register_file(self, file: FileAcl) -> EffectiveAcl:
        box = self._boxes.get(file.box_id)
        if box is None:
            raise AclUnknownBoxError(f"ยังไม่รู้จัก box {file.box_id} (ลงทะเบียน box ก่อน)")
        acl = effective_acl(box, file)  # ขยายสิทธิ์ -> AclWideningError ที่นี่
        self._files[file.file_id] = file
        self._effective[file.file_id] = acl
        return acl

    def remove_file(self, file_id: str) -> None:
        self._files.pop(file_id, None)
        self._effective.pop(file_id, None)

    def create_box(self, box: Box, owner_user_id: str) -> None:
        """สร้างกล่องใหม่พร้อม owner คนแรก — box_id ซ้ำ -> ``AclBoxExistsError`` (ไม่สร้างทับ)"""
        if not owner_user_id:
            raise AclError("กล่องต้องมี owner")
        if box.box_id in self._boxes:
            raise AclBoxExistsError(f"มี box {box.box_id} อยู่แล้ว")
        self.register_box(box)
        self._members.set_role(box.box_id, owner_user_id, BoxRole.OWNER)

    # ---------- สมาชิก ----------

    def _require_box(self, box_id: str) -> None:
        if box_id not in self._boxes:
            raise AclUnknownBoxError(f"ยังไม่รู้จัก box {box_id}")

    def _would_orphan(self, box_id: str, user_id: str) -> bool:
        """user นี้เป็น owner คนสุดท้ายของกล่องหรือไม่"""
        owners = [u for u, r in self._members.members(box_id).items() if r == BoxRole.OWNER]
        return owners == [user_id]

    def set_member(self, box_id: str, user_id: str, role: BoxRole) -> None:
        """เพิ่มสมาชิก หรือเปลี่ยน role — ลด role ของ owner คนสุดท้าย -> ``AclLastOwnerError``"""
        self._require_box(box_id)
        if not user_id:
            raise AclError("user_id ว่าง")
        if role != BoxRole.OWNER and self._would_orphan(box_id, user_id):
            raise AclLastOwnerError(f"{user_id} เป็น owner คนสุดท้ายของ box {box_id} ลด role ไม่ได้")
        self._members.set_role(box_id, user_id, role)

    def remove_member(self, box_id: str, user_id: str) -> bool:
        """ถอดสมาชิก (คืน False ถ้าไม่ได้เป็นสมาชิกอยู่แล้ว) — owner คนสุดท้าย -> ``AclLastOwnerError``"""
        self._require_box(box_id)
        if self._members.get_role(box_id, user_id) is None:
            return False
        if self._would_orphan(box_id, user_id):
            raise AclLastOwnerError(f"{user_id} เป็น owner คนสุดท้ายของ box {box_id} ถอดไม่ได้")
        self._members.remove(box_id, user_id)
        return True

    def role_of(self, user_id: str, box_id: str) -> Optional[BoxRole]:
        """role ของ user ในกล่องที่มีอยู่จริง (กล่องที่ไม่รู้จัก -> None)"""
        if box_id not in self._boxes:
            return None
        return self._members.get_role(box_id, user_id)

    def members(self, box_id: str) -> dict[str, BoxRole]:
        return self._members.members(box_id) if box_id in self._boxes else {}

    def principal_for(self, user_id: str, departments: Iterable[str] = (), security_level: int = 0) -> Principal:
        """Principal จากตัวตนที่ Gateway ส่งมา — ``box_ids`` มาจากสมาชิกภาพในที่นี่เท่านั้น"""
        boxes = frozenset(b for b in self._members.boxes_of(user_id) if b in self._boxes)
        return Principal(user_id=user_id, departments=frozenset(departments), security_level=security_level,
                         box_ids=boxes)

    # ---------- อ่าน ----------

    def get_box(self, box_id: str) -> Optional[Box]:
        return self._boxes.get(box_id)

    def get_effective(self, file_id: str) -> Optional[EffectiveAcl]:
        return self._effective.get(file_id)

    def boxes_for(self, principal: Principal) -> list[Box]:
        """box ที่ principal เป็นสมาชิก (เรียงตาม box_id) — ใช้ทำ list sources"""
        return sorted(
            (b for bid, b in self._boxes.items() if bid in principal.box_ids),
            key=lambda b: b.box_id,
        )

    # ---------- ตัดสิน ----------

    def decide(self, principal: Principal, file_id: str) -> AccessDecision:
        """ตัดสินสิทธิ์ของไฟล์หนึ่ง — ไฟล์ที่ไม่รู้จักถือว่าไม่มีสิทธิ์ (fail closed)"""
        acl = self._effective.get(file_id)
        if acl is None:
            return AccessDecision(allowed=False, denied_layer=LAYER_UNKNOWN)
        if self._members.get_role(acl.box_id, principal.user_id) is None:  # สมาชิกภาพสด (ถอดแล้วมีผลทันที)
            return AccessDecision(allowed=False, denied_layer=LAYER_BOX)
        return evaluate(principal, acl)  # ยังต้องผ่าน box snapshot + แผนก + ระดับความลับตามเดิม

    def can_access(self, principal: Principal, file_id: str) -> bool:
        """คำตอบที่ปลอดภัยสำหรับทุกที่ที่ไม่ใช่ test/log — ไม่บอกเหตุผล"""
        return self.decide(principal, file_id).allowed

    def accessible_files(self, principal: Principal, file_ids: Iterable[str]) -> list[str]:
        return [fid for fid in file_ids if self.can_access(principal, fid)]
