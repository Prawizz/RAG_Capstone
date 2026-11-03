"""ACL 4 ชั้น: box membership · แผนก · ระดับความปลอดภัย · สิทธิ์ระดับไฟล์

แยกจาก ``app/models.py`` ตามที่ contract ตั้งใจไว้ (``Document`` ไม่มีฟิลด์สิทธิ์) เชื่อมกันด้วย ``file_id``

**กฎเหล็กข้อ 9 ที่โค้ดนี้บังคับ**
1. ``Visibility`` (Public/Private) เป็นคุณสมบัติของ **box** และเปลี่ยนไม่ได้ (``Box`` เป็น frozen model)
   ไฟล์ไม่มี visibility ของตัวเอง — ย้าย box อย่างเดียวที่เปลี่ยนได้ และนั่นคือการ re-ingest
2. สิทธิ์ระดับไฟล์ทำให้ **แคบลง** จาก box ได้อย่างเดียว การตั้งค่าที่ทำให้กว้างขึ้นถูก **ปฏิเสธตอนลงทะเบียน**
   (``AclWideningError``) ไม่ใช่ปรับให้เข้ากันเงียบ ๆ — ACL ที่ตั้งผิดต้องดังตั้งแต่ตอนตั้ง ไม่ใช่ตอนมีคนค้นเจอ
3. กรณีไม่มีสิทธิ์ต้องตอบเหมือนกรณีหาไม่เจอ **ทุกประการ** — ที่นี่จึงคืนแค่ ``bool``
   ส่วน ``AccessDecision.denied_layer`` มีไว้ให้ test และ log ภายในเท่านั้น **ห้ามส่งออกทาง API**

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
from typing import Iterable, Optional

from pydantic import BaseModel, Field


class Visibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"


class AclError(ValueError):
    """ความผิดพลาดของการตั้งค่าสิทธิ์ (ไม่ใช่การปฏิเสธการเข้าถึง)"""


class AclWideningError(AclError):
    """สิทธิ์ระดับไฟล์พยายามทำให้กว้างกว่า box (กฎเหล็กข้อ 9)"""


class AclUnknownBoxError(AclError):
    """ลงทะเบียนไฟล์เข้า box ที่ยังไม่รู้จัก"""


class Box(BaseModel):
    """กล่องเอกสาร — หน่วยที่ผู้ใช้เลือกตอนถาม และเป็นเจ้าของ Public/Private

    frozen: visibility เปลี่ยนไม่ได้หลังสร้าง (กฎเหล็กข้อ 9) การเปลี่ยนกล่องคือการสร้างกล่องใหม่
    """

    model_config = {"frozen": True}

    box_id: str = Field(min_length=1)
    visibility: Visibility
    departments: Optional[frozenset[str]] = None  # None = ไม่จำกัด; set() = ไม่มีใคร
    min_security_level: int = Field(default=0, ge=0)


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
    box_ids: frozenset[str] = frozenset()  # box ที่เป็นสมาชิก


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


class AclStore:
    """แหล่งความจริงของสิทธิ์ (ด่านที่ 2 ของ PLAN.md ข้อ 8)

    vector store ใช้ payload เป็นด่านที่ 1 (pre-filter) แต่ payload อาจค้างเก่าได้หลังสิทธิ์เปลี่ยน
    ด่านที่ 2 จึงถามที่นี่ทุกครั้งก่อนคืนผล — ``app/vectorstore.py::VectorStore.search``

    ยังเก็บในหน่วยความจำ: ของจริงจะเป็นฐานข้อมูลหรือ API ขององค์กร (คำถามค้างใน PLAN.md ข้อ 9)
    ``register_file`` ตรวจการขยายสิทธิ์ทันทีเพื่อให้ config ที่ผิดดังตอนตั้ง ไม่ใช่ตอนค้น
    """

    def __init__(self) -> None:
        self._boxes: dict[str, Box] = {}
        self._files: dict[str, FileAcl] = {}
        self._effective: dict[str, EffectiveAcl] = {}

    # ---------- ลงทะเบียน ----------

    def register_box(self, box: Box) -> None:
        existing = self._boxes.get(box.box_id)
        if existing is not None and existing.visibility != box.visibility:
            raise AclError(
                f"box {box.box_id} เปลี่ยน visibility จาก {existing.visibility.value} "
                f"เป็น {box.visibility.value} ไม่ได้ (กฎเหล็กข้อ 9)"
            )
        self._boxes[box.box_id] = box
        for file in self._files.values():  # box เปลี่ยน -> คำนวณ effective ของไฟล์ใน box นั้นใหม่
            if file.box_id == box.box_id:
                self._effective[file.file_id] = effective_acl(box, file)

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
        return evaluate(principal, acl)

    def can_access(self, principal: Principal, file_id: str) -> bool:
        """คำตอบที่ปลอดภัยสำหรับทุกที่ที่ไม่ใช่ test/log — ไม่บอกเหตุผล"""
        return self.decide(principal, file_id).allowed

    def accessible_files(self, principal: Principal, file_ids: Iterable[str]) -> list[str]:
        return [fid for fid in file_ids if self.can_access(principal, fid)]
