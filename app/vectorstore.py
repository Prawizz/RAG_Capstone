"""Qdrant: สร้าง collection, upsert unit, ค้นแบบกรองสิทธิ์

**สิทธิ์ 2 ด่าน** (PLAN.md ข้อ 8)
- ด่าน 1 `pre-filter ใน Qdrant` — กรองจาก payload ตอนค้น เร็วแต่ payload อาจค้างเก่าหลังสิทธิ์เปลี่ยน
- ด่าน 2 `re-check กับ ACL store` — ทุก hit ที่ได้ถูกถาม ``AclStore`` ซ้ำก่อนคืนออกไป
  hit ที่ตกด่าน 2 ถูกทิ้งและนับไว้ใน ``SearchResult.dropped_by_recheck`` (> 0 แปลว่า index ค้างเก่า ต้อง re-index)

**Public / Private แยก collection** — ``visibility`` มาจาก ``Box`` ใน ``AclStore`` เท่านั้น ไม่เคยมาจาก input ของผู้ใช้
ผู้ใช้จึงขอค้น collection ของ private โดยอ้างชื่อ box ของ public ไม่ได้

**index เฉพาะ unit ที่ status = ok** — unit ที่เป็น needs_ocr / failed มีข้อความที่อ่านไม่ได้หรือพัง
ถ้า index ไว้ ระบบจะตอบจากข้อความพัง (CLAUDE.md ข้อ 5) ``upsert_units`` ข้ามให้เองและรายงานจำนวนที่ข้าม

**ไฟล์ที่ไม่รู้สิทธิ์ index ไม่ได้** — ``upsert_units`` ต้องหา ``EffectiveAcl`` ของ ``file_id`` ใน ``AclStore`` ได้
ไม่งั้นโยน ``AclUnknownFileError`` (fail closed: เนื้อหาที่ไม่รู้ว่าใครเห็นได้ ห้ามเข้า index)

**ห้ามส่งออกทาง API** — ``SearchResult.dropped_by_recheck`` เป็นตัวเลขสำหรับ ops/log เท่านั้น
เหมือน ``AccessDecision.denied_layer`` เพราะกรณีไม่มีสิทธิ์ต้องแยกจากกรณีหาไม่เจอไม่ได้ (กฎเหล็กข้อ 9)
``search`` ของ box ที่ไม่มีสิทธิ์คืน ``EMPTY_RESULT`` ซึ่งเท่ากับผลของการค้นที่ไม่เจออะไรทุกฟิลด์

**เวกเตอร์มาจากข้างนอก** — โมดูลนี้ไม่ embed เอง (``app/embedding.py`` ยังไม่มีโมเดลจริง)
``upsert_units`` รับ ``vectors`` ที่ map จาก ``Unit.index`` ตรวจแค่ขนาดมิติให้ตรง ``Settings.vector_size``
เปลี่ยนโมเดล embedding ที่มิติต่างไป = ต้องสร้าง collection ใหม่และ index ซ้ำ

point id = UUID5 ของ ``"<file_id>:<unit_index>"`` จึงคงที่ — ingest ไฟล์เดิมซ้ำเขียนทับ ไม่เกิด point ซ้ำ
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from qdrant_client import QdrantClient, models

from app.access import AclError, AclStore, Principal, Visibility
from app.config import Settings, get_settings
from app.models import Document, Unit, UnitStatus

# namespace คงที่ของ point id (เปลี่ยนค่านี้ = point id ทั้งระบบเปลี่ยน ต้อง re-index)
POINT_NAMESPACE = uuid.UUID("6f6b2c14-5f5a-4f2e-9a1a-0b7a2d9c4e31")

# ---------- ชื่อ field ใน payload (แก้ที่นี่ที่เดียว; เปลี่ยนแล้วต้อง re-index) ----------
P_FILE_ID = "file_id"
P_FILENAME = "filename"
P_BOX_ID = "box_id"
P_VISIBILITY = "visibility"
P_UNIT_INDEX = "unit_index"
P_UNIT_TYPE = "unit_type"
P_POSITION = "position"
P_LABEL = "label"
P_STATUS = "status"
P_METHOD = "method"
P_TEXT = "text"
P_ACL_DEPARTMENTS = "acl_departments"
P_ACL_ANY_DEPARTMENT = "acl_any_department"  # true = box/ไฟล์ไม่จำกัดแผนก (เลี่ยงความหมายของ null ใน filter)
P_ACL_MIN_LEVEL = "acl_min_security_level"

# field ที่ต้องมี index บน Qdrant จริง (local mode ไม่รองรับ payload index — ใช้ docker ตอนวัดของจริง)
INDEXED_FIELDS: tuple[tuple[str, models.PayloadSchemaType], ...] = (
    (P_FILE_ID, models.PayloadSchemaType.KEYWORD),
    (P_BOX_ID, models.PayloadSchemaType.KEYWORD),
    (P_STATUS, models.PayloadSchemaType.KEYWORD),
    (P_ACL_DEPARTMENTS, models.PayloadSchemaType.KEYWORD),
    (P_ACL_ANY_DEPARTMENT, models.PayloadSchemaType.BOOL),
    (P_ACL_MIN_LEVEL, models.PayloadSchemaType.INTEGER),
)


class AclUnknownFileError(AclError):
    """ไฟล์ยังไม่มีสิทธิ์ลงทะเบียนใน ``AclStore`` จึง index ไม่ได้"""


class VectorSizeError(ValueError):
    """ขนาดเวกเตอร์ไม่ตรงกับ ``Settings.vector_size``"""


@dataclass(frozen=True)
class Hit:
    """ผลค้นหนึ่งรายการ — ฟิลด์อ้างอิงมาจาก contract ใน ``app/models.py``"""

    score: float
    file_id: str
    filename: str
    unit_index: int
    unit_type: str
    label: str
    position: dict[str, Any]
    status: str
    method: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    hits: tuple[Hit, ...] = ()
    dropped_by_recheck: int = 0  # ห้ามส่งออกทาง API (ดู docstring ของโมดูล)


# ผลของ "ไม่มีสิทธิ์" และ "ไม่เจออะไร" ต้องเป็นสิ่งเดียวกันทุกฟิลด์ (กฎเหล็กข้อ 9)
EMPTY_RESULT = SearchResult()


@dataclass(frozen=True)
class UpsertReport:
    file_id: str
    collection: str
    indexed: int
    skipped_not_ok: int
    skipped_empty: int


def point_id(file_id: str, unit_index: int) -> str:
    """id คงที่ของ unit หนึ่ง — ingest ซ้ำแล้วเขียนทับ ไม่เกิด point ซ้ำ"""
    return str(uuid.uuid5(POINT_NAMESPACE, f"{file_id}:{unit_index}"))


class VectorStore:
    def __init__(
        self,
        client: QdrantClient,
        acl: AclStore,
        settings: Optional[Settings] = None,
        *,
        supports_payload_indexes: bool = True,
    ) -> None:
        self.client = client
        self.acl = acl
        self.settings = settings or get_settings()
        self._supports_payload_indexes = supports_payload_indexes

    # ---------- การสร้าง ----------

    @classmethod
    def in_memory(cls, acl: AclStore, settings: Optional[Settings] = None) -> "VectorStore":
        """Qdrant ในกระบวนการ (ใช้ใน test) — semantics ของ filter เหมือนของจริง แต่ไม่มี payload index"""
        store = cls(QdrantClient(":memory:"), acl, settings, supports_payload_indexes=False)
        store.ensure_collections()
        return store

    @classmethod
    def connect(cls, acl: AclStore, settings: Optional[Settings] = None) -> "VectorStore":
        settings = settings or get_settings()
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=int(settings.qdrant_timeout),
        )
        return cls(client, acl, settings)

    def collection_for(self, visibility: Visibility) -> str:
        return (
            self.settings.collection_private
            if visibility == Visibility.PRIVATE
            else self.settings.collection_public
        )

    @property
    def collections(self) -> tuple[str, str]:
        return (self.settings.collection_public, self.settings.collection_private)

    def ensure_collections(self) -> None:
        """สร้าง collection ที่ยังไม่มี (เรียกซ้ำได้)"""
        for name in self.collections:
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    name,
                    vectors_config=models.VectorParams(
                        size=self.settings.vector_size, distance=models.Distance.COSINE
                    ),
                )
            if self._supports_payload_indexes:
                for field, schema in INDEXED_FIELDS:
                    self.client.create_payload_index(name, field, field_schema=schema)

    # ---------- เขียน ----------

    def upsert_units(
        self,
        document: Document,
        vectors: Mapping[int, Sequence[float]],
        units: Optional[Sequence[Unit]] = None,
    ) -> UpsertReport:
        """index unit ที่ status = ok ของไฟล์หนึ่ง (``vectors`` map จาก ``Unit.index``)

        Raises:
            AclUnknownFileError: ``document.file_id`` ยังไม่ได้ลงทะเบียนสิทธิ์
            VectorSizeError: มิติของเวกเตอร์ไม่ตรงค่าตั้ง
            KeyError: unit ที่ต้อง index แต่ไม่มีเวกเตอร์ให้
        """
        acl = self.acl.get_effective(document.file_id)
        if acl is None:
            raise AclUnknownFileError(
                f"ไฟล์ {document.file_id} ยังไม่มีสิทธิ์ใน AclStore — ลงทะเบียน FileAcl ก่อน index"
            )
        collection = self.collection_for(acl.visibility)
        acl_payload = {
            P_BOX_ID: acl.box_id,
            P_VISIBILITY: acl.visibility.value,
            P_ACL_DEPARTMENTS: acl.department_list or [],
            P_ACL_ANY_DEPARTMENT: acl.departments is None,
            P_ACL_MIN_LEVEL: acl.min_security_level,
        }

        points: list[models.PointStruct] = []
        skipped_not_ok = skipped_empty = 0
        for unit in units if units is not None else document.units:
            if unit.status != UnitStatus.OK:
                skipped_not_ok += 1
                continue
            if not unit.text.strip():  # unit ที่ ok แต่ว่าง (flag empty) ไม่มีอะไรให้ค้น
                skipped_empty += 1
                continue
            vector = vectors[unit.index]
            if len(vector) != self.settings.vector_size:
                raise VectorSizeError(
                    f"unit {unit.index} ของ {document.file_id}: เวกเตอร์ {len(vector)} มิติ "
                    f"แต่ค่าตั้งคือ {self.settings.vector_size}"
                )
            points.append(
                models.PointStruct(
                    id=point_id(document.file_id, unit.index),
                    vector=list(vector),
                    payload={
                        P_FILE_ID: document.file_id,
                        P_FILENAME: document.filename,
                        P_UNIT_INDEX: unit.index,
                        P_UNIT_TYPE: unit.type.value,
                        P_POSITION: unit.position,
                        P_LABEL: unit.label,
                        P_STATUS: unit.status.value,
                        P_METHOD: unit.method.value,
                        P_TEXT: unit.text,
                        **acl_payload,
                    },
                )
            )
        if points:
            self.client.upsert(collection, points=points, wait=True)
        return UpsertReport(
            file_id=document.file_id,
            collection=collection,
            indexed=len(points),
            skipped_not_ok=skipped_not_ok,
            skipped_empty=skipped_empty,
        )

    def delete_file(self, file_id: str) -> None:
        """ลบทุก point ของไฟล์ — ลบจากทั้งสอง collection เพราะสิทธิ์อาจถูกถอนไปแล้ว (เรียกซ้ำได้)"""
        condition = models.Filter(
            must=[models.FieldCondition(key=P_FILE_ID, match=models.MatchValue(value=file_id))]
        )
        for name in self.collections:
            self.client.delete(
                name, points_selector=models.FilterSelector(filter=condition), wait=True
            )

    # ---------- ค้น ----------

    def _acl_filter(self, principal: Principal, box_id: str) -> models.Filter:
        """ด่าน 1: filter จาก payload — box ที่ขอ + แผนก + ระดับ + เฉพาะ unit ok"""
        department_clause = models.Filter(
            should=[
                models.FieldCondition(
                    key=P_ACL_ANY_DEPARTMENT, match=models.MatchValue(value=True)
                ),
                models.FieldCondition(
                    key=P_ACL_DEPARTMENTS,
                    match=models.MatchAny(any=sorted(principal.departments)),
                ),
            ]
        )
        return models.Filter(
            must=[
                models.FieldCondition(key=P_BOX_ID, match=models.MatchValue(value=box_id)),
                models.FieldCondition(
                    key=P_STATUS, match=models.MatchValue(value=UnitStatus.OK.value)
                ),
                models.FieldCondition(
                    key=P_ACL_MIN_LEVEL,
                    range=models.Range(lte=float(principal.security_level)),
                ),
                department_clause,
            ]
        )

    def search(
        self,
        query_vector: Sequence[float],
        principal: Principal,
        box_id: str,
        limit: int = 10,
    ) -> SearchResult:
        """ค้นใน box เดียว (PLAN.md: retrieve รอบจริงทำในกล่องเดียวเสมอ) แล้ว re-check ทุก hit

        box ที่ไม่รู้จัก / ไม่ได้เป็นสมาชิก คืน ``EMPTY_RESULT`` เหมือนค้นแล้วไม่เจอ
        """
        box = self.acl.get_box(box_id)
        if box is None or box_id not in principal.box_ids:
            return EMPTY_RESULT  # ไม่มีสิทธิ์ = หาไม่เจอ ทุกฟิลด์เหมือนกัน
        if len(query_vector) != self.settings.vector_size:
            raise VectorSizeError(
                f"เวกเตอร์คำถาม {len(query_vector)} มิติ แต่ค่าตั้งคือ {self.settings.vector_size}"
            )

        found = self.client.query_points(
            self.collection_for(box.visibility),
            query=list(query_vector),
            query_filter=self._acl_filter(principal, box_id),
            limit=limit,
            with_payload=True,
        ).points

        hits: list[Hit] = []
        dropped = 0
        for point in found:
            payload = point.payload or {}
            file_id = payload.get(P_FILE_ID, "")
            if not self.acl.can_access(principal, file_id):  # ด่าน 2
                dropped += 1
                continue
            hits.append(
                Hit(
                    score=point.score,
                    file_id=file_id,
                    filename=payload.get(P_FILENAME, ""),
                    unit_index=payload.get(P_UNIT_INDEX, -1),
                    unit_type=payload.get(P_UNIT_TYPE, ""),
                    label=payload.get(P_LABEL, ""),
                    position=payload.get(P_POSITION, {}),
                    status=payload.get(P_STATUS, ""),
                    method=payload.get(P_METHOD, ""),
                    text=payload.get(P_TEXT, ""),
                )
            )
        return SearchResult(hits=tuple(hits), dropped_by_recheck=dropped)
