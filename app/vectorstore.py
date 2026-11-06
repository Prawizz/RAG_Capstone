"""Qdrant: สร้าง collection, upsert unit, ค้นแบบกรองสิทธิ์

**สิทธิ์ 2 ด่าน** (PLAN.md ข้อ 8)
- ด่าน 1 `pre-filter ใน Qdrant` — กรองจาก payload ตอนค้น เร็วแต่ payload อาจค้างเก่าหลังสิทธิ์เปลี่ยน
- ด่าน 2 `re-check กับ ACL store` — ทุก hit ที่ได้ถูกถาม ``AclStore`` ซ้ำก่อนคืนออกไป ต้องผ่านครบ 3 ข้อ:
  ``can_access`` · ``effective.box_id`` = box ที่ค้น · ``effective.visibility`` = visibility ของ collection ที่ค้น
  (สองข้อหลังกันกรณีไฟล์ถูกย้าย box/visibility หลัง index โดยไม่ re-index เนื้อหา Private จะได้ไม่โผล่ในการค้นกล่อง Public)
  hit ที่ตกด่าน 2 ถูกทิ้งและนับไว้ใน ``SearchResult.dropped_by_recheck`` (> 0 แปลว่า index ค้างเก่า ต้อง re-index)
  ``search`` ดึงผลเกิน ``limit`` ไว้ (``fetch_limit``) เผื่อด่าน 2 ตัดทิ้ง แล้วคืนไม่เกิน ``limit``

**สิทธิ์เปลี่ยนหลัง index** — ``refresh_acl_payload(file_id)`` อัปเดต payload ของ ACL ด้วย ``set_payload`` โดยไม่ embed ใหม่
ถ้า visibility เปลี่ยน (point อยู่ผิด collection) จะลบ point ออกจาก collection เดิมและรายงาน ``needs_reindex`` —
ไม่ย้าย point ข้าม collection เงียบๆ เพราะต้องมีผู้ re-index ด้วยเวกเตอร์/ข้อความที่ถูกต้อง

**Public / Private แยก collection** — ``visibility`` มาจาก ``Box`` ใน ``AclStore`` เท่านั้น ไม่เคยมาจาก input ของผู้ใช้
ผู้ใช้จึงขอค้น collection ของ private โดยอ้างชื่อ box ของ public ไม่ได้

**index เฉพาะ unit ที่ status = ok** — unit ที่เป็น needs_ocr / failed มีข้อความที่อ่านไม่ได้หรือพัง
ถ้า index ไว้ ระบบจะตอบจากข้อความพัง (CLAUDE.md ข้อ 5) ``upsert_document`` ข้ามให้เองและรายงานจำนวนที่ข้าม

**ไฟล์ที่ไม่รู้สิทธิ์ index ไม่ได้** — ``upsert_chunks`` ต้องหา ``EffectiveAcl`` ของ ``file_id`` ใน ``AclStore`` ได้
ไม่งั้นโยน ``AclUnknownFileError`` (fail closed: เนื้อหาที่ไม่รู้ว่าใครเห็นได้ ห้ามเข้า index)

**ห้ามส่งออกทาง API** — ``SearchResult.dropped_by_recheck`` เป็นตัวเลขสำหรับ ops/log เท่านั้น
เหมือน ``AccessDecision.denied_layer`` เพราะกรณีไม่มีสิทธิ์ต้องแยกจากกรณีหาไม่เจอไม่ได้ (กฎเหล็กข้อ 9)
``search`` ของ box ที่ไม่มีสิทธิ์คืน ``EMPTY_RESULT`` ซึ่งเท่ากับผลของการค้นที่ไม่เจออะไรทุกฟิลด์

**1 point = 1 chunk** (``app/chunking.py::Chunk``) ไม่ใช่ 1 unit — chunk ตามขนาด ``Settings.chunk_target_chars`` ตัดตามขอบประโยค ไม่ข้าม unit
payload เก็บตำแหน่งประโยคแต่ละประโยค (``sentences`` = char_start/char_end ใน ``Unit.text``) ไว้ cite ระดับประโยค

**named vectors 3 ตัวใน collection เดียว** — ``dense`` (BGE-M3 1024, cosine) · ``sparse_bge`` (lexical weights ของ BGE-M3) ·
``sparse_bm25`` (ตัดคำ pythainlp ฝั่ง client, ค่า = tf, ``Modifier.IDF`` ให้ Qdrant คูณ IDF) — schema เปลี่ยน = ต้องสร้าง collection ใหม่
``search`` เลือกโหมดได้ (``SEARCH_MODES``): dense · dense+sparse_bge · dense+sparse_bm25 · all — โหมดที่มี sparse รวมผลด้วย RRF ของ Qdrant
(query API + prefetch) ทุกโหมดใช้ ACL filter ชุดเดียวกัน (ใส่ในทุก prefetch) และผ่านด่าน 2 ชุดเดียวกัน

**เวกเตอร์มาจากข้างนอก** — โมดูลนี้ไม่ embed ใน ``upsert_chunks`` (ผู้เรียกใช้ ``embed_chunks`` กับ ``app/embedding.py``)
ตรวจแค่ขนาดมิติ dense ให้ตรง ``Settings.vector_size`` เปลี่ยนโมเดลที่มิติต่างไป = ต้องสร้าง collection ใหม่และ index ซ้ำ

point id = UUID5 ของ ``"<file_id>:<unit_index>:<chunk_index>"`` และ ``upsert_chunks`` **ลบ point เก่าของไฟล์ทั้งสอง collection ก่อนเสมอ**
— ingest ซ้ำไม่ทิ้ง chunk ค้างเมื่อจำนวน chunk ลดลง (เขียนทับด้วย id เดิมอย่างเดียวไม่พอ)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from qdrant_client import QdrantClient, models

from app.access import AclError, AclStore, Box, EffectiveAcl, Principal, Visibility
from app.chunking import Chunk, make_document_chunks
from app.config import Settings, get_settings
from app.embedding import Embedded, Embedder, SparseVector, bm25_sparse
from app.models import Document, Unit, UnitStatus

# namespace คงที่ของ point id (เปลี่ยนค่านี้ = point id ทั้งระบบเปลี่ยน ต้อง re-index)
POINT_NAMESPACE = uuid.UUID("6f6b2c14-5f5a-4f2e-9a1a-0b7a2d9c4e31")

# ---------- ชื่อ field ใน payload (แก้ที่นี่ที่เดียว; เปลี่ยนแล้วต้อง re-index) ----------
P_FILE_ID = "file_id"
P_FILENAME = "filename"
P_BOX_ID = "box_id"
P_VISIBILITY = "visibility"
P_UNIT_INDEX = "unit_index"
P_CHUNK_INDEX = "chunk_index"
P_CHAR_START = "char_start"  # ขอบ chunk ใน Unit.text ดั้งเดิม
P_CHAR_END = "char_end"
P_SENTENCES = "sentences"  # [{"char_start", "char_end"}] ของแต่ละประโยคใน chunk (offset ใน Unit.text ดั้งเดิม)
P_CONTEXT = "context"  # context line ของ unit ที่แปะหน้า chunk ตอน embed (ไม่อยู่ใน text / offset)
P_UNIT_TYPE = "unit_type"
P_POSITION = "position"
P_LABEL = "label"
P_STATUS = "status"
P_METHOD = "method"
P_TEXT = "text"
P_ACL_DEPARTMENTS = "acl_departments"
P_ACL_ANY_DEPARTMENT = "acl_any_department"  # true = box/ไฟล์ไม่จำกัดแผนก (เลี่ยงความหมายของ null ใน filter)
P_ACL_MIN_LEVEL = "acl_min_security_level"

# named vectors ของ collection
V_DENSE = "dense"
V_SPARSE_BGE = "sparse_bge"
V_SPARSE_BM25 = "sparse_bm25"

# โหมดค้น -> vector ที่ใช้ (dense เสมอ; โหมดที่มี sparse รวมผลด้วย RRF)
SEARCH_MODES: dict[str, tuple[str, ...]] = {
    "dense": (V_DENSE,),
    "dense+sparse_bge": (V_DENSE, V_SPARSE_BGE),
    "dense+sparse_bm25": (V_DENSE, V_SPARSE_BM25),
    "all": (V_DENSE, V_SPARSE_BGE, V_SPARSE_BM25),
}
PREFETCH_LIMIT = 50  # จำนวน candidate ต่อ vector ก่อน RRF (ไม่น้อยกว่าจำนวนที่ดึงสุดท้าย)
UPSERT_BATCH = 64

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


class CollectionSchemaError(RuntimeError):
    """collection ที่มีอยู่แล้วไม่ใช่ schema named vectors ปัจจุบัน — ต้องสร้างใหม่แล้ว index ซ้ำ (ไม่แก้ให้เงียบๆ)"""


class SearchModeError(ValueError):
    """โหมดค้นไม่รู้จัก หรือ query ไม่มี vector ที่โหมดนั้นต้องใช้"""


@dataclass(frozen=True)
class Query:
    """คำถามที่ embed แล้ว: dense เสมอ; sparse ใช้เฉพาะโหมดที่เรียก"""

    dense: Sequence[float]
    sparse_bge: SparseVector = SparseVector()
    sparse_bm25: SparseVector = SparseVector()


@dataclass(frozen=True)
class IndexedChunk:
    """chunk พร้อมเวกเตอร์ทั้ง 3 ตัวที่จะเขียนลง Qdrant"""

    chunk: Chunk
    dense: Sequence[float]
    sparse_bge: SparseVector
    sparse_bm25: SparseVector


def embed_chunks(chunks: Sequence[Chunk], embedder: Embedder) -> list[IndexedChunk]:
    """embed ``Chunk.embed_text`` (context line + เนื้อ) -> dense + sparse_bge จาก embedder, sparse_bm25 จากตัดคำฝั่ง client"""
    texts = [c.embed_text for c in chunks]
    return [IndexedChunk(c, e.dense, e.sparse, bm25_sparse(t)) for c, t, e in zip(chunks, texts, embedder.embed_documents(texts))]


def embed_query(text: str, embedder: Embedder) -> Query:
    e: Embedded = embedder.embed_query(text)
    return Query(dense=e.dense, sparse_bge=e.sparse, sparse_bm25=bm25_sparse(text))


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
    chunk_index: int = 0
    char_start: int = 0
    char_end: int = 0
    sentences: tuple[tuple[int, int], ...] = ()  # (char_start, char_end) ของแต่ละประโยคใน chunk
    context: str = ""


@dataclass(frozen=True)
class SearchResult:
    hits: tuple[Hit, ...] = ()
    dropped_by_recheck: int = 0  # ห้ามส่งออกทาง API (ดู docstring ของโมดูล)


MAX_FETCH = 50  # เพดานจำนวน point ที่ดึงต่อการค้น (ดึงเกิน limit ไว้เผื่อด่าน 2 ตัดทิ้ง)


def fetch_limit(limit: int) -> int:
    """จำนวน point ที่ดึงจาก Qdrant: limit*2 แต่ไม่เกิน ``MAX_FETCH`` และไม่น้อยกว่า ``limit`` เอง"""
    return max(limit, min(limit * 2, MAX_FETCH))


# ผลของ "ไม่มีสิทธิ์" และ "ไม่เจออะไร" ต้องเป็นสิ่งเดียวกันทุกฟิลด์ (กฎเหล็กข้อ 9)
EMPTY_RESULT = SearchResult()


@dataclass(frozen=True)
class UpsertReport:
    file_id: str
    collection: str
    indexed: int  # จำนวน chunk (point) ที่เขียน
    skipped_not_ok: int
    skipped_empty: int


@dataclass(frozen=True)
class RefreshReport:
    """ผล ``refresh_acl_payload``: ``needs_reindex`` = มี point อยู่ผิด collection (visibility เปลี่ยน) ถูกลบแล้ว ต้อง index ใหม่"""

    file_id: str
    collection: str  # collection ที่ถูกต้องตาม visibility ปัจจุบัน
    updated: int  # จำนวน point ใน collection ที่ถูกต้องที่ payload ถูกเขียนทับ
    removed: int  # จำนวน point ที่ลบจาก collection เดิม
    removed_from: tuple[str, ...] = ()
    needs_reindex: bool = False


def _acl_payload(acl: EffectiveAcl) -> dict[str, Any]:
    return {
        P_BOX_ID: acl.box_id,
        P_VISIBILITY: acl.visibility.value,
        P_ACL_DEPARTMENTS: acl.department_list or [],
        P_ACL_ANY_DEPARTMENT: acl.departments is None,
        P_ACL_MIN_LEVEL: acl.min_security_level,
    }


def _file_filter(file_id: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key=P_FILE_ID, match=models.MatchValue(value=file_id))]
    )


def point_id(file_id: str, unit_index: int, chunk_index: int) -> str:
    """id คงที่ของ chunk หนึ่ง"""
    return str(uuid.uuid5(POINT_NAMESPACE, f"{file_id}:{unit_index}:{chunk_index}"))


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

    def _collection_config(self) -> dict[str, Any]:
        return dict(
            vectors_config={
                V_DENSE: models.VectorParams(size=self.settings.vector_size, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                V_SPARSE_BGE: models.SparseVectorParams(),
                V_SPARSE_BM25: models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
        )

    def _check_schema(self, name: str) -> None:
        info = self.client.get_collection(name).config.params
        vectors = info.vectors if isinstance(info.vectors, dict) else {}
        sparse = info.sparse_vectors or {}
        bm25 = sparse.get(V_SPARSE_BM25)
        if (
            V_DENSE not in vectors
            or vectors[V_DENSE].size != self.settings.vector_size
            or V_SPARSE_BGE not in sparse
            or bm25 is None
            or bm25.modifier != models.Modifier.IDF
        ):
            raise CollectionSchemaError(
                f"collection {name} มีอยู่แล้วแต่ schema ไม่ตรง (ต้องมี named vectors {V_DENSE}[{self.settings.vector_size}], "
                f"{V_SPARSE_BGE}, {V_SPARSE_BM25}+IDF) — ลบ/สร้างใหม่แล้ว index ซ้ำ"
            )

    def ensure_collections(self) -> None:
        """สร้าง collection ที่ยังไม่มี (เรียกซ้ำได้); ที่มีอยู่แล้วต้องมี schema ตรง ไม่งั้น ``CollectionSchemaError``"""
        for name in self.collections:
            if not self.client.collection_exists(name):
                self.client.create_collection(name, **self._collection_config())
            else:
                self._check_schema(name)
            if self._supports_payload_indexes:
                for field, schema in INDEXED_FIELDS:
                    self.client.create_payload_index(name, field, field_schema=schema)

    # ---------- เขียน ----------

    def upsert_chunks(self, document: Document, indexed: Sequence[IndexedChunk]) -> UpsertReport:
        """เขียน chunk ของไฟล์หนึ่ง: **ลบ point เก่าของไฟล์ทั้งสอง collection ก่อนเสมอ** แล้วค่อยเขียนชุดใหม่

        ลบก่อนเพราะ point id ขึ้นกับ chunk_index — ถ้าจำนวน chunk ลดลงหรือสิทธิ์ย้าย collection การเขียนทับด้วย id เดิมอย่างเดียว
        จะทิ้ง chunk เก่าไว้ (เนื้อหาที่ไม่มีแล้วยังค้นเจอ) ถ้าเขียนชุดใหม่ล้มกลางทาง ไฟล์นี้ไม่มี point เลย (ปลอดภัยกว่าค้าง)

        Raises:
            AclUnknownFileError: ``document.file_id`` ยังไม่ได้ลงทะเบียนสิทธิ์ (ตรวจก่อนลบ — ไม่แตะอะไร)
            VectorSizeError: มิติ dense ไม่ตรงค่าตั้ง (ตรวจก่อนลบ)
            KeyError: chunk อ้าง unit ที่ไม่มีใน document; ValueError: unit ของ chunk ไม่ใช่ ok
        """
        acl = self.acl.get_effective(document.file_id)
        if acl is None:
            raise AclUnknownFileError(
                f"ไฟล์ {document.file_id} ยังไม่มีสิทธิ์ใน AclStore — ลงทะเบียน FileAcl ก่อน index"
            )
        collection = self.collection_for(acl.visibility)
        acl_payload = _acl_payload(acl)
        units = {u.index: u for u in document.units}

        points: list[models.PointStruct] = []
        for item in indexed:
            c = item.chunk
            unit = units[c.unit_index]
            if unit.status != UnitStatus.OK:
                raise ValueError(f"chunk ของ unit {unit.index} ({unit.status.value}) — index เฉพาะ unit ที่ ok")
            if len(item.dense) != self.settings.vector_size:
                raise VectorSizeError(
                    f"chunk {c.unit_index}:{c.chunk_index} ของ {document.file_id}: เวกเตอร์ {len(item.dense)} มิติ "
                    f"แต่ค่าตั้งคือ {self.settings.vector_size}"
                )
            vector: dict[str, Any] = {V_DENSE: list(item.dense)}
            for name, sparse in ((V_SPARSE_BGE, item.sparse_bge), (V_SPARSE_BM25, item.sparse_bm25)):
                if sparse:
                    vector[name] = models.SparseVector(indices=list(sparse.indices), values=list(sparse.values))
            points.append(
                models.PointStruct(
                    id=point_id(document.file_id, c.unit_index, c.chunk_index),
                    vector=vector,
                    payload={
                        P_FILE_ID: document.file_id,
                        P_FILENAME: document.filename,
                        P_UNIT_INDEX: c.unit_index,
                        P_CHUNK_INDEX: c.chunk_index,
                        P_UNIT_TYPE: unit.type.value,
                        P_POSITION: unit.position,
                        P_LABEL: unit.label,
                        P_STATUS: unit.status.value,
                        P_METHOD: unit.method.value,
                        P_TEXT: c.text,
                        P_CHAR_START: c.char_start,
                        P_CHAR_END: c.char_end,
                        P_SENTENCES: [{P_CHAR_START: s.char_start, P_CHAR_END: s.char_end} for s in c.sentences],
                        P_CONTEXT: c.context,
                        **acl_payload,
                    },
                )
            )
        self.delete_file(document.file_id)
        for start in range(0, len(points), UPSERT_BATCH):
            self.client.upsert(collection, points=points[start:start + UPSERT_BATCH], wait=True)
        return UpsertReport(file_id=document.file_id, collection=collection, indexed=len(points), skipped_not_ok=0, skipped_empty=0)

    def upsert_document(
        self, document: Document, embedder: Embedder, units: Optional[Sequence[Unit]] = None
    ) -> UpsertReport:
        """chunk (ตาม ``Settings.chunk_target_chars``) + embed + ``upsert_chunks`` — index เฉพาะ unit ที่ status = ok

        unit ที่เป็น needs_ocr / failed มีข้อความที่อ่านไม่ได้หรือพัง ถ้า index ไว้ระบบจะตอบจากข้อความพัง (CLAUDE.md ข้อ 5)
        จึงข้ามและรายงานจำนวนที่ข้าม (ว่างแต่ ok = ``skipped_empty``)
        """
        todo = list(units) if units is not None else list(document.units)
        ok = [u for u in todo if u.status == UnitStatus.OK and u.text.strip()]
        report = self.upsert_chunks(
            document, embed_chunks(make_document_chunks(ok, self.settings.chunk_target_chars), embedder)
        )
        skipped_not_ok = sum(1 for u in todo if u.status != UnitStatus.OK)
        return UpsertReport(
            file_id=report.file_id, collection=report.collection, indexed=report.indexed,
            skipped_not_ok=skipped_not_ok, skipped_empty=len(todo) - len(ok) - skipped_not_ok,
        )

    def delete_file(self, file_id: str) -> None:
        """ลบทุก point ของไฟล์ — ลบจากทั้งสอง collection เพราะสิทธิ์อาจถูกถอนไปแล้ว (เรียกซ้ำได้)"""
        condition = _file_filter(file_id)
        for name in self.collections:
            self.client.delete(
                name, points_selector=models.FilterSelector(filter=condition), wait=True
            )

    def refresh_acl_payload(self, file_id: str) -> RefreshReport:
        """ทำให้ payload ของ ACL ใน index ตรงกับ ``AclStore`` ตอนนี้ (box_id, visibility, acl_*) โดยไม่ embed ใหม่

        - point ใน collection ที่ตรงกับ visibility ปัจจุบัน: ``set_payload`` แบบ filter ตาม ``file_id``
        - point ใน collection อื่น (visibility เปลี่ยน): **ลบทิ้ง** และรายงาน ``needs_reindex`` — ไม่ย้ายข้าม collection เงียบๆ

        Raises:
            AclUnknownFileError: ไฟล์ไม่มีสิทธิ์ใน ``AclStore`` (ไม่เดาว่าจะลบหรือเก็บ; ใช้ ``delete_file`` ถ้าต้องการลบ)
        """
        acl = self.acl.get_effective(file_id)
        if acl is None:
            raise AclUnknownFileError(f"ไฟล์ {file_id} ไม่มีสิทธิ์ใน AclStore — refresh payload ไม่ได้")
        target = self.collection_for(acl.visibility)
        condition = _file_filter(file_id)
        updated = 0
        removed_from: list[str] = []
        removed = 0
        for name in self.collections:
            n = self.client.count(name, count_filter=condition, exact=True).count
            if n == 0:
                continue
            if name == target:
                self.client.set_payload(name, payload=_acl_payload(acl), points=condition, wait=True)
                updated = n
            else:
                self.client.delete(name, points_selector=models.FilterSelector(filter=condition), wait=True)
                removed += n
                removed_from.append(name)
        return RefreshReport(
            file_id=file_id, collection=target, updated=updated, removed=removed,
            removed_from=tuple(removed_from), needs_reindex=bool(removed_from),
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

    def _passes_recheck(self, principal: Principal, file_id: str, box: Box) -> bool:
        """ด่าน 2: ``can_access`` + ไฟล์ยังอยู่ box ที่ค้น + visibility ตรงกับ collection ที่ค้น (ไม่ผ่านข้อใดข้อหนึ่ง = ทิ้ง)"""
        effective = self.acl.get_effective(file_id)
        return (
            effective is not None
            and effective.box_id == box.box_id
            and effective.visibility == box.visibility
            and self.acl.can_access(principal, file_id)
        )

    def search(
        self,
        query: "Query | Sequence[float]",
        principal: Principal,
        box_id: str,
        limit: int = 10,
        mode: Optional[str] = None,
    ) -> SearchResult:
        """ค้นใน box เดียว (PLAN.md: retrieve รอบจริงทำในกล่องเดียวเสมอ) แล้ว re-check ทุก hit

        ``mode`` (ไม่ระบุ = ``Settings.search_mode``) ดู ``SEARCH_MODES`` — ``query`` เป็น ``Query`` (dense + sparse ของคำถาม)
        หรือรายการตัวเลข (= dense อย่างเดียว ใช้ได้เฉพาะโหมด dense) box ที่ไม่รู้จัก / ไม่ได้เป็นสมาชิก คืน ``EMPTY_RESULT``
        เหมือนค้นแล้วไม่เจอ

        Raises:
            SearchModeError: โหมดไม่รู้จัก หรือโหมดต้องใช้ sparse แต่ ``query`` ไม่มี
        """
        mode = mode or self.settings.search_mode
        if mode not in SEARCH_MODES:
            raise SearchModeError(f"ไม่รู้จักโหมดค้น {mode!r} (มี {', '.join(SEARCH_MODES)})")
        q = query if isinstance(query, Query) else Query(dense=query)
        if mode != "dense" and not isinstance(query, Query):
            raise SearchModeError(f"โหมด {mode} ต้องใช้ sparse ของคำถาม — ส่ง Query ที่ embed แล้ว ไม่ใช่เวกเตอร์ dense")
        sparse_of = {V_SPARSE_BGE: q.sparse_bge, V_SPARSE_BM25: q.sparse_bm25}

        box = self.acl.get_box(box_id)
        if box is None or box_id not in principal.box_ids:
            return EMPTY_RESULT  # ไม่มีสิทธิ์ = หาไม่เจอ ทุกฟิลด์เหมือนกัน
        if len(q.dense) != self.settings.vector_size:
            raise VectorSizeError(
                f"เวกเตอร์คำถาม {len(q.dense)} มิติ แต่ค่าตั้งคือ {self.settings.vector_size}"
            )

        acl_filter = self._acl_filter(principal, box_id)
        fetch = fetch_limit(limit)
        collection = self.collection_for(box.visibility)
        # คำถามที่ตัดคำแล้วไม่เหลือ token (sparse ว่าง) = ข้าม vector นั้น (Qdrant ไม่รับ sparse ว่างเป็นคำค้น)
        branches = [v for v in SEARCH_MODES[mode] if v == V_DENSE or sparse_of[v]]
        if branches == [V_DENSE]:
            found = self.client.query_points(
                collection, query=list(q.dense), using=V_DENSE, query_filter=acl_filter,
                limit=fetch, with_payload=True,
            ).points
        else:
            prefetch = [
                models.Prefetch(
                    query=list(q.dense) if v == V_DENSE else models.SparseVector(
                        indices=list(sparse_of[v].indices), values=list(sparse_of[v].values)),
                    using=v, filter=acl_filter, limit=max(fetch, PREFETCH_LIMIT),
                )
                for v in branches
            ]
            found = self.client.query_points(
                collection, prefetch=prefetch, query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=acl_filter, limit=fetch, with_payload=True,
            ).points

        hits: list[Hit] = []
        dropped = 0
        for point in found:
            payload = point.payload or {}
            file_id = payload.get(P_FILE_ID, "")
            if not self._passes_recheck(principal, file_id, box):  # ด่าน 2
                dropped += 1
                continue
            if len(hits) >= limit:  # ดึงเกินไว้เผื่อถูกตัดทิ้ง; ที่เกิน limit ไม่ต้องคืน
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
                    chunk_index=payload.get(P_CHUNK_INDEX, 0),
                    char_start=payload.get(P_CHAR_START, 0),
                    char_end=payload.get(P_CHAR_END, 0),
                    sentences=tuple((x[P_CHAR_START], x[P_CHAR_END]) for x in payload.get(P_SENTENCES, [])),
                    context=payload.get(P_CONTEXT, ""),
                )
            )
        return SearchResult(hits=tuple(hits), dropped_by_recheck=dropped)
