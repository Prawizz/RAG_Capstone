"""API contract v1 (contract freeze) — request / response / error ทุกตัวที่ Gateway เห็น

ผู้ใช้: Enterprise AI Gateway (ถือตัวตนผู้ใช้ งบ คิว และเป็นคน generate คำตอบ) — เราคืนหลักฐาน ไม่ generate
ไฟล์ที่ export: ``contract/openapi.json`` + ``contract/examples/<endpoint>_<case>.json`` (``scripts/export_openapi.py``)

หลักที่ schema นี้บังคับ
- error ทุกตัวเป็น ``ErrorResponse`` {error_code, message, trace_id}; ไม่มี HTTP Forbidden จากเรื่องสิทธิ์ (กฎเหล็กข้อ 9):
  ไม่มีสิทธิ์ = ไม่เจอ (404 หรือผลว่าง) ทุกกรณี
- ``trace_id`` อยู่ในทุก body และ header ``X-Trace-Id``
- ``RetrieveResponse`` ที่ไม่มีผล: ``visibility`` และ ``top_score`` เป็น null เสมอ (validator) — response ของกล่องที่ไม่รู้จัก /
  ไม่ได้เป็นสมาชิก จึงเหมือนกล่องที่เป็นสมาชิกแต่ไม่มีผลทุกฟิลด์ยกเว้น trace_id (visibility ของกล่องที่ไม่เห็นไม่รั่ว)
- ห้ามมี ``denied_layer`` / ``dropped_by_recheck`` / ``box_ids`` ในสัญญานี้ (ตัวตนผู้ใช้มาทาง header, สมาชิกกล่องหาเองจาก AclStore)
- enum ที่ "ขยายได้" (``RetrieveStatus``) — Gateway ต้องรับค่าที่ไม่รู้จักได้ (ถือเป็นไม่มีบริบท) ดู docstring ของ enum

``EXAMPLES`` = ตัวอย่างทุกไฟล์ (ชื่อไฟล์ -> (model, payload)) ใช้ทั้งใน ``json_schema_extra`` ของ model, export และ test
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.access import BoxRole, Visibility
from app.models import DetectedType, DocumentStatus, ErrorCode as DocumentErrorCode
from app.models import ExtractionMethod, QualityFlag, UnitStatus, UnitType

API_VERSION = "1.0.0"
BOX_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$"
QUERY_MAX_CHARS = 2000
TOP_K_DEFAULT, TOP_K_MAX = 5, 20

_TRACE = "3f2a9c1e7b4d4e8f9a0b1c2d3e4f5a6b"
_TRACE_2 = "8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f"
_T0 = "2026-11-12T02:15:04Z"
_T1 = "2026-11-12T02:21:40Z"


def _examples(*payloads: dict) -> ConfigDict:
    return ConfigDict(extra="forbid", json_schema_extra={"examples": list(payloads)})


# ---------------------------------------------------------------------------------------------
# error
# ---------------------------------------------------------------------------------------------

class ApiErrorCode(str, Enum):
    """รหัส error ของ API — คู่กับ HTTP status ตาม ``ERROR_STATUS``"""

    UNAUTHORIZED = "unauthorized"
    INVALID_REQUEST = "invalid_request"
    NOT_FOUND = "not_found"
    TOO_LARGE = "too_large"
    UNSUPPORTED_TYPE = "unsupported_type"
    ACL_WIDENING = "acl_widening"
    UNAVAILABLE = "unavailable"
    INTERNAL = "internal"
    NOT_IMPLEMENTED = "not_implemented"  # เฉพาะช่วง contract freeze (route ยังเป็น stub) — จะถูกถอดออก


ERROR_STATUS: dict[ApiErrorCode, int] = {
    ApiErrorCode.UNAUTHORIZED: 401,
    ApiErrorCode.INVALID_REQUEST: 422,
    ApiErrorCode.NOT_FOUND: 404,
    ApiErrorCode.TOO_LARGE: 413,
    ApiErrorCode.UNSUPPORTED_TYPE: 415,
    ApiErrorCode.ACL_WIDENING: 422,
    ApiErrorCode.UNAVAILABLE: 503,
    ApiErrorCode.INTERNAL: 500,
    ApiErrorCode.NOT_IMPLEMENTED: 501,
}

EX_ERR_UNAUTHORIZED = {"error_code": "unauthorized", "message": "service credential ไม่ถูกต้อง", "trace_id": _TRACE}
EX_ERR_INVALID = {"error_code": "invalid_request",
                  "message": "header X-User-Security-Level: ต้องเป็นจำนวนเต็ม >= 0", "trace_id": _TRACE}
EX_ERR_NOT_FOUND = {"error_code": "not_found", "message": "ไม่พบ", "trace_id": _TRACE}
EX_ERR_TOO_LARGE = {"error_code": "too_large", "message": "ไฟล์ใหญ่เกิน 50 MB", "trace_id": _TRACE}
EX_ERR_UNSUPPORTED = {"error_code": "unsupported_type",
                      "message": "ชนิดไฟล์ (ตรวจจาก magic bytes) ไม่รองรับ: zip", "trace_id": _TRACE}
EX_ERR_ACL_WIDENING = {"error_code": "acl_widening",
                       "message": "file_acl ขยายสิทธิ์แผนกออกนอกกล่อง: ['sales'] (สิทธิ์ระดับไฟล์ทำให้แคบลงได้อย่างเดียว)",
                       "trace_id": _TRACE}
EX_ERR_UNAVAILABLE = {"error_code": "unavailable", "message": "vector store ไม่พร้อม ลองใหม่ภายหลัง", "trace_id": _TRACE}
EX_ERR_INTERNAL = {"error_code": "internal", "message": "เกิดข้อผิดพลาดภายใน (อ้างอิง trace_id)", "trace_id": _TRACE}
EX_ERR_NOT_IMPLEMENTED = {"error_code": "not_implemented",
                          "message": "endpoint นี้อยู่ในช่วง contract freeze ยังไม่เปิดใช้งาน", "trace_id": _TRACE}


class ErrorResponse(BaseModel):
    """รูปแบบ error เดียวของทุก endpoint (ไม่มี HTTP Forbidden: ไม่มีสิทธิ์ = not_found หรือผลว่าง)"""

    model_config = _examples(EX_ERR_NOT_FOUND, EX_ERR_UNAUTHORIZED, EX_ERR_ACL_WIDENING)

    error_code: ApiErrorCode
    message: str = Field(description="ข้อความสำหรับคน (ห้ามเผยว่ามีทรัพยากรที่ไม่มีสิทธิ์อยู่)")
    trace_id: str


# ---------------------------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------------------------

class ReadyStatus(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    NOT_READY = "not_ready"


class ComponentState(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


EX_LIVE = {"status": "ok", "trace_id": _TRACE}
EX_RUNTIME_GPU = {"gpu_available": True, "embedding_requested": "auto", "embedding_device": "cuda",
                  "embedding_dtype": "bf16", "ocr_requested": "auto", "ocr_backend": "typhoon-local", "notes": []}
EX_RUNTIME_CPU = {"gpu_available": False, "embedding_requested": "auto", "embedding_device": "cpu",
                  "embedding_dtype": "fp32", "ocr_requested": "auto", "ocr_backend": "tesseract-local",
                  "notes": ["embedding: auto: ไม่พบ GPU CUDA -> cpu",
                            "ocr: auto: ไม่พบ GPU CUDA -> tesseract-local (CPU)"]}
EX_READY = {"status": "ready", "version": API_VERSION,
            "components": {"qdrant": "ok", "embedding": "ok", "ocr": "ok"}, "runtime": EX_RUNTIME_GPU,
            "trace_id": _TRACE}
EX_READY_CPU = {"status": "ready", "version": API_VERSION,
                "components": {"qdrant": "ok", "embedding": "ok", "ocr": "ok"}, "runtime": EX_RUNTIME_CPU,
                "trace_id": _TRACE}
EX_READY_DEGRADED = {"status": "degraded", "version": API_VERSION,
                     "components": {"qdrant": "ok", "embedding": "ok", "ocr": "down"},
                     "runtime": {**EX_RUNTIME_CPU, "ocr_requested": "typhoon-local", "ocr_backend": "typhoon-local",
                                 "notes": ["embedding: auto: ไม่พบ GPU CUDA -> cpu",
                                           "ocr: ตั้ง typhoon-local แต่ไม่พบ GPU CUDA — ใช้ไม่ได้ (ไม่สลับเป็น tesseract เอง)"]},
                     "trace_id": _TRACE}


class LiveResponse(BaseModel):
    """process ยังทำงาน (ไม่ตรวจของที่พึ่งพา)"""

    model_config = _examples(EX_LIVE)

    status: Literal["ok"] = "ok"
    trace_id: str


class Components(BaseModel):
    model_config = _examples({"qdrant": "ok", "embedding": "ok", "ocr": "down"})

    qdrant: ComponentState
    embedding: ComponentState
    ocr: ComponentState


class Runtime(BaseModel):
    """อุปกรณ์/backend ที่ใช้จริง (ตัดสินที่ app/runtime.py — ตัวเดียวกับที่ embedding/OCR ใช้) · ``notes`` = เหตุผล
    ที่ได้ของด้อยกว่าที่ขอ หรือค่าที่ตั้งไว้ใช้ไม่ได้ (ว่าง = ได้ตามที่ขอบนเครื่องที่มี GPU)"""

    model_config = _examples(EX_RUNTIME_GPU, EX_RUNTIME_CPU)

    gpu_available: bool
    embedding_requested: Literal["auto", "cpu", "cuda"]
    embedding_device: Optional[Literal["cpu", "cuda"]] = Field(description="null = ตั้ง cuda แต่ไม่มี GPU (ใช้ไม่ได้)")
    embedding_dtype: Optional[Literal["fp32", "bf16", "fp16"]] = Field(description="CPU = fp32 เสมอ")
    ocr_requested: Literal["auto", "typhoon-local", "tesseract-local"]
    ocr_backend: Literal["typhoon-local", "tesseract-local"]
    notes: list[str] = Field(default_factory=list)


class ReadyResponse(BaseModel):
    """พร้อมรับงานไหม — HTTP 200 เมื่อ ``ready`` เท่านั้น นอกนั้น 503 (body เป็น model นี้ ไม่ใช่ ErrorResponse)

    - ready = ทุก component ok · not_ready = qdrant หรือ embedding down (ค้นไม่ได้) · degraded = นอกนั้น (เช่น OCR ใช้ไม่ได้:
      ค้นได้ แต่ ingest หน้าที่ต้อง OCR ไม่ได้)
    - embedding ok = อุปกรณ์ที่ตั้งไว้มีจริงและไม่ได้อยู่ในโหมด test (hash vector = degraded) — **ยังไม่ได้โหลดโมเดล**
    - ocr ok = backend ที่เลือกใช้ได้บนเครื่องนี้ (typhoon: มี GPU · tesseract: มีโปรแกรมและไฟล์ภาษา) — ยังไม่ได้โหลดโมเดล
    """

    model_config = _examples(EX_READY, EX_READY_CPU, EX_READY_DEGRADED)

    status: ReadyStatus
    version: str
    components: Components
    runtime: Runtime
    trace_id: str


# ---------------------------------------------------------------------------------------------
# ingest + jobs
# ---------------------------------------------------------------------------------------------

class JobStatus(str, Enum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    OCR = "ocr"
    INDEXING = "indexing"
    DONE = "done"
    DONE_WITH_GAPS = "done_with_gaps"
    FAILED = "failed"


EX_FILE_ACL = {"departments": ["finance"], "min_security_level": 3}


class FileAclInput(BaseModel):
    """สิทธิ์ระดับไฟล์ (ฟอร์มฟิลด์ ``file_acl`` เป็น JSON) — ทำให้แคบลงจากกล่องได้อย่างเดียว กว้างกว่า -> 422 acl_widening

    ``departments``: ไม่ส่ง/null = สืบทอดจากกล่อง · [] = ไม่มีแผนกไหนเห็น (fail closed ตาม app/access.py)
    ``min_security_level``: ไม่ส่ง/null = สืบทอดจากกล่อง · ต้อง >= ของกล่อง
    """

    model_config = _examples(EX_FILE_ACL)

    departments: Optional[list[str]] = None
    min_security_level: Optional[int] = Field(default=None, ge=0)


EX_INGEST_QUEUED = {"job_id": "job_01JD8Z5V7K", "file_id": "file_01JD8Z5V7M", "status": "queued", "duplicate": False,
                    "trace_id": _TRACE}
EX_INGEST_DUPLICATE = {"job_id": "job_01JD7X2A9Q", "file_id": "file_01JD7X2A9R", "status": "done_with_gaps",
                       "duplicate": True, "trace_id": _TRACE}


class IngestResponse(BaseModel):
    """202 = รับงานใหม่ (status queued, duplicate false) · 200 = ไฟล์ซ้ำ (content_hash เดียวกันในกล่องเดียวกัน):
    คืน job/file เดิมพร้อมสถานะปัจจุบัน และ duplicate true"""

    model_config = _examples(EX_INGEST_QUEUED, EX_INGEST_DUPLICATE)

    job_id: str
    file_id: str
    status: JobStatus
    duplicate: bool
    trace_id: str

    @model_validator(mode="after")
    def _new_job_is_queued(self) -> "IngestResponse":
        if not self.duplicate and self.status != JobStatus.QUEUED:
            raise ValueError("งานใหม่ (duplicate=false) ต้องมี status queued")
        return self


class Progress(BaseModel):
    model_config = _examples({"pages_total": 40, "pages_done": 12})

    pages_total: Optional[int] = Field(default=None, ge=0, description="null = ยังไม่รู้ (ยังไม่เปิดไฟล์)")
    pages_done: int = Field(default=0, ge=0)


class UnitCounts(BaseModel):
    model_config = _examples({"ok": 39, "needs_ocr": 0, "failed": 1, "rejected": 0})

    ok: int = Field(default=0, ge=0)
    needs_ocr: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    rejected: int = Field(default=0, ge=0)


class Gap(BaseModel):
    """unit ที่ไม่ ok — ต้องมี error_code หรือ reason อย่างน้อยหนึ่งอย่าง"""

    model_config = _examples({"unit_index": 11, "label": "หน้า 12", "status": "failed", "error_code": None, "reason": "OCR แล้วไม่ผ่านด่านตรวจ: latin_in_thai"})

    unit_index: int = Field(ge=0)
    label: str = Field(description="ตำแหน่งที่คนอ่าน เช่น 'หน้า 12'")
    status: UnitStatus
    error_code: Optional[str] = None
    reason: Optional[str] = None

    @model_validator(mode="after")
    def _explained(self) -> "Gap":
        if self.status == UnitStatus.OK:
            raise ValueError("gap ต้องเป็น unit ที่ไม่ ok")
        if not (self.error_code or self.reason):
            raise ValueError("gap ต้องมี error_code หรือ reason")
        return self


class JobError(BaseModel):
    """error ระดับไฟล์ (ทั้งไฟล์ล้ม) — code มาจาก ``app.models.ErrorCode``"""

    model_config = _examples({"code": "encrypted", "message": "ไฟล์ PDF ถูกเข้ารหัส เปิดไม่ได้"})

    code: DocumentErrorCode
    message: str


EX_JOB_DONE_WITH_GAPS = {
    "job_id": "job_01JD7X2A9Q", "file_id": "file_01JD7X2A9R", "box_id": "example-finance",
    "filename": "รายงานประจำปี_ตัวอย่าง.pdf", "status": "done_with_gaps",
    "progress": {"pages_total": 40, "pages_done": 40}, "document_status": "done_with_gaps",
    "units": {"ok": 39, "needs_ocr": 0, "failed": 1, "rejected": 0},
    "gaps": [{"unit_index": 11, "label": "หน้า 12", "status": "failed", "error_code": None,
              "reason": "OCR แล้วไม่ผ่านด่านตรวจ: latin_in_thai — พบจุดพัง 4 จุด (ละตินกลางคำไทย: 4)"}],
    "error": None, "created_at": _T0, "updated_at": _T1, "trace_id": _TRACE,
}
EX_JOB_QUEUED = {
    "job_id": "job_01JD8Z5V7K", "file_id": "file_01JD8Z5V7M", "box_id": "example-finance",
    "filename": "งบการเงิน_ตัวอย่าง.pdf", "status": "queued", "progress": {"pages_total": None, "pages_done": 0},
    "document_status": None, "units": {"ok": 0, "needs_ocr": 0, "failed": 0, "rejected": 0}, "gaps": [],
    "error": None, "created_at": _T0, "updated_at": _T0, "trace_id": _TRACE,
}
EX_JOB_FAILED = {
    "job_id": "job_01JD9A1B2C", "file_id": "file_01JD9A1B2D", "box_id": "example-finance",
    "filename": "เอกสารเข้ารหัส_ตัวอย่าง.pdf", "status": "failed", "progress": {"pages_total": None, "pages_done": 0},
    "document_status": "failed", "units": {"ok": 0, "needs_ocr": 0, "failed": 0, "rejected": 0}, "gaps": [],
    "error": {"code": "encrypted", "message": "ไฟล์ PDF ถูกเข้ารหัส เปิดไม่ได้"},
    "created_at": _T0, "updated_at": _T0, "trace_id": _TRACE,
}


class JobResponse(BaseModel):
    """สถานะงาน ingest — เห็นได้เฉพาะผู้ส่งหรือ owner/admin ของกล่อง (ไม่ใช่ -> 404)"""

    model_config = _examples(EX_JOB_DONE_WITH_GAPS, EX_JOB_QUEUED, EX_JOB_FAILED)

    job_id: str
    file_id: str
    box_id: str
    filename: str
    status: JobStatus
    progress: Progress
    document_status: Optional[DocumentStatus] = Field(default=None, description="null จนกว่าจะ extract เสร็จ")
    units: UnitCounts
    gaps: list[Gap] = Field(default_factory=list)
    error: Optional[JobError] = None
    created_at: datetime
    updated_at: datetime
    trace_id: str


# ---------------------------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------------------------

EX_SOURCES = {
    "sources": [
        {"box_id": "example-finance", "name": "กล่องตัวอย่าง ฝ่ายการเงิน", "visibility": "private", "role": "admin",
         "file_count": 12, "updated_at": _T1},
        {"box_id": "example-hr", "name": "กล่องตัวอย่าง ฝ่ายบุคคล", "visibility": "public", "role": "viewer",
         "file_count": 3, "updated_at": None},
    ],
    "trace_id": _TRACE,
}
EX_SOURCES_EMPTY = {"sources": [], "trace_id": _TRACE}


class SourceBox(BaseModel):
    model_config = _examples(EX_SOURCES["sources"][0])

    box_id: str
    name: str = Field(description="ชื่อกล่อง (กล่องที่ไม่ได้ตั้งชื่อ = box_id)")
    visibility: Visibility
    role: BoxRole
    file_count: int = Field(ge=0, description="จำนวนไฟล์ที่ผู้เรียกเห็น (กรองตาม ACL แล้ว)")
    updated_at: Optional[datetime] = Field(default=None, description="เวลาที่ไฟล์ล่าสุดเปลี่ยน (null = ยังไม่มีไฟล์)")


class SourcesResponse(BaseModel):
    """กล่องที่ผู้เรียกเป็นสมาชิกเท่านั้น (เรียงตาม box_id)"""

    model_config = _examples(EX_SOURCES, EX_SOURCES_EMPTY)

    sources: list[SourceBox]
    trace_id: str


EX_FILES = {
    "files": [
        {"file_id": "file_01JD7X2A9R", "filename": "รายงานประจำปี_ตัวอย่าง.pdf", "detected_type": "pdf",
         "status": "done_with_gaps", "units_total": 40, "gaps": 1, "ingested_at": _T1},
        {"file_id": "file_01JD8Z5V7M", "filename": "งบการเงิน_ตัวอย่าง.pdf", "detected_type": None,
         "status": "queued", "units_total": 0, "gaps": 0, "ingested_at": None},
    ],
    "next_cursor": "eyJhZnRlciI6ImZpbGVfMDFKRDhaNVY3TSJ9",
    "trace_id": _TRACE,
}
EX_FILES_LAST_PAGE = {"files": [], "next_cursor": None, "trace_id": _TRACE}


class FileItem(BaseModel):
    model_config = _examples(EX_FILES["files"][0])

    file_id: str
    filename: str
    detected_type: Optional[DetectedType] = Field(default=None, description="null = ยังไม่ตรวจชนิด (งานยังไม่เริ่ม)")
    status: JobStatus
    units_total: int = Field(ge=0)
    gaps: int = Field(ge=0, description="จำนวน unit ที่ไม่ ok")
    ingested_at: Optional[datetime] = Field(default=None, description="null = ยัง ingest ไม่เสร็จ")


class FilesResponse(BaseModel):
    """ไฟล์ในกล่อง กรองตาม ACL (ไฟล์ที่ไม่มีสิทธิ์ไม่ถูกนับและไม่ปรากฏ) · ``next_cursor`` null = หน้าสุดท้าย"""

    model_config = _examples(EX_FILES, EX_FILES_LAST_PAGE)

    files: list[FileItem]
    next_cursor: Optional[str] = Field(default=None, description="cursor ทึบ (opaque) ส่งกลับมาใน ?cursor=")
    trace_id: str


# ---------------------------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------------------------

class RetrieveStatus(str, Enum):
    """**enum ที่ขยายได้** — จะเพิ่มค่าใหม่ (เช่น ``source_suggested`` สำหรับ auto routing) โดยไม่นับเป็น breaking change
    Gateway ต้องรับค่าที่ไม่รู้จักได้ และปฏิบัติเหมือน ``no_relevant_context``"""

    OK = "ok"
    NO_RELEVANT_CONTEXT = "no_relevant_context"
    SOURCE_REQUIRED = "source_required"


EX_RETRIEVE_REQUEST = {"query": "บริษัทตัวอย่างจ่ายเงินปันผลหุ้นละเท่าไร", "box_id": "example-finance", "top_k": 5}
EX_RETRIEVE_REQUEST_NO_BOX = {"query": "บริษัทตัวอย่างจ่ายเงินปันผลหุ้นละเท่าไร"}


class RetrieveRequest(BaseModel):
    """ตัวตนผู้ใช้มาทาง header (ไม่อยู่ใน body) · ห้ามส่ง box_ids — สมาชิกกล่องหาเองจาก AclStore"""

    model_config = _examples(EX_RETRIEVE_REQUEST, EX_RETRIEVE_REQUEST_NO_BOX)

    query: str = Field(min_length=1, max_length=QUERY_MAX_CHARS,
                       description=f"1–{QUERY_MAX_CHARS} อักขระ (code point) และต้องมีอักขระที่ไม่ใช่ช่องว่าง")
    box_id: Optional[str] = Field(default=None, pattern=BOX_ID_PATTERN,
                                  description="ไม่ส่ง = status source_required (ยังไม่มี auto routing)")
    top_k: int = Field(default=TOP_K_DEFAULT, ge=1, le=TOP_K_MAX)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query ว่าง (มีแต่ช่องว่าง)")
        return v


class Citation(BaseModel):
    """ตำแหน่งที่อ้างอิงได้ — ฟิลด์ตำแหน่งที่ต้องมีขึ้นกับ unit_type (page: page · slide: slide ·
    section: heading_path · sheet_region: sheet + cell_range) · char_start/char_end = ช่วงของ ``text`` ใน Unit.text เดิม"""

    model_config = _examples({"unit_type": "page", "page": 46, "label": "หน้า 3", "char_start": 1204, "char_end": 1303}, {"unit_type": "sheet_region", "sheet": "ชีตตัวอย่าง", "cell_range": "A1:F40", "label": "ชีตตัวอย่าง A1:F40", "char_start": 0, "char_end": 512})

    unit_type: UnitType
    page: Optional[int] = Field(default=None, ge=1, description="หน้าจริงในไฟล์ เริ่มที่ 1")
    slide: Optional[int] = Field(default=None, ge=1)
    sheet: Optional[str] = None
    cell_range: Optional[str] = Field(default=None, description="เช่น 'A1:F40'")
    heading_path: Optional[list[str]] = None
    label: str = Field(description="สำหรับคนอ่าน เช่น 'หน้า 47'")
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    @model_validator(mode="after")
    def _position_matches_type(self) -> "Citation":
        need = {
            UnitType.PAGE: ("page",),
            UnitType.SLIDE: ("slide",),
            UnitType.SECTION: ("heading_path",),
            UnitType.SHEET_REGION: ("sheet", "cell_range"),
        }[self.unit_type]
        missing = [k for k in need if getattr(self, k) is None]
        if missing:
            raise ValueError(f"citation ของ {self.unit_type.value} ต้องมี {missing}")
        if self.char_end < self.char_start:
            raise ValueError("char_end < char_start")
        return self


class Extraction(BaseModel):
    """ที่มาของข้อความ — method ตาม ``app.models.ExtractionMethod`` (office_xml/converted = DOCX/PPTX/XLSX และไฟล์เก่าที่แปลง)
    flags = ``app.models.QualityFlag`` ของ unit (เช่น ocr_incomplete)"""

    model_config = _examples({"method": "ocr", "flags": ["ocr_incomplete"]})

    method: ExtractionMethod
    flags: list[QualityFlag] = Field(default_factory=list)


EX_RESULT_1 = {
    "rank": 1, "chunk_id": "file_01JD7X2A9R:u0045:c002", "score": 0.82,
    "text": "(ข้อความตัวอย่าง) บริษัท ตัวอย่าง จำกัด จ่ายเงินปันผลหุ้นละ 1.00 บาท สำหรับผลการดำเนินงานปีตัวอย่าง",
    "file_id": "file_01JD7X2A9R", "filename": "รายงานประจำปี_ตัวอย่าง.pdf",
    "citation": {"unit_type": "page", "page": 3, "slide": None, "sheet": None, "cell_range": None,
                 "heading_path": None, "label": "หน้า 3", "char_start": 1204, "char_end": 1303},
    "extraction": {"method": "ocr", "flags": []},
}
EX_RESULT_2 = {
    "rank": 2, "chunk_id": "file_01JD9C3D4E:u0007:c000", "score": 0.61,
    "text": "(ข้อความตัวอย่าง) นโยบายการจ่ายเงินปันผลของบริษัท ตัวอย่าง จำกัด",
    "file_id": "file_01JD9C3D4E", "filename": "นโยบาย_ตัวอย่าง.docx",
    "citation": {"unit_type": "section", "page": None, "slide": None, "sheet": None, "cell_range": None,
                 "heading_path": ["หมวดตัวอย่าง 1", "หัวข้อตัวอย่าง 1.1"], "label": "หมวดตัวอย่าง 1 › หัวข้อตัวอย่าง 1.1",
                 "char_start": 0, "char_end": 64},
    "extraction": {"method": "office_xml", "flags": []},
}
EX_RESULT_LOW = {**EX_RESULT_2, "rank": 1, "score": 0.18}
EX_RETRIEVE_OK = {"status": "ok", "box_id": "example-finance", "visibility": "private", "top_score": 0.82,
                  "results": [EX_RESULT_1, EX_RESULT_2], "trace_id": _TRACE}
EX_RETRIEVE_NO_RELEVANT = {"status": "no_relevant_context", "box_id": "example-finance", "visibility": "private",
                           "top_score": 0.18, "results": [EX_RESULT_LOW], "trace_id": _TRACE}
EX_RETRIEVE_SOURCE_REQUIRED = {"status": "source_required", "box_id": None, "visibility": None, "top_score": None,
                               "results": [], "trace_id": _TRACE}
# กล่องที่เป็นสมาชิกแต่ไม่มีผล กับกล่องที่ไม่รู้จัก/ไม่เป็นสมาชิก: ต่างกันแค่ trace_id (กฎเหล็กข้อ 9)
EX_RETRIEVE_EMPTY_MEMBER = {"status": "no_relevant_context", "box_id": "example-hr", "visibility": None,
                            "top_score": None, "results": [], "trace_id": _TRACE}
EX_RETRIEVE_NOT_MEMBER = {**EX_RETRIEVE_EMPTY_MEMBER, "trace_id": _TRACE_2}


class RetrieveResult(BaseModel):
    model_config = _examples(EX_RESULT_1)

    rank: int = Field(ge=1)
    chunk_id: str
    score: float
    text: str
    file_id: str
    filename: str
    citation: Citation
    extraction: Extraction


class RetrieveResponse(BaseModel):
    """ผลค้น (ไม่ generate) — กติกา (validator):
    - ``source_required``: ไม่ได้ส่ง box_id -> box_id null, results ว่าง
    - ``ok``: มีผล และ top_score >= RETRIEVE_MIN_SCORE
    - ``no_relevant_context``: top_score ต่ำกว่าเกณฑ์ (ยังคืน results) **หรือ** ไม่มีผลเลย
    - ไม่มีผล -> visibility และ top_score เป็น null เสมอ: กล่องที่ไม่รู้จัก / ไม่ได้เป็นสมาชิก จึงตอบเหมือน
      กล่องที่เป็นสมาชิกแต่ไม่มีผลทุกฟิลด์ยกเว้น trace_id
    - visibility มาจาก Box ใน AclStore เท่านั้น
    """

    model_config = _examples(EX_RETRIEVE_OK, EX_RETRIEVE_NO_RELEVANT, EX_RETRIEVE_SOURCE_REQUIRED,
                             EX_RETRIEVE_EMPTY_MEMBER)

    status: RetrieveStatus
    box_id: Optional[str]
    visibility: Optional[Visibility]
    top_score: Optional[float]
    results: list[RetrieveResult]
    trace_id: str

    @model_validator(mode="after")
    def _consistent(self) -> "RetrieveResponse":
        if not self.results and (self.visibility is not None or self.top_score is not None):
            raise ValueError("ไม่มีผล -> visibility และ top_score ต้องเป็น null (ห้ามเผยว่ากล่องมีอยู่)")
        if self.results:
            if self.visibility is None or self.top_score is None:
                raise ValueError("มีผล -> ต้องมี visibility และ top_score")
            if self.top_score != max(r.score for r in self.results):
                raise ValueError("top_score ต้องเท่ากับคะแนนสูงสุดของ results")
            if [r.rank for r in self.results] != list(range(1, len(self.results) + 1)):
                raise ValueError("rank ต้องเรียง 1..n")
        if self.status == RetrieveStatus.SOURCE_REQUIRED and (self.box_id is not None or self.results):
            raise ValueError("source_required -> box_id null และ results ว่าง")
        if self.status == RetrieveStatus.OK and not self.results:
            raise ValueError("status ok ต้องมีผล (ไม่มีผล = no_relevant_context)")
        return self


# ---------------------------------------------------------------------------------------------
# admin: boxes + members
# ---------------------------------------------------------------------------------------------

EX_CREATE_BOX = {"box_id": "example-finance", "name": "กล่องตัวอย่าง ฝ่ายการเงิน", "visibility": "private",
                 "departments": ["finance", "audit"], "min_security_level": 2}
EX_BOX_CREATED = {**EX_CREATE_BOX, "owner": "u1024", "trace_id": _TRACE}


class CreateBoxRequest(BaseModel):
    """สร้างกล่อง — ผู้เรียกต้องอยู่ใน SYSTEM_ADMIN_USERS และกลายเป็น owner คนแรก
    visibility ตั้งได้ครั้งเดียวตอนสร้าง (ไม่มี endpoint แก้ — กฎเหล็กข้อ 9) · box_id ซ้ำ -> 422 invalid_request
    ``departments``: ไม่ส่ง/null = ไม่จำกัดแผนก · [] = ไม่มีแผนกไหนเห็น"""

    model_config = _examples(EX_CREATE_BOX)

    box_id: str = Field(pattern=BOX_ID_PATTERN)
    name: str = Field(min_length=1, max_length=200)
    visibility: Visibility
    departments: Optional[list[str]] = None
    min_security_level: int = Field(default=0, ge=0)


class BoxResponse(BaseModel):
    model_config = _examples(EX_BOX_CREATED)

    box_id: str
    name: str
    visibility: Visibility
    departments: Optional[list[str]]
    min_security_level: int
    owner: str = Field(description="user_id ของ owner คนแรก (ผู้สร้าง)")
    trace_id: str


EX_PUT_MEMBER = {"user_id": "u2048", "role": "viewer"}
EX_MEMBER = {"box_id": "example-finance", "user_id": "u2048", "role": "viewer", "trace_id": _TRACE}
EX_MEMBER_REMOVED = {"box_id": "example-finance", "user_id": "u2048", "removed": True, "trace_id": _TRACE}


class PutMemberRequest(BaseModel):
    """เพิ่มสมาชิก หรือเปลี่ยน role — ลด role ของ owner คนสุดท้าย -> 422 invalid_request
    ให้/ถอด role owner ได้เฉพาะ owner (admin จัดการได้แค่ admin/viewer)"""

    model_config = _examples(EX_PUT_MEMBER)

    user_id: str = Field(min_length=1, max_length=128)
    role: BoxRole


class MemberResponse(BaseModel):
    model_config = _examples(EX_MEMBER)

    box_id: str
    user_id: str
    role: BoxRole
    trace_id: str


class MemberRemovedResponse(BaseModel):
    """ถอดสมาชิกแล้ว (มีผลทันทีกับการค้น: ด่าน 2 ถามสมาชิกภาพสด) — ไม่ได้เป็นสมาชิกอยู่ -> 404 · owner คนสุดท้าย -> 422"""

    model_config = _examples(EX_MEMBER_REMOVED)

    box_id: str
    user_id: str
    removed: Literal[True] = True
    trace_id: str


# ---------------------------------------------------------------------------------------------
# ตัวอย่างทั้งหมด: ชื่อไฟล์ (<endpoint>_<case>) -> (model, payload)
# ---------------------------------------------------------------------------------------------

EXAMPLES: dict[str, tuple[type[BaseModel], dict[str, Any]]] = {
    "health_live_ok": (LiveResponse, EX_LIVE),
    "health_ready_ready": (ReadyResponse, EX_READY),
    "health_ready_ready_cpu": (ReadyResponse, EX_READY_CPU),
    "health_ready_degraded": (ReadyResponse, EX_READY_DEGRADED),
    "ingest_file_acl": (FileAclInput, EX_FILE_ACL),
    "ingest_queued": (IngestResponse, EX_INGEST_QUEUED),
    "ingest_duplicate": (IngestResponse, EX_INGEST_DUPLICATE),
    "ingest_not_found": (ErrorResponse, EX_ERR_NOT_FOUND),
    "ingest_acl_widening": (ErrorResponse, EX_ERR_ACL_WIDENING),
    "ingest_too_large": (ErrorResponse, EX_ERR_TOO_LARGE),
    "ingest_unsupported_type": (ErrorResponse, EX_ERR_UNSUPPORTED),
    "jobs_done_with_gaps": (JobResponse, EX_JOB_DONE_WITH_GAPS),
    "jobs_queued": (JobResponse, EX_JOB_QUEUED),
    "jobs_failed": (JobResponse, EX_JOB_FAILED),
    "jobs_not_found": (ErrorResponse, EX_ERR_NOT_FOUND),
    "sources_list": (SourcesResponse, EX_SOURCES),
    "sources_list_empty": (SourcesResponse, EX_SOURCES_EMPTY),
    "sources_files": (FilesResponse, EX_FILES),
    "sources_files_last_page": (FilesResponse, EX_FILES_LAST_PAGE),
    "sources_files_not_found": (ErrorResponse, EX_ERR_NOT_FOUND),
    "retrieve_request": (RetrieveRequest, EX_RETRIEVE_REQUEST),
    "retrieve_request_no_box": (RetrieveRequest, EX_RETRIEVE_REQUEST_NO_BOX),
    "retrieve_ok": (RetrieveResponse, EX_RETRIEVE_OK),
    "retrieve_no_relevant_context": (RetrieveResponse, EX_RETRIEVE_NO_RELEVANT),
    "retrieve_source_required": (RetrieveResponse, EX_RETRIEVE_SOURCE_REQUIRED),
    "retrieve_member_empty_box": (RetrieveResponse, EX_RETRIEVE_EMPTY_MEMBER),
    "retrieve_not_member_box": (RetrieveResponse, EX_RETRIEVE_NOT_MEMBER),
    "boxes_create_request": (CreateBoxRequest, EX_CREATE_BOX),
    "boxes_create_created": (BoxResponse, EX_BOX_CREATED),
    "boxes_create_not_found": (ErrorResponse, EX_ERR_NOT_FOUND),
    "members_put_request": (PutMemberRequest, EX_PUT_MEMBER),
    "members_put_ok": (MemberResponse, EX_MEMBER),
    "members_put_not_found": (ErrorResponse, EX_ERR_NOT_FOUND),
    "members_delete_ok": (MemberRemovedResponse, EX_MEMBER_REMOVED),
    "error_unauthorized": (ErrorResponse, EX_ERR_UNAUTHORIZED),
    "error_invalid_request": (ErrorResponse, EX_ERR_INVALID),
    "error_unavailable": (ErrorResponse, EX_ERR_UNAVAILABLE),
    "error_internal": (ErrorResponse, EX_ERR_INTERNAL),
    "error_not_implemented": (ErrorResponse, EX_ERR_NOT_IMPLEMENTED),
}
