"""Contract กลางของ ingestion (output ที่ extractor ทุกตัวต้องคืนค่า)

extractor ต่อไฟล์ -> ``Document`` ที่ประกอบด้วย ``Unit`` หลายตัว แต่ละ ``Unit`` มี ``Block`` ข้างใน
chunking / citation / eval อ่านต่อจาก contract นี้

- Unit ต่อประเภท: PDF=page, PPTX=slide, DOCX=section, XLSX=sheet_region
- ทุก Unit มี ``position`` (เครื่องอ่าน) และ ``label`` (คนอ่าน เช่น "หน้า 47")
- สถานะราย unit: unit ที่พังไม่ทำให้ทั้งไฟล์พัง ``Document.status`` คำนวณจากสถานะของ unit
  (ทุก unit ok -> done, บางส่วนไม่ ok -> done_with_gaps, ไม่มี unit ok เลย -> failed)
- นิยาม ``position``:
  - ``page`` = ลำดับหน้าจริงในไฟล์ เริ่มที่ 1 (ไม่ใช่เลขที่พิมพ์บนหน้า)
    ``page_label`` (ไม่บังคับ, str) = เลขหน้าที่พิมพ์บนหน้า เช่น "iv" หรือ "45"
  - ``heading_path`` = หัวข้อจากบนลงล่าง **รวมหัวข้อของ section นั้นเองเป็นตัวสุดท้าย**
    ``paragraphs`` = [ย่อหน้าเริ่ม, ย่อหน้าจบ]
- ``text_hash`` = SHA-256 (hex) ของ ``text`` หลัง normalize เข้ารหัส UTF-8
- Unit ที่ status = ok แต่ text ว่าง (หลัง strip) ต้องมี flag ``empty`` ใน quality
- ``Document.error`` ถ้าไม่ใช่ None ``status`` จะเป็น failed เสมอ
- ``Document`` ไม่มีฟิลด์สิทธิ์ (box / security level / แผนก / ACL) โดยตั้งใจ
  จะแยกเป็น model ของตัวเองในเฟส ACL และเชื่อมกันด้วย ``file_id``

ตัวอย่าง JSON ของ unit ทั้ง 4 ประเภทอยู่ที่ ``tests/fixtures/units/`` (ใช้เป็น test ด้วย)
แก้ contract นี้ต้องอัปเดต fixtures และเอกสารที่เกี่ยวข้องใน commit เดียวกัน (CLAUDE.md ข้อ 8)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field, model_validator


class UnitType(str, Enum):
    PAGE = "page"
    SLIDE = "slide"
    SECTION = "section"
    SHEET_REGION = "sheet_region"


class UnitStatus(str, Enum):
    OK = "ok"
    NEEDS_OCR = "needs_ocr"
    FAILED = "failed"
    REJECTED = "rejected"


class ExtractionMethod(str, Enum):
    TEXT_LAYER = "text_layer"
    OCR = "ocr"
    OFFICE_XML = "office_xml"
    CONVERTED = "converted"


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"
    CAPTION = "caption"


class QualityFlag(str, Enum):
    LATIN_IN_THAI = "latin_in_thai"
    CONTROL_CHARS = "control_chars"
    REPLACEMENT_CHAR = "replacement_char"
    INVALID_MARK_SEQUENCE = "invalid_mark_sequence"
    LOW_DICTIONARY_RATIO = "low_dictionary_ratio"
    SCAN_LIKE = "scan_like"
    EMPTY = "empty"


class DetectedType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    DOC = "doc"
    PPT = "ppt"
    XLS = "xls"


class ErrorCode(str, Enum):
    ENCRYPTED = "encrypted"
    CORRUPT = "corrupt"
    UNSUPPORTED_TYPE = "unsupported_type"
    TOO_LARGE = "too_large"
    CONVERSION_TIMEOUT = "conversion_timeout"
    INTERNAL = "internal"


# ลองใหม่ได้เฉพาะ error ชั่วคราว (retriable คำนวณจาก code ไม่ให้กรอกเอง)
RETRIABLE_CODES = frozenset({ErrorCode.CONVERSION_TIMEOUT, ErrorCode.INTERNAL})


class DocumentError(BaseModel):
    code: ErrorCode
    message: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def retriable(self) -> bool:
        return self.code in RETRIABLE_CODES


class DocumentStatus(str, Enum):
    DONE = "done"
    DONE_WITH_GAPS = "done_with_gaps"
    FAILED = "failed"


# key ที่ position ต้องมีของแต่ละ UnitType
POSITION_REQUIRED_KEYS: dict[UnitType, frozenset[str]] = {
    UnitType.PAGE: frozenset({"page"}),
    UnitType.SLIDE: frozenset({"slide"}),
    UnitType.SECTION: frozenset({"heading_path", "paragraphs"}),
    UnitType.SHEET_REGION: frozenset({"sheet", "range"}),
}


class MergedCell(BaseModel):
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)


class Table(BaseModel):
    rows: list[list[str]]
    header_rows: int = Field(default=0, ge=0)
    merged_cells: list[MergedCell] = Field(default_factory=list)


class Block(BaseModel):
    type: BlockType
    text: str  # หลัง normalize
    order: int = Field(ge=0)  # ลำดับใน unit
    level: Optional[int] = Field(default=None, ge=1)  # ระดับหัวข้อ ถ้าเป็น heading
    bbox: Optional[tuple[float, float, float, float]] = None  # x0, y0, x1, y1 (ถ้ามี)
    font_size: Optional[float] = None  # ไว้อนุมานหัวข้อทีหลัง
    bold: Optional[bool] = None
    table: Optional[Table] = None  # เฉพาะ type=table


class QualityReport(BaseModel):
    score: float = Field(ge=0, le=1)
    flags: list[QualityFlag] = Field(default_factory=list)
    dictionary_ratio: Optional[float] = Field(default=None, ge=0, le=1)
    normalization_changes: int = Field(default=0, ge=0)


class Unit(BaseModel):
    index: int = Field(ge=0)  # ลำดับใน document เริ่มที่ 0
    type: UnitType
    position: dict[str, Any]  # เครื่องอ่าน ดู POSITION_REQUIRED_KEYS
    label: str  # คนอ่าน เช่น "หน้า 47"
    status: UnitStatus
    reason: Optional[str] = None  # ต้องมีเมื่อ status != ok
    method: ExtractionMethod
    blocks: list[Block] = Field(default_factory=list)
    text: str  # รวมทั้ง unit หลัง normalize
    text_hash: str
    quality: QualityReport
    elapsed_ms: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_contract(self) -> "Unit":
        missing = POSITION_REQUIRED_KEYS[self.type] - self.position.keys()
        if missing:
            raise ValueError(
                f"position ของ {self.type.value} ขาด key: {sorted(missing)}"
            )
        if self.status != UnitStatus.OK and not self.reason:
            raise ValueError("unit ที่ status != ok ต้องมี reason")
        if (
            self.status == UnitStatus.OK
            and not self.text.strip()
            and QualityFlag.EMPTY not in self.quality.flags
        ):
            raise ValueError("unit ที่ ok แต่ text ว่าง ต้องมี flag empty")
        if self.type == UnitType.PAGE:
            page = self.position["page"]
            if isinstance(page, bool) or not isinstance(page, int) or page < 1:
                raise ValueError("position.page ต้องเป็นจำนวนเต็มเริ่มที่ 1")
        return self


class Document(BaseModel):
    file_id: str
    content_hash: str  # SHA-256
    filename: str
    detected_type: DetectedType  # จาก magic bytes ไม่ใช่นามสกุล
    size_bytes: int = Field(ge=0)
    converted_from: Optional[str] = None  # ถ้าแปลงผ่าน LibreOffice
    # TODO(T01 คำถามค้างข้อ 1): ยังไม่ตกลงกับบริษัทว่า DOCX/XLSX นับอย่างไร
    #   ค่าชั่วคราว: DOCX = จำนวนหน้าที่ LibreOffice แปลงได้, XLSX = 1 ต่อ sheet_region
    page_equivalent: Optional[int] = Field(default=None, ge=0)
    has_outline: bool = False
    producer: Optional[str] = None
    units: list[Unit] = Field(default_factory=list)
    error: Optional[DocumentError] = None  # ถ้ามี status ต้องเป็น failed
    pipeline_version: str
    normalizer_version: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unit_count(self) -> int:
        return len(self.units)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unit_summary(self) -> dict[str, int]:
        summary = {s.value: 0 for s in UnitStatus}
        for u in self.units:
            summary[u.status.value] += 1
        return summary

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> DocumentStatus:
        if self.error is not None:
            return DocumentStatus.FAILED
        ok = sum(1 for u in self.units if u.status == UnitStatus.OK)
        if ok == 0:
            return DocumentStatus.FAILED
        if ok == len(self.units):
            return DocumentStatus.DONE
        return DocumentStatus.DONE_WITH_GAPS
