"""Pipeline: ไฟล์ -> detect -> extractor -> ``Document``

error ระดับไฟล์ไม่โยนออกนอก ``extract_file`` แต่ใส่ใน ``Document.error`` (status = failed)
error ระดับ unit อยู่ใน status ของ unit เอง (CLAUDE.md ข้อ 3)
"""

import hashlib
import uuid
from pathlib import Path
from typing import Optional

from app.extraction.convert import TARGETS, converted, legacy_page_count
from app.extraction.detect import detect
from app.extraction.errors import ExtractionError
from app.extraction.docx import extract_docx
from app.extraction.pdf import extract_pdf
from app.extraction.pptx import extract_pptx
from app.extraction.quality import apply_gate
from app.extraction.xlsx import extract_xlsx
from app.models import DetectedType, Document, DocumentError, ErrorCode, ExtractionMethod
from app.thai.normalizer import NORMALIZER_VERSION

PIPELINE_VERSION = "t02-0.1"
_CHUNK = 1024 * 1024


def _hash_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _extract(path: Path, detected: DetectedType, pages):
    """คืน (ผลของ extractor, page_equivalent); ``pages`` ใช้กับ PDF เท่านั้น"""
    if detected == DetectedType.PDF:
        result = extract_pdf(path, pages=pages)
        return result, result.page_count
    if detected == DetectedType.DOCX:
        result = extract_docx(path)
        return result, result.page_equivalent
    if detected == DetectedType.PPTX:
        result = extract_pptx(path)
        return result, result.slide_count
    if detected == DetectedType.XLSX:
        result = extract_xlsx(path)
        return result, result.page_equivalent  # ชั่วคราวตาม T01: 1 ต่อ sheet_region
    raise ExtractionError(ErrorCode.UNSUPPORTED_TYPE, "ยังไม่รองรับใน v1")


def extract_file(
    path: str | Path,
    file_id: Optional[str] = None,
    pages: Optional[tuple[int, int]] = None,
) -> Document:
    """``pages`` ใช้กับ PDF เท่านั้น (เริ่ม, จบ) นับจาก 1 ไว้ลองกับไฟล์ใหญ่"""
    path = Path(path)
    content_hash, size = _hash_file(path)
    fields = dict(
        file_id=file_id or uuid.uuid4().hex,
        content_hash=content_hash,
        filename=path.name,
        size_bytes=size,
        pipeline_version=PIPELINE_VERSION,
        normalizer_version=NORMALIZER_VERSION,
    )
    detected: Optional[DetectedType] = None
    converted_from: Optional[str] = None
    try:
        detected = detect(path)
        if detected in TARGETS:  # doc / xls / ppt: แปลงด้วย LibreOffice แล้วใช้ extractor ของชนิดสมัยใหม่
            converted_from = detected.value
            with converted(path, detected) as new_path:
                result, page_equivalent = _extract(new_path, TARGETS[detected][2], pages)
            for u in result.units:
                u.method = ExtractionMethod.CONVERTED
            page_equivalent = legacy_page_count(path, detected) or page_equivalent  # .doc: หน้าที่ Word บันทึกไว้
        else:
            result, page_equivalent = _extract(path, detected, pages)
        if not result.units:
            raise ExtractionError(ErrorCode.EMPTY, "ไฟล์ไม่มีเนื้อหาที่อ่านได้")
    except ExtractionError as e:
        error = DocumentError(code=e.code, message=e.message)
        return Document(**fields, detected_type=detected, converted_from=converted_from, error=error)
    except Exception as e:  # ไม่ให้ bug ใน extractor ทำให้ทั้ง batch ล้ม
        error = DocumentError(code=ErrorCode.INTERNAL, message=f"{type(e).__name__}: {e}")
        return Document(**fields, detected_type=detected, converted_from=converted_from, error=error)

    return Document(
        **fields,
        detected_type=detected,
        converted_from=converted_from,
        units=apply_gate(result.units),
        has_outline=result.has_outline,
        producer=result.producer,
        page_equivalent=page_equivalent,
    )
