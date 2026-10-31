"""PDF extractor: หนึ่งหน้า = หนึ่ง Unit (type=page, method=text_layer)

- ข้อความดึงด้วย ``get_text("dict")``; text block แต่ละก้อนเป็น ``Block`` ประเภท paragraph
  (อนุมานหัวข้อทีหลัง) ลำดับ block ตามที่ PyMuPDF ให้มา
- ข้อความทุกส่วนผ่าน ``normalize_with_report()`` ทีละ block ก่อนใส่ Unit.text / Block.text
  ผลนับของ normalizer รวมราย unit: ``QualityReport.normalization_changes`` = ผลรวม fixes,
  ``Unit.signals`` ใช้ key แบบแบน ``norm_fix:<rule>`` / ``norm_sig:<signal>`` (ใส่เฉพาะค่าที่ไม่ใช่ 0)
- หน้าที่พังได้ status failed พร้อม reason หน้าอื่นทำต่อ
- ไม่ตัดสิน needs_ocr ที่นี่ (ทำใน app/extraction/quality.py ที่ pipeline เรียกหลังได้ทุก unit); quality.score เป็น None
"""

import hashlib
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pymupdf

from app.extraction.errors import ExtractionError
from app.models import (
    Block,
    BlockType,
    ErrorCode,
    ExtractionMethod,
    QualityFlag,
    QualityReport,
    Unit,
    UnitStatus,
    UnitType,
)
from app.thai.normalizer import NormalizeResult, normalize_with_report

_TEXT_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES  # ไม่ดึงไบต์ภาพมาเก็บ
_BOLD_SPAN_FLAG = 16


@dataclass
class PdfResult:
    units: list[Unit] = field(default_factory=list)
    has_outline: bool = False
    producer: Optional[str] = None
    page_count: int = 0


def _is_thai(ch: str) -> bool:
    return "฀" <= ch <= "๿"


def _is_junk(ch: str) -> bool:
    """อักขระที่ไม่ใช่ตัวอักษรและไม่ใช่ช่องว่าง (Cc, Cf, U+FFFD) — ข้ามตอนตัดสินที่รอยต่อเท่านั้น"""
    return not ch.isspace() and (unicodedata.category(ch) in ("Cc", "Cf") or ch == "\ufffd")


def _edge_char(text: str, last: bool) -> Optional[str]:
    """อักขระท้าย (last=True) หรือต้น ของ text โดยข้าม junk; ช่องว่างนับเป็นอักขระปกติ"""
    for ch in reversed(text) if last else text:
        if not _is_junk(ch):
            return ch
    return None


def join_lines(lines: list[str]) -> str:
    """ต่อบรรทัดใน block: มีช่องว่างอยู่แล้ว -> ต่อตรง, ไทย+ไทย -> ต่อตรง, นอกนั้นแทรกช่องว่าง 1 ช่อง

    ตอนตัดสินจะข้าม junk (Cc/Cf/U+FFFD) ตรงรอยต่อก่อน แต่ junk ยังอยู่ในข้อความ
    (การทำความสะอาดเป็นหน้าที่ของ normalizer ที่เดียว)
    """
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    out = lines[0]
    for nxt in lines[1:]:
        prev_last, next_first = _edge_char(out, last=True), _edge_char(nxt, last=False)
        if prev_last is None or next_first is None:
            out += nxt
        elif prev_last.isspace() or next_first.isspace():
            out += nxt
        elif _is_thai(prev_last) and _is_thai(next_first):
            out += nxt
        else:
            out += " " + nxt
    return out


def _span_bold(span: dict) -> bool:
    return bool(span.get("flags", 0) & _BOLD_SPAN_FLAG) or "bold" in span.get("font", "").lower()


@dataclass
class RawBlock:
    """text block ก่อน normalize (ดึงจาก PyMuPDF dict; ใช้ร่วมกับสคริปต์วัดผล)"""

    text: str
    bbox: tuple[float, float, float, float]
    font_size: float
    bold: bool


def _raw_block(raw: dict) -> Optional[RawBlock]:
    lines: list[str] = []
    spans_all: list[dict] = []
    for line in raw.get("lines", []):
        spans = line.get("spans", [])
        spans_all.extend(spans)
        lines.append("".join(s.get("text", "") for s in spans))
    text = join_lines(lines)
    if not text.strip() or not spans_all:
        return None
    main = max(spans_all, key=lambda s: len(s.get("text", "")))
    return RawBlock(
        text=text,
        bbox=tuple(raw["bbox"]),
        font_size=round(main.get("size", 0.0), 2),
        bold=_span_bold(main),
    )


def raw_blocks(page: "pymupdf.Page") -> list[RawBlock]:
    """text block ของหน้า (ก่อน normalize) ตามลำดับที่ PyMuPDF ให้มา — จุดดึงข้อความจุดเดียว"""
    raw = page.get_text("dict", flags=_TEXT_FLAGS)
    blocks = (_raw_block(rb) for rb in raw.get("blocks", []) if rb.get("type") == 0)
    return [b for b in blocks if b]


def _norm_signals(total_fixes: dict[str, int], total_signals: dict[str, int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, n in total_fixes.items():
        if n:
            out[f"norm_fix:{name}"] = float(n)
    for name, n in total_signals.items():
        if n:
            out[f"norm_sig:{name}"] = float(n)
    return out


def _image_coverage(page: "pymupdf.Page") -> float:
    area = abs(page.rect.width * page.rect.height)
    if area <= 0:
        return 0.0
    largest = 0.0
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & page.rect
        if not r.is_empty:
            largest = max(largest, r.width * r.height)
    return min(largest / area, 1.0)


def _unit_from_blocks(index: int, page_no: int, label: Optional[str], blocks: list[Block],
                      image_coverage: float, elapsed_ms: float,
                      norm_fixes: dict[str, int], norm_signals: dict[str, int]) -> Unit:
    text = "\n".join(b.text for b in blocks)
    position: dict = {"page": page_no}
    if label:
        position["page_label"] = label
    return Unit(
        index=index,
        type=UnitType.PAGE,
        position=position,
        label=f"หน้า {page_no}",
        status=UnitStatus.OK,
        method=ExtractionMethod.TEXT_LAYER,
        blocks=blocks,
        text=text,
        text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        quality=QualityReport(
            score=None,
            flags=[] if text.strip() else [QualityFlag.EMPTY],
            normalization_changes=sum(norm_fixes.values()),
        ),
        elapsed_ms=elapsed_ms,
        signals={
            "char_count": float(len(text.strip())),
            "block_count": float(len(blocks)),
            "image_coverage": image_coverage,
            **_norm_signals(norm_fixes, norm_signals),
        },
    )


def _extract_page(page: "pymupdf.Page", index: int) -> Unit:
    t0 = time.perf_counter()
    blocks: list[Block] = []
    fixes: dict[str, int] = {}
    signals: dict[str, int] = {}
    for rb in raw_blocks(page):
        res: NormalizeResult = normalize_with_report(rb.text)
        for k, v in res.fixes.items():
            fixes[k] = fixes.get(k, 0) + v
        for k, v in res.signals.items():
            signals[k] = signals.get(k, 0) + v
        text = res.text.strip()
        if not text:  # เช่น block ที่มีแต่ #MARKER: — นับผล normalize ไว้แล้ว แต่ไม่สร้าง block
            continue
        blocks.append(
            Block(
                type=BlockType.PARAGRAPH,
                text=text,
                order=len(blocks),
                bbox=rb.bbox,
                font_size=rb.font_size,
                bold=rb.bold,
            )
        )
    return _unit_from_blocks(
        index,
        page.number + 1,
        page.get_label() or None,
        blocks,
        _image_coverage(page),
        (time.perf_counter() - t0) * 1000,
        fixes,
        signals,
    )


def _failed_unit(index: int, page_no: int, error: Exception, elapsed_ms: float) -> Unit:
    return Unit(
        index=index,
        type=UnitType.PAGE,
        position={"page": page_no},
        label=f"หน้า {page_no}",
        status=UnitStatus.FAILED,
        reason=f"อ่านหน้า {page_no} ไม่สำเร็จ: {type(error).__name__}: {error}",
        method=ExtractionMethod.TEXT_LAYER,
        text="",
        text_hash=hashlib.sha256(b"").hexdigest(),
        quality=QualityReport(),
        elapsed_ms=elapsed_ms,
        signals={"char_count": 0.0, "block_count": 0.0, "image_coverage": 0.0},
    )


def extract_pdf(path: str | Path, pages: Optional[tuple[int, int]] = None) -> PdfResult:
    """อ่าน PDF ทีละหน้า. ``pages`` = (เริ่ม, จบ) นับจาก 1 รวมปลายทั้งสองข้าง (None = ทั้งไฟล์)

    ``Unit.index`` นับต่อเนื่องจาก 0 ในชุด unit ที่ได้ ส่วน ``position.page`` เป็นหน้าจริงในไฟล์
    """
    try:
        doc = pymupdf.open(str(path))
    except Exception as e:
        raise ExtractionError(ErrorCode.CORRUPT, f"เปิด PDF ไม่ได้: {e}") from e
    with doc:
        if doc.needs_pass:
            raise ExtractionError(ErrorCode.ENCRYPTED, "PDF ตั้งรหัสผ่านสำหรับเปิดอ่าน")
        if doc.page_count == 0:
            raise ExtractionError(ErrorCode.CORRUPT, "PDF ไม่มีหน้า")

        start, end = pages if pages else (1, doc.page_count)
        start, end = max(start, 1), min(end, doc.page_count)
        result = PdfResult(
            has_outline=bool(doc.get_toc(simple=True)),
            producer=(doc.metadata or {}).get("producer") or None,
            page_count=doc.page_count,
        )
        for page_no in range(start, end + 1):
            index = len(result.units)
            t0 = time.perf_counter()
            try:
                unit = _extract_page(doc.load_page(page_no - 1), index)
            except Exception as e:
                unit = _failed_unit(index, page_no, e, (time.perf_counter() - t0) * 1000)
            result.units.append(unit)
        return result
