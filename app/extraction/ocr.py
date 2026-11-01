"""OCR เป็นขั้นแยกจาก ``extract_file`` (T06a)

``extract_file`` ไม่เรียก OCR ข้างใน: มันทำเครื่องหมาย unit ที่ต้อง OCR (status = needs_ocr) แล้วหยุด
``run_ocr(document, source_path, backend)`` คืน Document ใหม่ที่ OCR เฉพาะ unit ที่ needs_ocr
- unit ที่ไม่ใช่ needs_ocr ไม่ถูกแตะ (คืนวัตถุเดิม)
- unit ที่ OCR แล้ว: method = ocr เสมอ; ผลของโมเดลถูกแปลงเป็น block (``ocr_markdown.py``) แล้วข้อความทุกส่วนผ่าน
  ``normalize_with_report`` (ตัวเดียวของระบบ) แล้วผ่านด่านตรวจ ``evaluate_unit`` ตัวเดิมรอบใหม่
  (ไม่เรียกกฎระดับไฟล์ ``apply_document_rule`` เพราะจะไปแตะ unit อื่น) สถานะตามผลด่านตรวจ — ถ้า OCR ออกมาพัง
  เกณฑ์เดิมก็ยัง needs_ocr; ``image_coverage`` ของ unit เดิมไม่ถูกยกมา (หน้าภาพที่ OCR แล้วได้ตัวอักษรน้อย
  ต้องเป็น low_text ไม่ใช่ scan_like วน)
- ``<figure>`` (คำบรรยายภาพที่โมเดลแต่งเอง) เก็บเป็น block image แต่ไม่รวมใน ``Unit.text``
- หน้ากว้าง (กว้าง/สูง > ``WIDE_PAGE_RATIO`` เช่นหน้าคู่ของรายงานประจำปี) ถูกตัดซ้าย-ขวา OCR ทีละครึ่ง
  แล้วต่อข้อความซ้ายก่อนขวา เพราะโมเดลย่อภาพให้ด้านยาว 1800 px (ตาม model card) หน้ากว้างจึงเหลือตัวหนังสือเล็กเกินอ่าน
  จุดตัด = กลางช่องว่างแนวตั้งที่กว้างที่สุดในช่วง ``GUTTER_WINDOW`` ของความกว้าง (``find_gutter``: โปรไฟล์หมึกของภาพที่
  render แล้ว) เพราะหน้าคู่ไม่สมมาตร (เช่น KBANK มีแถบสารบัญด้านซ้าย) การตัดกลางหน้าจะผ่ากลางตัวหนังสือแล้วโมเดลเดาคำ
  ที่ขอบตัด; ไม่มีช่องว่างกว้างพอ -> ตัดกลางหน้า (``ocr:split_fallback`` = 1)
- OCR หน้าไหนล้มเหลว (โมเดล error, GPU หน่วยความจำไม่พอ) -> unit นั้น failed พร้อมเหตุผล หน้าอื่นทำต่อ

signals ของ unit ที่ OCR: ``ocr:seconds`` (เวลา backend ต่อหน้า รวมทุกครึ่ง), ``ocr:figure_count``,
``ocr:page_number`` (เลขหน้าที่โมเดลอ่านได้จาก <page_number>; หน้าคู่ที่ได้สองเลขใส่เลขที่สองใน ``ocr:page_number_2``),
``ocr:tokens`` (จำนวนโทเคนที่โมเดลสร้าง รวมทุกครึ่ง; ถ้ามี), ``ocr:split_x`` (ตำแหน่งตัดหน้ากว้าง สัดส่วน 0–1 ของความกว้าง) และ
``ocr:split_fallback`` (1 = ไม่เจอร่อง ใช้กลางหน้า, 0 = ตัดที่ร่อง) เฉพาะหน้าที่ถูกตัด,
``ocr:repeated_lines`` (ตัวนับการพิมพ์วน: ดูด้านล่าง),
``ocr:text_layer_recall`` / ``ocr:text_layer_precision`` (ดูด้านล่าง)
ชื่อ backend / รุ่น / revision ของโมเดลเป็นข้อความ ใส่ใน ``reason`` ("OCR: <backend> · <model>@<revision>")

ความสอดคล้องกับชั้นข้อความเดิม (หลัง normalize ทั้งสองฝั่ง ตัดคำ pythainlp newmm ตัดโทเคนที่ไม่มีตัวอักษร/ตัวเลข
นับแบบ multiset ไม่สนลำดับ; ข้อความ OCR ที่ใช้คือ ``Unit.text`` ซึ่งไม่มี figure / page_number) — วัดอย่างเดียว ยังไม่ใช้ตัดสิน
- ``ocr:text_layer_recall`` = สัดส่วนคำของชั้นข้อความเดิมที่เจอในผล OCR: จับ OCR ที่ **ตกหล่น** หรืออ่านคำเพี้ยนไป
  ข้อจำกัด: ถ้าชั้นข้อความเดิมเองพัง (mojibake) recall จะต่ำทั้งที่ OCR ถูก; ชั้นข้อความที่มีแค่บางส่วนของหน้า recall อาจสูงทั้งที่ OCR ที่เหลือผิด
- ``ocr:text_layer_precision`` = สัดส่วนคำในผล OCR ที่เจอในชั้นข้อความเดิม: จับคำที่ OCR **เติมขึ้นมาเอง**
  ข้อจำกัด: ใช้ได้เฉพาะหน้าที่ชั้นข้อความครอบคลุมทั้งหน้า ถ้าชั้นข้อความมีแค่ส่วนหนึ่ง (เนื้อหาที่เหลือเป็นภาพ) precision ต่ำเป็นปกติ
  และคำที่ชั้นข้อความพัง ก็จะนับว่า OCR "เติมเอง"
- ทั้งสองตัวเทียบเป็นคำ ไม่เห็นความผิดที่ทำให้ได้คำที่มีอยู่จริงในชั้นข้อความ (เช่นสลับตัวเลข) และไม่ดูลำดับการอ่าน
- หน้าที่ไม่มีชั้นข้อความเดิม (ข้อความว่าง) ไม่ใส่ signal ทั้งสอง

``ocr:repeated_lines`` = จำนวนบรรทัดที่เกินจากบรรทัดแรก ของบรรทัดที่เหมือนกันทุกตัวและปรากฏ >= ``REPEATED_LINE_MIN`` ครั้ง
ใน ``Unit.text`` (หลัง normalize ไม่นับบรรทัดว่าง; ผลรวมของ (จำนวนครั้ง - 1) ต่อบรรทัดที่ซ้ำ) จับโมเดลพิมพ์วน — วัดอย่างเดียว
ยังไม่ใช้ตัดสิน ข้อจำกัด: เอกสารจริงมีบรรทัดซ้ำได้ (ส่วนหัว/ท้ายหน้า ช่องตาราง "-" "0") ค่าสูงจึงแปลว่า "ควรดู" ไม่ใช่ "ผิด"

backend: interface ``OcrBackend`` รับภาพ PNG ของหนึ่งภาพ คืน ``OcrResult``; เลือกจาก config (env ``OCR_BACKEND``,
ค่าเริ่มต้น ``typhoon-local`` = รันบนเครื่อง; ``tesseract-local`` = Tesseract บน CPU ไว้เทียบความแม่น); backend ที่ส่งข้อมูลออกนอกเครื่อง (``sends_data_offsite = True``)
``run_ocr`` ปฏิเสธเว้นแต่เปิดอนุญาตชัดเจน (พารามิเตอร์ ``allow_offsite`` หรือ env ``OCR_ALLOW_OFFSITE=1``)

แคช (``ocr_cache.py``): ``run_ocr`` ใช้แคชราย "หน้า" ก่อนเสมอ (``use_cache=False`` = ข้าม, ``refresh_cache=True`` = ไม่อ่านแต่เขียนทับ)
signal ``ocr:cache_hit`` = 1 เมื่อได้ผลจากแคช (``ocr:seconds`` คือเวลาที่ใช้จริงตอนรันครั้งแรก)
"""

import hashlib
import io
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol, Union

import numpy as np
import pymupdf

from app.extraction.image import load_oriented_image
from app.extraction.ocr_cache import OcrCache, make_key, make_key_fields
from app.extraction.ocr_markdown import parse_ocr_markdown
from app.extraction.pdf import _norm_signals
from app.extraction.quality import evaluate_unit
from app.models import (
    Block,
    BlockType,
    DetectedType,
    Document,
    ExtractionMethod,
    QualityFlag,
    QualityReport,
    Table,
    Unit,
    UnitStatus,
)
from app.thai.normalizer import normalize_with_report

OCR_RENDER_DPI = 200  # ความละเอียดตอน render หน้า PDF เป็นภาพ — ปรับที่เดียว
WIDE_PAGE_RATIO = 1.3  # กว้าง/สูง เกินนี้ = หน้าคู่ -> ตัดซ้าย-ขวา
GUTTER_WINDOW = (0.35, 0.65)  # หาร่องเฉพาะช่วงนี้ของความกว้าง (สัดส่วน 0–1)
GUTTER_MIN_WIDTH = 0.02  # ช่องว่างแนวตั้งต้องกว้างอย่างน้อยเท่านี้ของความกว้างหน้า (ในช่วง window) ถึงนับเป็นร่อง
GUTTER_INK_GRAY = 160  # พิกเซลที่เข้มกว่านี้ (ขาวดำ 0–255) นับเป็นหมึก
GUTTER_ROW_BAND = (0.15, 0.90)  # ดูหมึกเฉพาะช่วงความสูงนี้ (ข้ามหัว/ท้ายหน้า)
GUTTER_NOISE_FRACTION = 0.002  # คอลัมน์ที่มีหมึกไม่เกินสัดส่วนนี้ของแถวใน band นับเป็นว่าง (กันจุดฝุ่นจากการสแกน)
GUTTER_PROBE_DPI = 100  # ความละเอียดของภาพที่ใช้หาร่อง (PDF)
REPEATED_LINE_MIN = 3  # บรรทัดเหมือนกันตั้งแต่กี่ครั้งถึงนับเป็นการพิมพ์วน
DEFAULT_BACKEND = "typhoon-local"
SUPPORTED_TYPES = (DetectedType.PDF, DetectedType.JPG, DetectedType.PNG)
PART_SEPARATOR = "\n\n"


class OcrError(Exception):
    """OCR ใช้ไม่ได้ทั้งงาน (ไม่ใช่ความล้มเหลวรายหน้า)"""


class OcrNotAllowed(OcrError):
    """backend ส่งข้อมูลออกนอกเครื่องแต่ไม่ได้เปิดอนุญาต"""


class OcrUnavailable(OcrError):
    """backend ใช้งานไม่ได้ (เช่น ไม่มี GPU / ไม่มีโมเดล) — ห้ามถอยไปทางอื่นเงียบๆ"""


@dataclass
class OcrResult:
    text: str  # markdown (อาจมี <table> / <figure> / <page_number>)
    backend: str
    model: str
    revision: str
    seconds: float
    tokens: Optional[int] = None  # จำนวนโทเคนที่สร้าง (ถ้า backend รายงาน)
    extra: Optional[dict] = None  # ข้อมูลเฉพาะ backend (JSON ได้) เช่น บรรทัด+bbox ของ Tesseract; หน้าตัดครึ่งเก็บเป็น {"parts": [...]}


class OcrBackend(Protocol):
    name: str
    sends_data_offsite: bool

    def recognize(self, image_png: bytes) -> OcrResult: ...

    def cache_params(self) -> dict: ...  # พารามิเตอร์ที่มีผลต่อข้อความ (เข้า key ของแคช); ไม่มี = {}


class FakeBackend:
    """backend ปลอมสำหรับ test: ``texts`` เป็นข้อความเดียว, list (ตามลำดับที่เรียก วนซ้ำ) หรือฟังก์ชัน (ลำดับ call -> ข้อความ)
    ถ้าได้ ``Exception`` ให้โยนแทนการคืนข้อความ"""

    name = "fake"
    model = "fake-model"
    revision = "fake-rev"

    def __init__(self, texts: Union[str, list, Callable[[int], object]] = "ข้อความปลอม", sends_data_offsite: bool = False):
        self.texts = texts
        self.sends_data_offsite = sends_data_offsite
        self.calls = 0
        self.images: list[bytes] = []

    def cache_params(self) -> dict:
        return {"texts": repr(self.texts)}  # ข้อความต่างกัน = key ต่างกัน

    def recognize(self, image_png: bytes) -> OcrResult:
        n = self.calls
        self.calls += 1
        self.images.append(image_png)
        t = self.texts(n) if callable(self.texts) else self.texts[n % len(self.texts)] if isinstance(self.texts, list) else self.texts
        if isinstance(t, Exception):
            raise t
        return OcrResult(text=t, backend=self.name, model=self.model, revision=self.revision, seconds=0.01)


def get_backend(name: Optional[str] = None) -> OcrBackend:
    """เลือก backend จาก config (``OCR_BACKEND``); ค่าเริ่มต้น = รันบนเครื่อง"""
    name = name or os.environ.get("OCR_BACKEND", DEFAULT_BACKEND)
    if name == "typhoon-local":
        from app.extraction.ocr_typhoon import TyphoonLocalBackend  # import หนัก (torch) เมื่อใช้จริงเท่านั้น

        return TyphoonLocalBackend()
    if name == "tesseract-local":
        from app.extraction.ocr_tesseract import TesseractLocalBackend

        return TesseractLocalBackend()
    raise OcrUnavailable(f"ไม่รู้จัก OCR backend: {name}")


def offsite_allowed() -> bool:
    return os.environ.get("OCR_ALLOW_OFFSITE", "").strip().lower() in ("1", "true", "yes")


# ---------- ภาพของหน้า ----------

def _png(im) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def find_gutter(gray: "np.ndarray") -> tuple[float, bool]:
    """หาจุดตัดหน้ากว้างจากภาพขาวดำ (แถว x คอลัมน์, 0–255): กลางช่องว่างแนวตั้ง (คอลัมน์ที่แทบไม่มีหมึกใน
    ``GUTTER_ROW_BAND``) ที่กว้างที่สุดภายใน ``GUTTER_WINDOW``; คืน (สัดส่วนของความกว้าง, ใช้กลางหน้าแทนหรือไม่)
    ช่องว่างแคบกว่า ``GUTTER_MIN_WIDTH`` หรือไม่มีเลย -> (0.5, True) ช่องว่างที่ยื่นออกนอก window ถูกตัดให้เหลือแค่ในช่วง"""
    h, w = gray.shape
    band = gray[int(GUTTER_ROW_BAND[0] * h): max(int(GUTTER_ROW_BAND[1] * h), int(GUTTER_ROW_BAND[0] * h) + 1)]
    ink = (band < GUTTER_INK_GRAY).sum(axis=0)
    empty = ink <= GUTTER_NOISE_FRACTION * band.shape[0]
    lo, hi = int(GUTTER_WINDOW[0] * w), int(GUTTER_WINDOW[1] * w)
    best: Optional[tuple[int, int]] = None
    start: Optional[int] = None
    for x in range(lo, hi + 1):
        free = x < hi and bool(empty[x])
        if free and start is None:
            start = x
        elif not free and start is not None:
            if best is None or x - start > best[1] - best[0]:
                best = (start, x)
            start = None
    if best is None or (best[1] - best[0]) < GUTTER_MIN_WIDTH * w:
        return 0.5, True
    return (best[0] + best[1]) / 2 / w, False


@dataclass
class PageImages:
    images: list[bytes]  # หนึ่งภาพ หรือสองภาพ (ซ้าย, ขวา)
    split_x: Optional[float] = None  # สัดส่วนความกว้างที่ตัด (เฉพาะหน้าที่ถูกตัด)
    split_fallback: bool = False  # True = ไม่เจอร่อง ใช้กลางหน้า
    dpi: Optional[int] = None  # dpi ที่ render (PDF); ภาพ = None

    @property
    def split_method(self) -> str:
        return "none" if self.split_x is None else ("center" if self.split_fallback else "gutter")


def render_page_images(path: Path, detected: DetectedType, page_no: int, dpi: Optional[int] = None) -> PageImages:
    """ภาพ PNG ที่ส่งให้ backend: หนึ่งภาพต่อหน้า หรือสองภาพ (ซ้าย, ขวา) ถ้าหน้ากว้างเกิน ``WIDE_PAGE_RATIO``
    PDF: render ด้วย PyMuPDF ที่ ``OCR_RENDER_DPI`` (ครึ่งหน้า render ด้วย clip; หาร่องจากภาพที่ ``GUTTER_PROBE_DPI``)
    jpg/png: เปิดด้วย PIL หมุนตาม EXIF แล้ว"""
    if detected == DetectedType.PDF:
        dpi = dpi or OCR_RENDER_DPI
        with pymupdf.open(str(path)) as doc:
            page = doc.load_page(page_no - 1)
            r = page.rect
            if r.width / r.height <= WIDE_PAGE_RATIO:
                return PageImages([page.get_pixmap(dpi=dpi).tobytes("png")], dpi=dpi)
            probe = page.get_pixmap(dpi=GUTTER_PROBE_DPI, colorspace=pymupdf.csGRAY)
            gray = np.frombuffer(probe.samples, np.uint8).reshape(probe.height, probe.width)
            frac, fallback = find_gutter(gray)
            cut = r.x0 + frac * r.width
            return PageImages(
                [
                    page.get_pixmap(dpi=dpi, clip=pymupdf.Rect(r.x0, r.y0, cut, r.y1)).tobytes("png"),
                    page.get_pixmap(dpi=dpi, clip=pymupdf.Rect(cut, r.y0, r.x1, r.y1)).tobytes("png"),
                ],
                frac, fallback, dpi,
            )
    if detected in (DetectedType.JPG, DetectedType.PNG):
        im = load_oriented_image(path)
        if im.width / im.height <= WIDE_PAGE_RATIO:
            return PageImages([_png(im)])
        frac, fallback = find_gutter(np.asarray(im.convert("L")))
        x = min(max(int(round(frac * im.width)), 1), im.width - 1)
        return PageImages([_png(im.crop((0, 0, x, im.height))), _png(im.crop((x, 0, im.width, im.height)))],
                          frac, fallback)
    raise OcrError(f"OCR ยังไม่รองรับไฟล์ชนิด {detected.value}")


# ---------- ความสอดคล้องกับชั้นข้อความเดิม ----------

_TAG = re.compile(r"<[^>]+>")
_WORDISH = re.compile(r"[ก-๛A-Za-z0-9]")


def _tokens(text: str) -> list[str]:
    from pythainlp.tokenize import word_tokenize

    text = _TAG.sub(" ", text)
    return [t for t in word_tokenize(text, engine="newmm", keep_whitespace=False) if _WORDISH.search(t)]


def text_layer_agreement(text_layer: str, ocr_text: str) -> Optional[tuple[float, float]]:
    """(recall, precision) ของคำ (multiset) ระหว่างชั้นข้อความเดิมกับผล OCR; ไม่มีชั้นข้อความ (ไม่มีโทเคน) -> None
    recall = คำของชั้นข้อความที่เจอใน OCR ÷ คำของชั้นข้อความ; precision = คำของ OCR ที่เจอในชั้นข้อความ ÷ คำของ OCR
    (OCR ไม่มีโทเคนเลย -> precision 0.0) ทั้งสองฝั่งต้อง normalize แล้ว"""
    base = _tokens(text_layer)
    if not base:
        return None
    ocr = _tokens(ocr_text)
    pool: dict[str, int] = {}
    for t in ocr:
        pool[t] = pool.get(t, 0) + 1
    hit = 0
    for t in base:
        if pool.get(t, 0) > 0:
            pool[t] -= 1
            hit += 1
    return hit / len(base), (hit / len(ocr) if ocr else 0.0)


# ---------- ต่อ unit ----------

def repeated_lines(text: str) -> int:
    """จำนวนบรรทัดที่เกิน (นับ - 1 ต่อบรรทัด) ของบรรทัดที่เหมือนกันทุกตัวและปรากฏ >= ``REPEATED_LINE_MIN`` ครั้ง
    ไม่นับบรรทัดว่าง (หลัง strip)"""
    counts: dict[str, int] = {}
    for line in text.splitlines():
        line = line.strip()
        if line:
            counts[line] = counts.get(line, 0) + 1
    return sum(n - 1 for n in counts.values() if n >= REPEATED_LINE_MIN)


def _ocr_info(res: OcrResult) -> str:
    return f"OCR: {res.backend} · {res.model}@{res.revision}"


def _failed(unit: Unit, error: Exception, elapsed_ms: float) -> Unit:
    return Unit(
        index=unit.index,
        type=unit.type,
        position=unit.position,
        label=unit.label,
        status=UnitStatus.FAILED,
        reason=f"OCR {unit.label} ไม่สำเร็จ: {type(error).__name__}: {error}",
        method=ExtractionMethod.OCR,
        text="",
        text_hash=hashlib.sha256(b"").hexdigest(),
        quality=QualityReport(),
        elapsed_ms=elapsed_ms,
        signals={"char_count": 0.0, "block_count": 0.0},
    )


def _ocr_unit(unit: Unit, res: OcrResult, elapsed_ms: float, page: Optional[PageImages] = None,
              cache_hit: bool = False) -> Unit:
    parsed = parse_ocr_markdown(res.text)
    blocks: list[Block] = []
    fixes: dict[str, int] = {}
    sigs: dict[str, int] = {}

    def norm(raw: str, count: bool = True) -> str:
        n = normalize_with_report(raw)
        if count:  # คำบรรยายภาพไม่ใช่ข้อความของเอกสาร ผลนับของมันไม่เข้าด่านตรวจ
            for k, v in n.fixes.items():
                fixes[k] = fixes.get(k, 0) + v
            for k, v in n.signals.items():
                sigs[k] = sigs.get(k, 0) + v
        return n.text.strip()

    for rb in parsed.blocks:
        if rb.type == BlockType.TABLE:
            rows = [[norm(c) for c in row] for row in rb.table.rows]
            text = "\n".join(" | ".join(c for c in row if c) for row in rows if any(row))
            if not text:
                continue
            table = Table(rows=rows, header_rows=rb.table.header_rows, merged_cells=rb.table.merged_cells)
            blocks.append(Block(type=BlockType.TABLE, text=text, order=len(blocks), table=table))
            continue
        text = norm(rb.text, count=rb.type != BlockType.IMAGE)
        if text:
            blocks.append(Block(type=rb.type, text=text, order=len(blocks), level=rb.level))
    text = "\n".join(b.text for b in blocks if b.type != BlockType.IMAGE)
    signals = {
        "char_count": float(len(text.strip())),
        "block_count": float(len(blocks)),
        **_norm_signals(fixes, sigs),
        "ocr:seconds": res.seconds,
        "ocr:figure_count": float(parsed.figure_count),
        "ocr:repeated_lines": float(repeated_lines(text)),
    }
    if res.tokens is not None:
        signals["ocr:tokens"] = float(res.tokens)
    if cache_hit:
        signals["ocr:cache_hit"] = 1.0
    if page is not None and page.split_x is not None:
        signals["ocr:split_x"] = float(page.split_x)
        signals["ocr:split_fallback"] = 1.0 if page.split_fallback else 0.0
    for i, n in enumerate(parsed.page_numbers[:2]):
        signals["ocr:page_number" if i == 0 else "ocr:page_number_2"] = float(n)
    agreement = text_layer_agreement(unit.text, text) if unit.text.strip() else None
    if agreement is not None:
        signals["ocr:text_layer_recall"], signals["ocr:text_layer_precision"] = agreement
    new = Unit(
        index=unit.index,
        type=unit.type,
        position=unit.position,
        label=unit.label,
        status=UnitStatus.OK,
        method=ExtractionMethod.OCR,
        blocks=blocks,
        text=text,
        text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        quality=QualityReport(flags=[] if text.strip() else [QualityFlag.EMPTY],
                              normalization_changes=sum(fixes.values())),
        elapsed_ms=elapsed_ms,
        signals=signals,
    )
    new = evaluate_unit(new)
    info = _ocr_info(res)
    reason = info if new.status == UnitStatus.OK else f"{new.reason} ({info})"
    return new.model_copy(update={"reason": reason})


def _recognize_page(backend: OcrBackend, images: list[bytes]) -> OcrResult:
    results = [backend.recognize(img) for img in images]
    first = results[0]
    counted = [r.tokens for r in results if r.tokens is not None]
    extra = None
    if any(r.extra is not None for r in results):  # ต่อภาพ (ซ้าย, ขวา) ตามลำดับ
        extra = {"parts": [r.extra for r in results]}
    return OcrResult(
        text=PART_SEPARATOR.join(r.text for r in results),
        backend=first.backend, model=first.model, revision=first.revision,
        seconds=sum(r.seconds for r in results),
        tokens=sum(counted) if counted else None,
        extra=extra,
    )


def backend_params(backend: OcrBackend) -> dict:
    fn = getattr(backend, "cache_params", None)
    return dict(fn()) if callable(fn) else {}


def recognize_page_cached(
    backend: OcrBackend, path: Path, detected: DetectedType, content_hash: str, page_no: int,
    *, use_cache: bool = True, refresh_cache: bool = False, cache: Optional[OcrCache] = None,
) -> tuple[OcrResult, PageImages, bool]:
    """OCR หนึ่งหน้า (ผ่านแคช): คืน (ผลดิบ, ภาพที่ส่งโมเดล, ได้จากแคชหรือไม่)"""
    page = render_page_images(path, detected, page_no)
    fields = make_key_fields(
        content_hash=content_hash, page=page_no, backend=backend.name,
        revision=getattr(backend, "revision", ""), dpi=page.dpi, split_method=page.split_method,
        split_x=page.split_x, params=backend_params(backend),
    )
    key = make_key(fields)
    store = cache or OcrCache()
    if use_cache and not refresh_cache:
        hit = store.get(key)
        if hit is not None:
            return OcrResult(text=hit["text"], backend=hit["backend"], model=hit["model"], revision=hit["revision"],
                             seconds=hit["seconds"], tokens=hit.get("tokens"), extra=hit.get("extra")), page, True
    res = _recognize_page(backend, page.images)
    if use_cache:
        store.put(key, fields, res)
    return res, page, False


def run_ocr(
    document: Document,
    source_path: str | Path,
    backend: OcrBackend,
    allow_offsite: Optional[bool] = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> Document:
    """OCR เฉพาะ unit ที่ needs_ocr; คืน Document ใหม่ (ของเดิมไม่ถูกแก้)"""
    allowed = offsite_allowed() if allow_offsite is None else allow_offsite
    if backend.sends_data_offsite and not allowed:
        raise OcrNotAllowed(
            f"backend {backend.name} ส่งข้อมูลออกนอกเครื่อง ต้องเปิดอนุญาตชัดเจน (OCR_ALLOW_OFFSITE=1)"
        )
    todo = [u for u in document.units if u.status == UnitStatus.NEEDS_OCR]
    if document.error is not None or not todo:
        return document
    if document.detected_type not in SUPPORTED_TYPES:
        raise OcrError(f"OCR รองรับเฉพาะ PDF / jpg / png (ไฟล์นี้เป็น {document.detected_type.value})")
    path = Path(source_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != document.content_hash:
        raise OcrError("source_path ไม่ใช่ไฟล์เดียวกับที่ extract ไว้ (content_hash ไม่ตรง)")

    out: list[Unit] = []
    for unit in document.units:
        if unit.status != UnitStatus.NEEDS_OCR:
            out.append(unit)
            continue
        t0 = time.perf_counter()
        try:
            res, page, hit = recognize_page_cached(
                backend, path, document.detected_type, document.content_hash, int(unit.position["page"]),
                use_cache=use_cache, refresh_cache=refresh_cache,
            )
            out.append(_ocr_unit(unit, res, (time.perf_counter() - t0) * 1000, page, hit))
        except Exception as e:  # รายหน้า: ไม่ให้หน้าเดียวล้มทั้งไฟล์
            out.append(_failed(unit, e, (time.perf_counter() - t0) * 1000))
    return document.model_copy(update={"units": out})
