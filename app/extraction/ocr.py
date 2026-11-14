"""OCR เป็นขั้นแยกจาก ``extract_file`` (T06a)

``extract_file`` ไม่เรียก OCR ข้างใน: มันทำเครื่องหมาย unit ที่ต้อง OCR (status = needs_ocr) แล้วหยุด
``run_ocr(document, source_path, backend)`` คืน Document ใหม่ที่ OCR เฉพาะ unit ที่ needs_ocr
- unit ที่ไม่ใช่ needs_ocr ไม่ถูกแตะ (คืนวัตถุเดิม)
- unit ที่ OCR แล้ว: method = ocr เสมอ; ผลของโมเดลถูกแปลงเป็น block (``ocr_markdown.py``) แล้วข้อความทุกส่วนผ่าน
  ``normalize_with_report`` (ตัวเดียวของระบบ) แล้วผ่านด่านตรวจ ``evaluate_unit`` ตัวเดิมรอบใหม่
  (ไม่เรียกกฎระดับไฟล์ ``apply_document_rule`` เพราะจะไปแตะ unit อื่น) ผ่าน = ok; ไม่ผ่าน (OCR ออกมาพัง) = **failed**
  reason "OCR แล้วไม่ผ่านด่านตรวจ: <flag ของด่านตรวจ> — <เหตุผลของด่านตรวจ> (OCR: ...)" ไม่กลับไปเป็น needs_ocr
  (ไม่งั้นวนส่ง OCR ซ้ำ และ contract "unit ที่ OCR ล้มเหลว = failed"); ข้อความ OCR ยังเก็บไว้ใน unit ให้ตรวจย้อนหลัง
  (unit ที่ไม่ใช่ ok ไม่ถูก chunk/index); ``image_coverage`` ของ unit เดิมไม่ถูกยกมา (หน้าภาพที่ OCR แล้วได้ตัวอักษรน้อย
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

backend: interface ``OcrBackend`` รับภาพ PNG ของหนึ่งภาพ คืน ``OcrResult``; เลือกจาก config (``Settings.ocr_backend`` /
env ``OCR_BACKEND`` = auto | typhoon-local | tesseract-local; auto = ``typhoon-local`` (GPU) เมื่อมี GPU, ``tesseract-local``
(CPU) เมื่อไม่มี — ตัดสินและ log ที่ ``app/runtime.py``; ทั้งสองรันบนเครื่อง); backend ที่ส่งข้อมูลออกนอกเครื่อง (``sends_data_offsite = True``)
``run_ocr`` ปฏิเสธเว้นแต่เปิดอนุญาตชัดเจน (พารามิเตอร์ ``allow_offsite`` หรือ env ``OCR_ALLOW_OFFSITE=1``)

ชนเพดานโทเคน / พิมพ์วน (ใช้กับทุก backend):
- ผลของ backend เก็บสถิติรายชิ้นภาพใน ``OcrResult.part_stats`` ([{"tokens", "chars"}] ตามลำดับชิ้น) เพื่อรู้ว่าชิ้นไหน
  ชนเพดาน และแยกข้อความกลับเป็นรายชิ้นได้ (ผลในแคชรุ่นเก่าไม่มี -> ถือทั้งหน้าเป็นชิ้นเดียว และนับว่าชนเพดานเมื่อ
  โทเคนรวม >= เพดาน ซึ่งเป็นการประมาณข้างมาก: หน้าคู่ที่สองครึ่งรวมกันเกินเพดานแต่ไม่มีครึ่งไหนชนจะถูกนับว่าชนด้วย)
- ``ocr_loop.trim_tail_loop`` ตัดลูปท้ายผลดิบ "ทีละชิ้น" ก่อนแปลงเป็น block (เก็บข้อความก่อนลูป + ท่อนซ้ำหนึ่งรอบ ไม่เติม
  ไม่เดา) แคชเก็บผลดิบเสมอ (ปรับตัวตรวจแล้วไม่ต้อง OCR ใหม่)
- signals: ``ocr:hit_max_tokens`` = จำนวนชิ้นที่ชนเพดาน (ใส่เมื่อ backend มีเพดาน ``max_new_tokens``),
  ``ocr:loop_trimmed_chars`` = อักขระที่ตัดทิ้ง; ค่าใด > 0 -> flag ``ocr_incomplete`` (status ไม่เปลี่ยนเพราะเรื่องนี้)
- OCR ซ้ำแบบแบ่งบน/ล่าง (resplit, ``recognize_page_auto``): ผลรอบแรกมีชิ้นชนเพดาน -> แบ่งแต่ละครึ่งหน้า (หรือทั้งหน้า
  ถ้าไม่ได้ตัดซ้ายขวา) เป็นบน/ล่างที่ช่องว่างแนวนอน (``find_row_gap``) ที่ใกล้กลางที่สุด OCR ทีละชิ้นแล้วต่อตามลำดับ
  (ซ้าย-บน, ซ้าย-ล่าง, ขวา-บน, ขวา-ล่าง); ชิ้นไหนยังชนเพดาน แบ่งบน/ล่างอีก 1 ระดับ (``RESPLIT_MAX_DEPTH``) ถ้ายังชนอีก
  ใช้ผลนั้น (ตัดลูป + flag) · ไม่เจอช่องว่าง = ไม่ตัด (ห้ามตัดกลางบรรทัด) OCR ทั้งชิ้นแล้วนับใน ``ocr:resplit_no_gap``
  แคชรายชิ้น: split_method = "<วิธีเดิม>+rows" + ``region`` ของชิ้น (ไม่แตะ key ของผลรอบแรก/หน้าอื่น; GPU ล่มกลางหน้า
  แล้วรันใหม่ ชิ้นที่เสร็จแล้วไม่ต้องทำซ้ำ) · signals เพิ่ม: ``ocr:resplit`` = 1, ``ocr:first_pass_tokens``, ``ocr:pieces``,
  ``ocr:resplit_no_gap``, ``ocr:resplit_rotated``; ``ocr:tokens`` / ``ocr:seconds`` = ของรอบ resplit (ไม่รวมรอบแรก)
- ข้อความหมุน (PDF, ตอน resplit): ครึ่งหน้าที่อักขระในชั้นข้อความเป็นบรรทัดไม่ใช่แนวนอนตั้งแต่ ``ROTATED_TEXT_SHARE``
  -> หมุนภาพให้ตั้งตรงก่อน (``text_orientation``: ทิศจาก dir ของบรรทัดที่มีอักขระมากที่สุด, ``rotate_piece``) แล้วแบ่ง
  บน/ล่างของภาพที่ตั้งตรงตามปกติ (บนของภาพที่หมุนตามเข็ม 90° = ซ้ายของหน้า) · region ในแคชเป็นพิกัดหน้าเสมอ + key
  ``rotation`` (องศาตามเข็ม) · ครึ่งที่หมุนแล้วแบ่งไม่ได้ OCR ภาพที่หมุนทั้งชิ้น (ไม่ใช้ผลรอบแรกซึ่งเป็นภาพตะแคง)
  · หมุนเฉพาะตอน resplit (รอบแรกไม่หมุน: key ของผลรอบแรกไม่เปลี่ยน)
- ช่องว่างแนวนอน (``find_row_gap``): แถวพิกเซลที่ไม่มี "หมึก" เลย (หมึก = ต่างจากค่ากลางของแถวนั้นเกิน ``ROW_INK_DELTA``
  จึงใช้ได้กับพื้นสี) ไม่นับเส้นแนวตั้งบางยาว (เส้นตาราง: ยาว >= ``RULE_MIN_LEN_PX`` กว้าง <= ``RULE_MAX_WIDTH_PX``) เพื่อ
  ตัดระหว่างแถวของตารางที่มีเส้นได้ · ช่องต้องสูง >= ``ROW_GAP_MIN_PX`` และกลางช่องอยู่ใน ``ROW_GAP_WINDOW`` ของความสูง

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
from app.extraction.ocr_loop import trim_tail_loop
from app.extraction.ocr_markdown import parse_ocr_markdown
from app.extraction.pdf import _norm_signals
from app.extraction.quality import evaluate_unit
from app.runtime import is_fatal_cuda
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
# resplit: แบ่งบน/ล่างเมื่อชนเพดานโทเคน (ค่าพิกเซลคิดที่ OCR_RENDER_DPI)
ROW_GAP_WINDOW = (0.2, 0.8)  # กลางช่องว่างต้องอยู่ในช่วงนี้ของความสูงชิ้น
ROW_GAP_MIN_PX = 4  # ช่องว่างแนวนอนต้องสูงอย่างน้อยเท่านี้
ROW_INK_DELTA = 60  # ต่างจากค่ากลาง (median) ของแถวเกินนี้ = หมึก
RULE_MIN_LEN_PX = 80  # เส้นแนวตั้งยาวอย่างน้อยเท่านี้ (~1 ซม.) ...
RULE_MAX_WIDTH_PX = 4  # ... และกว้างไม่เกินนี้ = เส้นตาราง ไม่นับเป็นหมึก (ขีดของตัวอักษรใหญ่กว้างกว่านี้)
RESPLIT_MAX_DEPTH = 2  # 1 = บน/ล่างของครึ่งหน้า, 2 = แบ่งชิ้นที่ยังชนเพดานอีกครั้ง
# ชิ้นที่อักขระในชั้นข้อความ PDF เป็นข้อความหมุน (บรรทัดแนวตั้ง dir != (1, 0)) ตั้งแต่สัดส่วนนี้ = หมุนภาพให้ตั้งตรง
# ก่อนแบ่งบน/ล่าง (ถ้าตัดแนวนอนบนภาพตะแคง จะแยกช่องของแถวเดียวกันออกจากกัน) — 0.2 ไม่ใช่ 0.5 เพราะครึ่งหน้า
# ซ้ายของ KBANK หน้า 132 มีแถบสารบัญแนวนอนเจือ ได้ 0.39 ทั้งที่ตารางเป็นข้อความหมุน; หน้าที่ไม่หมุนได้ 0.0
ROTATED_TEXT_SHARE = 0.2
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
    # สถิติรายชิ้นภาพ (ระดับหน้า): [{"tokens": int|None, "chars": int}] ตามลำดับที่ต่อด้วย PART_SEPARATOR; None = ไม่รู้ (แคชเก่า)
    part_stats: Optional[list] = None


class OcrBackend(Protocol):
    name: str
    sends_data_offsite: bool

    def recognize(self, image_png: bytes) -> OcrResult: ...

    def cache_params(self) -> dict: ...  # พารามิเตอร์ที่มีผลต่อข้อความ (เข้า key ของแคช); ไม่มี = {}


class FakeBackend:
    """backend ปลอมสำหรับ test: ``texts`` เป็นข้อความเดียว, list (ตามลำดับที่เรียก วนซ้ำ) หรือฟังก์ชัน (ลำดับ call -> ข้อความ)
    ถ้าได้ ``Exception`` ให้โยนแทนการคืนข้อความ; ได้ tuple (ข้อความ, tokens) = รายงานจำนวนโทเคนด้วย
    ``max_new_tokens`` = เพดานโทเคน (None = ไม่มีเพดาน เหมือน backend ที่ไม่ใช่ generate)"""

    name = "fake"
    model = "fake-model"
    revision = "fake-rev"

    def __init__(self, texts: Union[str, list, Callable[[int], object]] = "ข้อความปลอม", sends_data_offsite: bool = False,
                 max_new_tokens: Optional[int] = None):
        self.texts = texts
        self.max_new_tokens = max_new_tokens
        self.sends_data_offsite = sends_data_offsite
        self.calls = 0
        self.images: list[bytes] = []

    def cache_params(self) -> dict:
        return {"texts": repr(self.texts), "max_new_tokens": self.max_new_tokens}  # ข้อความต่างกัน = key ต่างกัน

    def recognize(self, image_png: bytes) -> OcrResult:
        n = self.calls
        self.calls += 1
        self.images.append(image_png)
        t = self.texts(n) if callable(self.texts) else self.texts[n % len(self.texts)] if isinstance(self.texts, list) else self.texts
        if isinstance(t, Exception):
            raise t
        tokens = None
        if isinstance(t, tuple):
            t, tokens = t
        return OcrResult(text=t, backend=self.name, model=self.model, revision=self.revision, seconds=0.01, tokens=tokens)


def get_backend(name: Optional[str] = None, settings=None) -> OcrBackend:
    """สร้าง OCR backend — ``name`` ไม่ระบุ = ตาม config (``Settings.ocr_backend`` / env ``OCR_BACKEND``, ค่าเริ่มต้น auto:
    typhoon-local เมื่อมี GPU, tesseract-local เมื่อไม่มี) ตัดสินและ log ที่ ``app.runtime.resolve_ocr_backend``
    ระบุ ``name`` เอง (สคริปต์เปรียบเทียบ) = ใช้ชื่อนั้นและ log ด้วยเช่นกัน — ไม่มีการสลับ backend ให้เงียบๆ"""
    from app.runtime import resolve_ocr_backend

    if name is None:
        if settings is None:
            from app.config import get_settings

            settings = get_settings()
        name = resolve_ocr_backend(settings.ocr_backend).name
    elif name in ("typhoon-local", "tesseract-local"):
        resolve_ocr_backend(name)  # log + เตือนถ้าเครื่องนี้ใช้ไม่ได้
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


# ---------- แบ่งบน/ล่าง (resplit) ----------

def _long_runs(mask: "np.ndarray", min_len: int) -> "np.ndarray":
    """พิกเซลที่อยู่ในช่วง True ต่อเนื่องตามแนวตั้ง (ในคอลัมน์เดียวกัน) ยาว >= ``min_len``"""
    h, w = mask.shape
    pad = np.zeros((1, w), dtype=np.int8)
    d = np.diff(np.vstack([pad, mask.astype(np.int8), pad]), axis=0)
    start_c, start_r = np.nonzero(d.T == 1)  # เรียงตามคอลัมน์แล้วแถว -> จับคู่ start/end ตามลำดับได้
    _end_c, end_r = np.nonzero(d.T == -1)
    out = np.zeros(mask.shape, dtype=bool)
    long = (end_r - start_r) >= min_len
    for c, r0, r1 in zip(start_c[long], start_r[long], end_r[long]):
        out[r0:r1, c] = True
    return out


def _ink_mask(gray: "np.ndarray") -> "np.ndarray":
    """หมึก = ต่างจากค่ากลางของแถวเกิน ``ROW_INK_DELTA`` ยกเว้นเส้นแนวตั้งบางยาว (เส้นตาราง)
    เส้นตารางหาจากหมึกเทียบพื้นของทั้งภาพ (ไม่ใช่รายแถว): แถวที่เป็นเส้นขอบแนวนอนมีค่ากลางเป็นสีเส้น ถ้าใช้ค่ารายแถว
    เส้นแนวตั้งจะขาดตรงเส้นขอบทุกแถวของตาราง แล้วสั้นกว่า ``RULE_MIN_LEN_PX``"""
    g = gray.astype(np.int16)
    ink = np.abs(g - np.median(g, axis=1)[:, None]) > ROW_INK_DELTA
    vertical = _long_runs(np.abs(g - np.median(g)) > ROW_INK_DELTA, RULE_MIN_LEN_PX)
    wide = _long_runs(vertical.T, RULE_MAX_WIDTH_PX + 1).T  # เส้นแนวตั้งที่หนาเกิน = ขีดตัวอักษร/ภาพ ยังนับเป็นหมึก
    return ink & ~(vertical & ~wide)


def find_row_gap(gray: "np.ndarray") -> Optional[int]:
    """แถว (พิกเซล) ที่ใช้ตัดบน/ล่าง: กลางช่องว่างแนวนอน (แถวที่ไม่มีหมึกเลย) สูง >= ``ROW_GAP_MIN_PX`` ที่กลางช่องอยู่ใน
    ``ROW_GAP_WINDOW`` และใกล้กลางภาพที่สุด; ไม่มี -> None (ห้ามตัดกลางบรรทัด)"""
    h = gray.shape[0]
    empty = ~_ink_mask(gray).any(axis=1)
    lo, hi = ROW_GAP_WINDOW[0] * h, ROW_GAP_WINDOW[1] * h
    best: Optional[tuple[float, int]] = None  # (ระยะจากกลาง, แถวที่ตัด)
    start: Optional[int] = None
    for y in range(h + 1):
        free = y < h and bool(empty[y])
        if free and start is None:
            start = y
        elif not free and start is not None:
            mid = (start + y) / 2
            if y - start >= ROW_GAP_MIN_PX and lo <= mid <= hi:
                cand = (abs(mid - h / 2), (start + y) // 2)
                if best is None or cand < best:
                    best = cand
            start = None
    return None if best is None else best[1]


@dataclass
class Piece:
    png: bytes  # ภาพที่ส่งโมเดล (ถ้า rotation != 0 คือภาพที่หมุนตั้งตรงแล้ว)
    region: tuple[float, float, float, float]  # x0, y0, x1, y1 เป็นสัดส่วนของหน้า (ใช้ใน cache key) — พิกัดหน้าเสมอ
    depth: int  # 0 = ครึ่งหน้า/ทั้งหน้า, 1 = บน/ล่าง, 2 = แบ่งอีกครั้ง
    rotation: int = 0  # องศาที่หมุนภาพตามเข็มนาฬิกาให้ตั้งตรง (0 / 90 / 180 / 270)


def page_pieces(page: PageImages) -> list[Piece]:
    if page.split_x is None:
        return [Piece(page.images[0], (0.0, 0.0, 1.0, 1.0), 0)]
    return [Piece(page.images[0], (0.0, 0.0, page.split_x, 1.0), 0),
            Piece(page.images[1], (page.split_x, 0.0, 1.0, 1.0), 0)]


# ทิศบรรทัด (dir ของ PyMuPDF ปัดแล้ว แกน y ชี้ลง) -> องศาที่ต้องหมุนภาพตามเข็มนาฬิกาให้บรรทัดวิ่งซ้ายไปขวา
_UPRIGHT_ROTATION = {(0, -1): 90, (0, 1): 270, (-1, 0): 180}


def text_orientation(path: Path, page_no: int,
                     region: tuple[float, float, float, float]) -> Optional[tuple[float, int]]:
    """(สัดส่วนข้อความหมุน, องศาที่ต้องหมุนภาพตามเข็มนาฬิกา) ของบรรทัดในชั้นข้อความ PDF ที่กลางบรรทัดอยู่ใน ``region``
    สัดส่วน = อักขระ (ไม่นับช่องว่าง) ของบรรทัดที่ไม่ใช่แนวนอน (dir ปัดแล้วไม่ใช่ (1, 0)) ÷ อักขระทั้งหมด
    องศา = ทิศที่มีอักขระมากที่สุดในบรรทัดที่หมุน (ไม่มีบรรทัดหมุน / ทิศเฉียงที่ไม่รู้จัก = 0)
    ไม่มีอักขระ -> None · ใช้แค่ทิศของบรรทัด (geometry) ไม่ได้ใช้เนื้อข้อความ"""
    x0, y0, x1, y1 = region
    horizontal = 0
    by_dir: dict[tuple[int, int], int] = {}
    with pymupdf.open(str(path)) as doc:
        page = doc.load_page(page_no - 1)
        r = page.rect
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                bx0, by0, bx1, by1 = line["bbox"]
                cx = ((bx0 + bx1) / 2 - r.x0) / r.width
                cy = ((by0 + by1) / 2 - r.y0) / r.height
                if not (x0 <= cx < x1 and y0 <= cy < y1):
                    continue
                n = sum(len(sp["text"].strip()) for sp in line["spans"])
                d = (round(line["dir"][0]), round(line["dir"][1]))
                if d == (1, 0):
                    horizontal += n
                else:
                    by_dir[d] = by_dir.get(d, 0) + n
    rotated = sum(by_dir.values())
    total = horizontal + rotated
    if not total:
        return None
    main = max(by_dir.items(), key=lambda kv: kv[1])[0] if rotated else None
    return rotated / total, _UPRIGHT_ROTATION.get(main, 0)


def rotated_text_share(path: Path, page_no: int, region: tuple[float, float, float, float]) -> Optional[float]:
    """สัดส่วนข้อความหมุนของ ``text_orientation`` (ไม่มีอักขระ -> None)"""
    o = text_orientation(path, page_no, region)
    return None if o is None else o[0]


def rotate_piece(piece: Piece, rotation: int) -> Piece:
    """หมุนภาพของชิ้นตามเข็มนาฬิกา ``rotation`` องศา (region คงเป็นพิกัดหน้าเดิม)"""
    from PIL import Image

    if not rotation:
        return piece
    im = Image.open(io.BytesIO(piece.png))
    im.load()
    ccw = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}
    return Piece(_png(im.transpose(ccw[rotation])), piece.region, piece.depth, (piece.rotation + rotation) % 360)


def _sub_region(region: tuple[float, float, float, float], rotation: int,
                f0: float, f1: float) -> tuple[float, float, float, float]:
    """ช่วง [f0, f1] ตามแนวตั้งของภาพที่หมุนแล้ว (0 = บนของภาพตั้งตรง) -> region ในพิกัดหน้า"""
    x0, y0, x1, y1 = region
    w, h = x1 - x0, y1 - y0
    if rotation == 90:  # หมุนตามเข็ม: บนของภาพตั้งตรง = ซ้ายของหน้า
        return (x0 + w * f0, y0, x0 + w * f1, y1)
    if rotation == 270:  # หมุนทวนเข็ม: บน = ขวาของหน้า
        return (x1 - w * f1, y0, x1 - w * f0, y1)
    if rotation == 180:
        return (x0, y1 - h * f1, x1, y1 - h * f0)
    return (x0, y0 + h * f0, x1, y0 + h * f1)


def split_piece_rows(piece: Piece) -> Optional[tuple[Piece, Piece]]:
    """แบ่งชิ้นเป็นบน/ล่าง (ของภาพที่ส่งโมเดล ซึ่งตั้งตรงแล้ว) ที่ ``find_row_gap`` (ครอปจากภาพเดิม ไม่ render ใหม่)
    ไม่มีช่องว่าง -> None"""
    from PIL import Image

    im = Image.open(io.BytesIO(piece.png))
    im.load()
    y = find_row_gap(np.asarray(im.convert("L")))
    if y is None:
        return None
    f = y / im.height
    return (Piece(_png(im.crop((0, 0, im.width, y))), _sub_region(piece.region, piece.rotation, 0.0, f),
                  piece.depth + 1, piece.rotation),
            Piece(_png(im.crop((0, y, im.width, im.height))), _sub_region(piece.region, piece.rotation, f, 1.0),
                  piece.depth + 1, piece.rotation))


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


OCR_UNAVAILABLE_PREFIX = "OCR ใช้ไม่ได้: "


def ocr_unavailable_unit(unit: Unit, error: Exception | str) -> Unit:
    """unit ที่ยังไม่ได้ OCR เพราะ OCR ใช้ไม่ได้ — สถานะคง needs_ocr, reason บอกเหตุผล (ส่งไฟล์ซ้ำเมื่อ OCR พร้อมจะ OCR ต่อ)"""
    return unit.model_copy(update={"status": UnitStatus.NEEDS_OCR, "reason": f"{OCR_UNAVAILABLE_PREFIX}{error}"})


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
              cache_hit: bool = False, max_tokens: Optional[int] = None,
              resplit: Optional["ResplitInfo"] = None) -> Unit:
    parts = result_parts(res)
    trims = [trim_tail_loop(t) for t, _tok in parts]
    loop_trimmed = sum(t.removed_chars for t in trims)
    hit_max = hits_ceiling(res, max_tokens)
    parsed = parse_ocr_markdown(PART_SEPARATOR.join(t.text for t in trims))
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
        "ocr:loop_trimmed_chars": float(loop_trimmed),
    }
    if max_tokens:
        signals["ocr:hit_max_tokens"] = float(hit_max)
    if resplit is not None:
        signals["ocr:resplit"] = 1.0
        signals["ocr:pieces"] = float(resplit.pieces)
        signals["ocr:resplit_no_gap"] = float(resplit.no_gap)
        signals["ocr:resplit_rotated"] = float(resplit.rotated)
        if resplit.first_pass_tokens is not None:
            signals["ocr:first_pass_tokens"] = float(resplit.first_pass_tokens)
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
        quality=QualityReport(flags=([] if text.strip() else [QualityFlag.EMPTY])
                              + ([QualityFlag.OCR_INCOMPLETE] if hit_max or loop_trimmed else []),
                              normalization_changes=sum(fixes.values())),
        elapsed_ms=elapsed_ms,
        signals=signals,
    )
    before = set(new.quality.flags)
    new = evaluate_unit(new)
    info = _ocr_info(res)
    if new.status == UnitStatus.OK:
        return new.model_copy(update={"reason": info})
    # OCR แล้วยังไม่ผ่านด่านตรวจ = OCR ล้มเหลว -> failed (ห้ามกลับไปเป็น needs_ocr: วนส่ง OCR ซ้ำไม่จบ)
    gate_flags = [f.value for f in new.quality.flags if f not in before] or [new.status.value]
    reason = f"OCR แล้วไม่ผ่านด่านตรวจ: {', '.join(gate_flags)} — {new.reason} ({info})"
    return new.model_copy(update={"status": UnitStatus.FAILED, "reason": reason})


def _combine(results: list[OcrResult]) -> OcrResult:
    """ต่อผลรายชิ้นตามลำดับ (PART_SEPARATOR) พร้อม ``part_stats``"""
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
        part_stats=[{"tokens": r.tokens, "chars": len(r.text)} for r in results],
    )


def _recognize_page(backend: OcrBackend, images: list[bytes]) -> OcrResult:
    return _combine([backend.recognize(img) for img in images])


def max_tokens_of(backend: OcrBackend) -> Optional[int]:
    """เพดานโทเคนของ backend (``max_new_tokens``); ไม่มี = None"""
    v = getattr(backend, "max_new_tokens", None)
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else None


def result_parts(res: OcrResult) -> list[tuple[str, Optional[int]]]:
    """แยกผลระดับหน้ากลับเป็นรายชิ้น [(ข้อความ, tokens)] ด้วย ``part_stats``; ไม่มี / ไม่ตรงกับข้อความ -> ทั้งหน้าเป็นชิ้นเดียว"""
    stats = res.part_stats
    if stats:
        lengths = [int(st.get("chars", -1)) for st in stats]
        if min(lengths) >= 0 and sum(lengths) + len(PART_SEPARATOR) * (len(lengths) - 1) == len(res.text):
            out, pos = [], 0
            for st, n in zip(stats, lengths):
                out.append((res.text[pos:pos + n], st.get("tokens")))
                pos += n + len(PART_SEPARATOR)
            return out
    return [(res.text, res.tokens)]


def hits_ceiling(res: OcrResult, max_tokens: Optional[int]) -> int:
    """จำนวนชิ้นที่โทเคน >= เพดาน (ไม่มี part_stats = ใช้โทเคนรวม: ประมาณข้างมาก)"""
    if not max_tokens:
        return 0
    return sum(1 for _t, tok in result_parts(res) if tok is not None and tok >= max_tokens)


def backend_params(backend: OcrBackend) -> dict:
    fn = getattr(backend, "cache_params", None)
    return dict(fn()) if callable(fn) else {}


def recognize_page_cached(
    backend: OcrBackend, path: Path, detected: DetectedType, content_hash: str, page_no: int,
    *, use_cache: bool = True, refresh_cache: bool = False, cache: Optional[OcrCache] = None,
) -> tuple[OcrResult, PageImages, bool]:
    """OCR หนึ่งหน้า (ผ่านแคช): คืน (ผลดิบ, ภาพที่ส่งโมเดล, ได้จากแคชหรือไม่)"""
    page = render_page_images(path, detected, page_no)
    fields = _key_fields(backend, content_hash, page_no, page, page.split_method)
    res, hit = _through_cache(fields, lambda: _recognize_page(backend, page.images),
                              use_cache=use_cache, refresh_cache=refresh_cache, store=cache or OcrCache())
    return res, page, hit


def _key_fields(backend: OcrBackend, content_hash: str, page_no: int, page: PageImages, split_method: str,
                region: Optional[tuple[float, float, float, float]] = None, rotation: int = 0) -> dict:
    return make_key_fields(
        content_hash=content_hash, page=page_no, backend=backend.name,
        revision=getattr(backend, "revision", ""), dpi=page.dpi, split_method=split_method,
        split_x=page.split_x, params=backend_params(backend), region=region, rotation=rotation,
    )


def _through_cache(fields: dict, compute: Callable[[], OcrResult], *, use_cache: bool, refresh_cache: bool,
                   store: OcrCache) -> tuple[OcrResult, bool]:
    key = make_key(fields)
    if use_cache and not refresh_cache:
        hit = store.get(key)
        if hit is not None:
            return OcrResult(text=hit["text"], backend=hit["backend"], model=hit["model"], revision=hit["revision"],
                             seconds=hit["seconds"], tokens=hit.get("tokens"), extra=hit.get("extra"),
                             part_stats=hit.get("part_stats")), True
    res = compute()
    if use_cache:
        store.put(key, fields, res)
    return res, False


@dataclass
class ResplitInfo:
    first_pass_tokens: Optional[int]  # โทเคนรวมของรอบแรก (ทั้งหน้า)
    first_pass_hits: int  # จำนวนชิ้นของรอบแรกที่ชนเพดาน
    pieces: int  # จำนวนชิ้นในผลสุดท้าย
    no_gap: int  # ครั้งที่ต้องการแบ่งแต่แบ่งไม่ได้ (ไม่มีช่องว่างแนวนอน)
    rotated: int  # จำนวนครึ่งหน้าที่หมุนภาพให้ตั้งตรงก่อน OCR
    regions: list  # region ของชิ้นสุดท้ายตามลำดับ
    all_cached: bool


def recognize_page_resplit(
    backend: OcrBackend, path: Path, detected: DetectedType, content_hash: str, page_no: int, page: PageImages,
    *, use_cache: bool = True, refresh_cache: bool = False, cache: Optional[OcrCache] = None,
    on_piece: Optional[Callable[[Piece, OcrResult, bool], None]] = None, first_pass: Optional[OcrResult] = None,
) -> tuple[OcrResult, ResplitInfo]:
    """OCR ซ้ำแบบแบ่งบน/ล่าง (ดู docstring ของ module) ทีละชิ้น (ไม่ขนาน) ผ่านแคชรายชิ้น
    ``on_piece(piece, result, from_cache)`` ถูกเรียกหลังแต่ละชิ้น (ไว้ log ความคืบหน้า)
    ``first_pass`` = ผลรอบแรก: ครึ่งหน้าที่แบ่งไม่ได้ใช้ผลรายชิ้นของรอบแรกซ้ำ (ภาพเดียวกัน) แทน OCR ใหม่ — ทำได้เมื่อ
    ผลรอบแรกมี ``part_stats`` ครบตามจำนวนครึ่งเท่านั้น (แคชรุ่นเก่าไม่มี -> OCR ครึ่งนั้นใหม่ด้วย key ของชิ้น)"""
    store = cache or OcrCache()
    limit = max_tokens_of(backend)
    method = f"{page.split_method}+rows"
    info = ResplitInfo(None, 0, 0, 0, 0, [], True)
    halves = page_pieces(page)
    reuse: list[Optional[OcrResult]] = [None] * len(halves)
    if first_pass is not None and first_pass.part_stats and len(first_pass.part_stats) == len(halves):
        parts = result_parts(first_pass)
        if len(parts) == len(halves):
            reuse = [OcrResult(text=t, backend=first_pass.backend, model=first_pass.model,
                               revision=first_pass.revision, seconds=0.0, tokens=tok) for t, tok in parts]

    def ocr(piece: Piece) -> OcrResult:
        fields = _key_fields(backend, content_hash, page_no, page, method, region=piece.region,
                             rotation=piece.rotation)
        res, hit = _through_cache(fields, lambda: backend.recognize(piece.png),
                                  use_cache=use_cache, refresh_cache=refresh_cache, store=store)
        info.all_cached = info.all_cached and hit
        if on_piece is not None:
            on_piece(piece, res, hit)
        return res

    def upright(piece: Piece) -> Piece:
        """ครึ่งหน้าที่ข้อความหมุน >= ``ROTATED_TEXT_SHARE`` (PDF) -> หมุนภาพให้ตั้งตรงตามทิศที่วัดได้"""
        if detected != DetectedType.PDF or piece.rotation:
            return piece
        o = text_orientation(path, page_no, piece.region)
        if o is None or o[0] < ROTATED_TEXT_SHARE or not o[1]:
            return piece
        info.rotated += 1
        return rotate_piece(piece, o[1])

    def split(piece: Piece) -> Optional[tuple[Piece, Piece]]:
        sub = split_piece_rows(piece)
        if sub is None:
            info.no_gap += 1
        return sub

    def expand(piece: Piece, half: int = 0) -> list[tuple[Piece, OcrResult]]:
        if piece.depth == 0:  # ครึ่งหน้า/ทั้งหน้า: หมุนให้ตั้งตรง (ถ้าต้อง) แล้วแบ่งบน/ล่างก่อนเสมอ
            up = upright(piece)
            sub = split(up)
            if sub is None:  # ผลรอบแรกใช้ซ้ำได้เฉพาะภาพเดียวกัน (ไม่ได้หมุน)
                same = up is piece and reuse[half] is not None
                return [(up, reuse[half] if same else ocr(up))]
            return expand(sub[0]) + expand(sub[1])
        res = ocr(piece)
        if limit and res.tokens is not None and res.tokens >= limit and piece.depth < RESPLIT_MAX_DEPTH:
            sub = split(piece)
            if sub is not None:
                return [(p, ocr(p)) for p in sub]
        return [(piece, res)]

    done = [x for i, half in enumerate(halves) for x in expand(half, i)]
    info.pieces = len(done)
    info.regions = [list(p.region) for p, _ in done]
    return _combine([r for _, r in done]), info


def recognize_page_auto(
    backend: OcrBackend, path: Path, detected: DetectedType, content_hash: str, page_no: int,
    *, use_cache: bool = True, refresh_cache: bool = False, cache: Optional[OcrCache] = None, resplit: bool = True,
    on_piece: Optional[Callable[[Piece, OcrResult, bool], None]] = None,
) -> tuple[OcrResult, PageImages, bool, Optional[ResplitInfo]]:
    """รอบแรกตามปกติ (``recognize_page_cached``) ถ้ามีชิ้นชนเพดานโทเคนและ ``resplit`` -> OCR ซ้ำแบบแบ่งบน/ล่าง
    คืน (ผลที่ใช้, ภาพของหน้า, ได้จากแคชทั้งหมดหรือไม่, ResplitInfo หรือ None ถ้าไม่ได้ resplit)"""
    store = cache or OcrCache()
    res, page, hit = recognize_page_cached(backend, path, detected, content_hash, page_no,
                                           use_cache=use_cache, refresh_cache=refresh_cache, cache=store)
    first_hits = hits_ceiling(res, max_tokens_of(backend))
    if not resplit or not first_hits:
        return res, page, hit, None
    res2, info = recognize_page_resplit(backend, path, detected, content_hash, page_no, page, use_cache=use_cache,
                                        refresh_cache=refresh_cache, cache=store, on_piece=on_piece,
                                        first_pass=res)
    info.first_pass_tokens = res.tokens
    info.first_pass_hits = first_hits
    return res2, page, hit and info.all_cached, info


def run_ocr(
    document: Document,
    source_path: str | Path,
    backend: OcrBackend,
    allow_offsite: Optional[bool] = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    resplit: bool = True,
    on_unit: Optional[Callable[[Unit], None]] = None,
) -> Document:
    """OCR เฉพาะ unit ที่ needs_ocr; คืน Document ใหม่ (ของเดิมไม่ถูกแก้)
    ``resplit`` = OCR ซ้ำแบบแบ่งบน/ล่างอัตโนมัติเมื่อรอบแรกชนเพดานโทเคน (ค่าเริ่มต้น เปิด)
    ``on_unit(unit_ใหม่)`` ถูกเรียกหลัง OCR แต่ละ unit (ไว้รายงานความคืบหน้า)
    หน้าที่ error -> unit นั้น failed หน้าอื่นทำต่อ **ยกเว้น** CUDA error ที่ context ใช้ต่อไม่ได้ (``is_fatal_cuda``):
    โยนต่อทันที (หน้าที่เหลือจะพังด้วยเหตุเดียวกัน และผู้เรียกต้องรู้ว่า GPU ใช้ต่อไม่ได้ — ไม่กลืนเป็น failed เงียบๆ)
    ``OcrUnavailable`` (backend ใช้ไม่ได้ เช่น ไม่มี GPU / โหลดโมเดลไม่ได้ / ไม่มีโปรแกรม) -> unit **คง needs_ocr** พร้อม reason
    ``OCR ใช้ไม่ได้: <เหตุผล>`` (ไม่ใช่ failed: หน้ายังไม่ได้ OCR ไม่ใช่ OCR แล้วพัง) หน้าอื่นทำต่อ (หน้าที่อยู่ในแคชยังได้ผลเดิม)
    ผลของ unit ที่ OCR แล้วจึงมีแค่ ok / failed (ไม่ผ่านด่านตรวจ) — needs_ocr หลัง run_ocr = OCR ใช้ไม่ได้เท่านั้น"""
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
            res, page, hit, info = recognize_page_auto(
                backend, path, document.detected_type, document.content_hash, int(unit.position["page"]),
                use_cache=use_cache, refresh_cache=refresh_cache, resplit=resplit,
            )
            out.append(_ocr_unit(unit, res, (time.perf_counter() - t0) * 1000, page, hit,
                                 max_tokens=max_tokens_of(backend), resplit=info))
        except OcrUnavailable as e:  # ยังไม่ได้ OCR (ไม่ใช่ OCR แล้วพัง) -> คง needs_ocr
            if is_fatal_cuda(e) or (e.__cause__ is not None and is_fatal_cuda(e.__cause__)):
                raise
            out.append(ocr_unavailable_unit(unit, e))
        except Exception as e:  # รายหน้า: ไม่ให้หน้าเดียวล้มทั้งไฟล์
            if is_fatal_cuda(e):
                raise
            out.append(_failed(unit, e, (time.perf_counter() - t0) * 1000))
        if on_unit is not None:
            on_unit(out[-1])
    return document.model_copy(update={"units": out})
