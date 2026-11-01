"""เตรียมรายการให้คนตัดสินข้อความที่ถูกของ 7 หน้า (T06b ช่วงที่ 1)

    python scripts/ocr_truth_prepare.py            # อ่านผล OCR จากแคช (ต้องรัน scripts/ocr_pages.py ทั้งสอง backend ก่อน)
    python scripts/ocr_truth_prepare.py --no-html  # สร้างแค่ items.json

ผลลัพธ์
- eval/datasets/ocr_truth/<file>_p<page>.items.json   (ข้อความจาก PDF สาธารณะ เข้า git ได้)
- eval/datasets/files/ocr_truth_review.html           (ไฟล์เดียว มีภาพ ไม่เข้า git) เปิดด้วยเบราว์เซอร์

ขั้นตอนต่อหน้า
1. normalize (normalizer กลาง) ทั้งสองฝั่ง แตกเป็นบรรทัด (ตาราง = แถวละบรรทัด) ``<figure>`` ของ Typhoon เก็บแยก (``figures``)
   ไม่เอามาเป็นข้อความที่ถูก; เลขหน้าที่ Typhoon ตัดออก (``<page_number>``) ไม่ถือเป็น "มีแต่ใน Tesseract"
2. canonical = ตัดเครื่องหมาย markdown (# * | และ "- " หัวบรรทัด), ช่องว่างทุกชนิด, ปรับเครื่องหมายคำพูดโค้งเป็นตรง
3. หน่วยที่ให้คนตัดสิน = บรรทัดที่มองเห็นในภาพ (บรรทัดของ Tesseract) เพราะ Typhoon เขียนย่อหน้ายาวเป็นบรรทัดเดียวและลำดับอ่านข้ามคอลัมน์ได้
   แต่ละบรรทัด Tesseract ถูกนำไปหาตำแหน่งที่คล้ายที่สุดในข้อความ Typhoon ทั้งหน้า (canonical ต่อกัน, rapidfuzz partial_ratio_alignment
   แล้วขยับขอบ ± ``REFINE_SHIFT`` ตัวให้ความคล้ายแบบ Levenshtein สูงสุด) — **ไม่ตามลำดับ ไม่ต้องติดกัน** จึงรับมือหน้า 2 คอลัมน์ได้
   ข้อความ Typhoon ช่วงนั้น (ตัดมาจากบรรทัดของ Typhoon ตามที่ Typhoon เขียน) กลายเป็นอีกฝั่งของคู่
   3.1 ช่วงที่ซ้อนกัน: บรรทัดที่ (ความคล้าย x จำนวนตัวอักษร) สูงสุดครอบก่อน (บรรทัดยาวมีบริบท บรรทัดสั้นที่ตรงเป๊ะแต่คลุมเครือหาที่ว่างทีหลัง); ช่วงของบรรทัดอื่นที่ซ้อน ถูกหาตำแหน่งใหม่ในข้อความที่กันตัวที่ถูกครอบออก (วลีซ้ำสองที่)
       ถ้าไม่ได้ก็ตัดเหลือส่วนว่างที่ยาวสุด (< ``TRIM_MIN_KEEP`` ของเดิม = ไม่จับคู่ เป็น tesseract_only)
   3.2 snap ขอบ: ขยายต้น/ท้ายช่วงไปที่ขอบบรรทัดของ Typhoon หรือขอบคำ (ช่องว่างในข้อความแสดงผล) ที่ใกล้ที่สุด ไม่เกิน ``SNAP_MAX`` ตัวต่อข้าง
       ไม่ทับรายการอื่น (ชนรายการอื่นหยุดที่ขอบรายการนั้น) ไม่เจอขอบ = ไม่ตัดกลางพยางค์ไทย (สระบน/ล่าง วรรณยุกต์ ำ ติดพยัญชนะ) + ``cut_mid``
   3.3 ตัวที่ไม่มีใครครอบ: ยาว >= ``MIN_GAP_CANON`` = typhoon_only; สั้นกว่านั้นรวมเข้ารายการข้างเคียง (บรรทัด Typhoon เดียวกัน ถ้าเท่ากัน
       ตัวก่อนหน้า) ไม่มีข้างเคียง = typhoon_only (``gap_merged`` = จำนวนตัวที่รวม; ``snap`` = จำนวนตัวที่ขยายตอน snap)
   3.4 kind คำนวณหลังขยายช่วง (agree เฉพาะเมื่อ canonical ของช่วง Typhoon เท่ากับ Tesseract ทุกตัว); invariant: ทุกตัวของข้อความ Typhoon อยู่ใน
       agree/diff/typhoon_only พอดี 1 รายการ (assert ตอน build; ``invariant.dropped`` ต่อหน้า); ``pos``/``pos_end`` = ช่วงในข้อความ Typhoon
   3.5 Tesseract แยก region (แถบข้างสีพื้น / เนื้อหา) ได้ในตัว backend (ดู ``app/extraction/ocr_tesseract.py``): แต่ละรายการมี
       ``tesseract.column`` (0 แถบ / 1 เนื้อหา / None) และหน้ามี ``column_split`` (x ต่อภาพ) + ภาพ debug ที่ ``DEBUG_DIR``
4. แบ่งเป็น 4 แบบ
   - agree: canonical ตรงกันทุกตัวอักษร -> รับอัตโนมัติ แต่สุ่ม ``AUDIT_FRACTION`` (seed ตายตัว) ให้คนตรวจซ้ำ (``audit``)
   - diff: คล้าย (ความคล้าย >= ``MATCH_SCORE``) แต่ไม่ตรง
   - tesseract_only: บรรทัด Tesseract ที่หาช่วงคล้ายใน Typhoon ไม่ได้ (สงสัย Typhoon อ่านตก)
   - typhoon_only: ข้อความ Typhoon ตั้งแต่ ``MIN_GAP_CANON`` ตัวที่ไม่มีบรรทัด Tesseract ใดคลุม (สงสัยโมเดลแต่ง; ภาพที่ให้ดู = แถบ
     ระหว่างบรรทัดข้างเคียงที่จับคู่ได้ ถ้าไม่ได้/สูงเกิน ``MAX_REGION_HEIGHT`` = ภาพทั้งหน้า)
   บรรทัด Tesseract ที่เป็นขยะ (canonical สั้นกว่า ``MIN_TESS_CANON`` หรือ conf ต่ำกว่า ``MIN_TESS_CONF``) ถูกข้ามและบันทึกไว้
   ใน ``skipped_tesseract`` ให้ตรวจย้อนหลังได้
   ``pos`` ของแต่ละรายการ = ตำแหน่งในข้อความ Typhoon (ไว้เรียงลำดับ / ประกอบข้อความที่ถูกในช่วงที่ 2)
5. กรองสิ่งที่ไม่ใช่ "บรรทัดที่ Typhoon อ่านตก" ออกจาก tesseract_only (ไม่ส่งให้คนตัดสิน ไม่นับใน ``to_judge``; ยังอยู่ในไฟล์
   และเปิดดูในหน้าตรวจได้ไว้สุ่มตรวจว่ากรองถูก)
   - fragment: bbox สูงน้อยกว่า ``FRAGMENT_MAX_HEIGHT`` px **และ** ข้อความดูเป็นขยะ (conf < ``FRAGMENT_MAX_CONF`` หรือมีตัวอักษร/ตัวเลข
     น้อยกว่า ``FRAGMENT_MIN_TEXT_CHARS`` ตัวหลังตัดช่องว่างและเครื่องหมาย) = Tesseract จับได้แค่หัวสระ/วรรณยุกต์/ลายน้ำ (เศษของบรรทัดข้างเคียง)
     ตัวหนังสือเล็กจริงในอินโฟกราฟิก (สูง < 20 px แต่ conf สูงและยาวพอ เช่น "Hydrogen & Carbon Business") ไม่เข้าเกณฑ์ ยังให้คนตัดสิน;
     ``covered_by`` = รายการข้างเคียงที่มันเป็นเศษของ (ดู ``FRAGMENT_NEIGHBOUR_PAD`` / ``FRAGMENT_MAX_NEIGHBOURS``)
   - duplicate: ทับ bbox ของรายการ agree/diff ในภาพเดียวกัน **ทั้งสองทาง** คือ >= ``DUPLICATE_OVERLAP_SELF`` ของพื้นที่ตัวเอง
     และ >= ``DUPLICATE_OVERLAP_OTHER`` ของพื้นที่รายการที่ทับ = Tesseract อ่านบรรทัดเดียวกันสองรอบ; ``covered_by`` = id ของรายการที่ทับ
     (ต้องทับสองทาง เพราะ Tesseract บางครั้งรวมแถบข้างกับตารางเป็นบรรทัดกว้างเกือบทั้งภาพ บรรทัดจริงที่อยู่ในนั้นทับ "ฝั่งตัวเอง" เกิน 50%
     แต่เป็นสัดส่วนเล็กของบรรทัดกว้าง ถ้านับเป็น duplicate บรรทัดจริงจะหายเงียบ — ตัวอย่าง ONEREPORTKBANKT หน้า 87)
   - ทับทางเดียว (ถึงเกณฑ์ฝั่งตัวเอง แต่ไม่ถึงฝั่งอีกรายการ) = ยังเป็น tesseract_only ให้คนตัดสินตามปกติ แต่ใส่ ``maybe_duplicate_of``
     (id ที่ทับ) หน้าตรวจแสดงป้าย "อาจซ้ำกับ <id>" กับข้อความของรายการนั้น คนกด 3 (ไม่มีในภาพ) ได้เร็วถ้าเป็นบรรทัดเดียวกัน

ช่วงที่ 2 — ประกอบข้อความที่ถูก (ยังไม่ได้ทำ; วางกติกาไว้ตรงนี้ก่อน)
- ประกอบจากรายการที่มีคำตัดสินหรือรับอัตโนมัติ (agree ที่รับอัตโนมัติ/ที่สุ่มตรวจแล้ว, diff, tesseract_only, typhoon_only) เรียงตาม ``pos``
  (``order``); คำตัดสิน ``absent`` = ไม่ประกอบบรรทัดนั้น
- fragment / duplicate **ไม่ถูกประกอบ** (ไม่มีคำตัดสินอยู่แล้ว) และไม่นับเป็นบรรทัดที่ Typhoon อ่านตก/แต่งขึ้น
- บรรทัดเดียวกันที่ยังเป็น 2 รายการ (เช่น tesseract_only ที่คนพิมพ์เอง + typhoon_only ที่คนกด "Typhoon ถูก") ต้องไม่ซ้ำในข้อความที่ถูก
  หลังได้ข้อความที่ถูกของแต่ละรายการแล้ว ให้เทียบ canonical ของรายการที่อยู่ติดกันตามลำดับ ``pos`` (ห่างกันไม่เกิน 3 รายการ):
  1) เท่ากัน หรือฝั่งหนึ่งเป็นข้อความย่อยของอีกฝั่ง -> เก็บเฉพาะรายการที่ยาวกว่า
  2) คล้ายกันตั้งแต่ ``MATCH_SCORE`` (ทับกันบางส่วน) -> เก็บรายการที่คนแก้/พิมพ์เอง (verdict ``edited``) ก่อน ไม่มีก็เก็บรายการที่มี bbox
  3) ต่างกันชัดเจน (เช่น หัวตารางซ้ำ 2 ที่จริง) -> เก็บทั้งคู่ — เงื่อนไข "อยู่ติดกัน" กันการตัดทิ้งผิดข้อความที่ซ้ำกันจริง
  รายการที่ถูกตัดทิ้งเพราะซ้ำต้องนับและรายงานแยก ไม่เงียบ
"""

import argparse
import base64
import hashlib
import html as html_lib
import io
import json
import random
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.extraction.ocr import OcrResult  # noqa: E402
from app.extraction.ocr_markdown import parse_ocr_markdown  # noqa: E402
from app.models import BlockType  # noqa: E402
from app.thai.normalizer import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "eval" / "datasets"
TRUTH_DIR = DATASETS / "ocr_truth"
REVIEW_HTML = DATASETS / "files" / "ocr_truth_review.html"
DEBUG_DIR = DATASETS / "files" / "_debug"
TEMPLATE = Path(__file__).resolve().parent / "ocr_review_template.html"

MATCH_SCORE = 70.0  # ความคล้าย (0–100; partial_ratio ตอนหาตำแหน่ง, Levenshtein หลังขยับขอบ) ขั้นต่ำที่ถือว่า "คล้าย"
MIN_FUZZY_LEN = 4  # บรรทัด Typhoon ที่ canonical สั้นกว่านี้ ไม่ใช้ fuzzy (ต้องตรงทั้งบรรทัด Tesseract เท่านั้น)
REFINE_SHIFT = 4  # ขยับขอบช่วงข้อความ Typhoon ได้กี่ตัวอักษรเพื่อให้ตรงกับบรรทัด Tesseract
SNAP_MAX = 10  # ตัว canonical สูงสุดที่ขยายขอบช่วง Typhoon ต่อข้างเพื่อให้ลงที่ขอบคำ/บรรทัด
TRIM_MIN_KEEP = 0.5  # ช่วงที่ซ้อนกับรายการที่ครอบไปแล้ว ถ้าเหลือน้อยกว่าสัดส่วนนี้ของช่วงเดิม = ไม่จับคู่
LEADING_VOWELS = "เแโใไ"
MIN_GAP_CANON = 6  # ข้อความ Typhoon ที่ไม่มีบรรทัด Tesseract คลุม ยาวเท่านี้ขึ้นไปถึงเป็นรายการ typhoon_only
MAX_REGION_HEIGHT = 700  # px สูงสุดของแถบภาพระหว่างบรรทัดข้างเคียง (เกินนี้ = ลำดับอ่านข้ามคอลัมน์ ใช้ภาพทั้งหน้า)
MIN_TESS_CANON = 2
MIN_TESS_CONF = 25.0
FRAGMENT_MAX_HEIGHT = 20  # px: tesseract_only ที่ bbox เตี้ยกว่านี้ **และ** ดูเป็นขยะ = เศษบรรทัด (fragment)
FRAGMENT_MAX_CONF = 70.0  # ขยะ = conf ต่ำกว่านี้ (ไม่มี conf ก็นับเป็นขยะ)
FRAGMENT_MIN_TEXT_CHARS = 4  # หรือมีตัวอักษร/ตัวเลข (รวมวรรณยุกต์/สระประกอบ) น้อยกว่านี้ — ไม่นับช่องว่างและเครื่องหมาย
FRAGMENT_NEIGHBOUR_PAD = 12  # px: เศษบรรทัดนับรายการข้างเคียงที่อยู่ห่างในแนวตั้งไม่เกินเท่านี้ว่าเป็น "บรรทัดที่มันเป็นเศษของ"
FRAGMENT_MAX_NEIGHBOURS = 2
DUPLICATE_OVERLAP_SELF = 0.5  # สัดส่วนพื้นที่ "ตัวเอง" ที่ถูกรายการ agree/diff ทับ
DUPLICATE_OVERLAP_OTHER = 0.5  # สัดส่วนพื้นที่ "รายการที่ทับ" ที่ถูกตัวเองทับ — ต้องถึงทั้งสองทางถึงเป็น duplicate
TESS_ONLY_KINDS = ("tesseract_only", "fragment", "duplicate")  # แบบที่มาจากบรรทัด Tesseract ที่ไม่มีคู่ใน Typhoon
AUDIT_FRACTION = 0.10
AUDIT_SEED = 20260101
CROP_PAD = 14  # px รอบ bbox
PAGE_THUMB_LONG_SIDE = 1500

_MD_CHARS = re.compile(r"[#*|]")
_LEADING_DASH = re.compile(r"(?m)^\s*-\s+")
_WS = re.compile(r"\s+")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def canon(text: str) -> str:
    """รูปที่ใช้เทียบ: normalize -> ตัดเครื่องหมาย markdown -> ตัดช่องว่าง"""
    t = normalize(text).translate(_QUOTES)
    t = _LEADING_DASH.sub("", t)
    return _WS.sub("", _MD_CHARS.sub("", t))


# ---------- แตกบรรทัด ----------

def typhoon_lines(raw: str) -> tuple[list[str], list[str], list[int]]:
    """(บรรทัดข้อความ, ข้อความ figure, เลขหน้า) จากผลดิบของ Typhoon; ข้อความผ่าน normalizer แล้วแต่ยังไม่ตัด markdown ออก"""
    parsed = parse_ocr_markdown(raw)
    lines: list[str] = []
    figures: list[str] = []
    for b in parsed.blocks:
        if b.type == BlockType.IMAGE:
            figures.append(normalize(b.text))
        elif b.type == BlockType.TABLE:
            for row in b.table.rows:
                t = normalize(" | ".join(c for c in row if c))
                if t.strip():
                    lines.append(t)
        else:
            for ln in b.text.splitlines():
                t = normalize(ln)
                t = re.sub(r"^\s*[*\-]\s+", "", t)  # จุดนำหน้ารายการ (เลขลำดับ "1." คงไว้ เพราะมีในภาพ)
                if t.strip():
                    lines.append(t.strip())
    return [ln for ln in lines if canon(ln)], figures, parsed.page_numbers


def tesseract_parts(res: OcrResult) -> list[dict]:
    extra = res.extra or {}
    return extra["parts"] if "parts" in extra else [extra]


# ---------- จับคู่ ----------

@dataclass
class TLine:
    text: str
    canon: str
    image: int
    bbox: list[int]
    conf: Optional[float]
    start: int = 0
    end: int = 0
    covered: int = 0
    col: Optional[int] = None  # 0 = แถบข้าง / 1 = เนื้อหา เมื่อ Tesseract แยก region (ไม่แยก = None)


def _ratio(a: str, b: str) -> float:
    """ความคล้าย 0–100 แบบ Levenshtein (แทนที่ตัวอักษรได้ ไม่ใช่แค่ใส่/ลบ) — ใช้ตัดสินขอบช่วงข้อความ Typhoon:
    ratio แบบ indel จะชอบ "ตัดตัวอักษรที่ Tesseract อ่านผิดทิ้ง" มากกว่า "ยอมให้ผิดหนึ่งตัว" ทำให้ขอบช่วงเลื่อนเข้าไปกลางคำ"""
    from rapidfuzz.distance import Levenshtein

    return 100.0 * Levenshtein.normalized_similarity(a, b)


def _align(needle: str, haystack: str):
    from rapidfuzz import fuzz

    return fuzz.partial_ratio_alignment(needle, haystack)


def _span_display(segs: list[dict], s: int, e: int) -> str:
    """ข้อความแสดงผล (มีช่องว่างตามที่ Typhoon เขียน) ของช่วง canonical [s, e) ของข้อความ Typhoon ทั้งหน้า"""
    parts = []
    for sg in segs:
        lo, hi = max(s, sg["start"]), min(e, sg["end"])
        if lo >= hi:
            continue
        if sg["idx"] is None:
            parts.append(sg["display"])
        else:
            a, b = lo - sg["start"], hi - sg["start"]
            parts.append(sg["display"][sg["idx"][a]: sg["idx"][b - 1] + 1])
    return " ".join(parts).strip()


def _refine(c: str, hay: str, s: int, e: int) -> tuple[float, int, int]:
    """ขยับขอบช่วง [s, e) ทีละไม่เกิน ``REFINE_SHIFT`` ตัวอักษร ให้ fuzz.ratio กับ c สูงสุด (เท่ากัน = ใกล้ขอบเดิมกว่า)"""
    best = (_ratio(c, hay[s:e]), s, e, 0)
    for ds in range(-REFINE_SHIFT, REFINE_SHIFT + 1):
        for de in range(-REFINE_SHIFT, REFINE_SHIFT + 1):
            s2, e2 = s + ds, e + de
            if s2 < 0 or e2 > len(hay) or e2 <= s2:
                continue
            r = _ratio(c, hay[s2:e2])
            cost = abs(ds) + abs(de)
            if r > best[0] + 1e-9 or (abs(r - best[0]) <= 1e-9 and cost < best[3]):
                best = (r, s2, e2, cost)
    return best[0], best[1], best[2]


def _area(b: list[int]) -> int:
    return max(b[2] - b[0], 0) * max(b[3] - b[1], 0)


def _intersection(a: list[int], b: list[int]) -> int:
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def _same_image(a: dict, b: dict) -> bool:
    return bool(a.get("tesseract") and b.get("tesseract") and a["tesseract"]["image"] == b["tesseract"]["image"])


def _text_chars(text: str) -> int:
    """จำนวนตัวอักษร/ตัวเลข/เครื่องหมายประกอบตัวอักษร (ไม่นับช่องว่างและเครื่องหมายวรรคตอน)"""
    import unicodedata

    return sum(1 for ch in text if unicodedata.category(ch)[0] in "LNM")


def looks_like_junk(it: dict) -> bool:
    t = it["tesseract"]
    return t["conf"] is None or t["conf"] < FRAGMENT_MAX_CONF or _text_chars(t["text"]) < FRAGMENT_MIN_TEXT_CHARS


def classify_noise(items: list[dict]) -> None:
    """แยก fragment / duplicate ออกจาก tesseract_only (แก้ ``kind`` ในที่เดิม) และใส่ ``covered_by`` / ``maybe_duplicate_of``
    (เป็น item ก่อนแปลงเป็น id) ให้ทุกรายการที่ส่งเข้ามา — เกณฑ์ดูที่ docstring ของโมดูลข้อ 5"""
    for it in items:
        it["covered_by"], it["maybe_duplicate_of"] = [], []
    tess_only = [it for it in items if it["kind"] == "tesseract_only"]
    fragments = [it for it in tess_only
                 if it["tesseract"]["bbox"][3] - it["tesseract"]["bbox"][1] < FRAGMENT_MAX_HEIGHT and looks_like_junk(it)]
    for it in fragments:
        it["kind"] = "fragment"
    matched = [it for it in items if it["kind"] in ("agree", "diff")]
    neighbours = matched + [it for it in tess_only if it["kind"] == "tesseract_only"]
    for it in fragments:
        b = it["tesseract"]["bbox"]
        band = [b[0], b[1] - FRAGMENT_NEIGHBOUR_PAD, b[2], b[3] + FRAGMENT_NEIGHBOUR_PAD]
        near = []
        for o in neighbours:
            if not _same_image(it, o):
                continue
            ob = o["tesseract"]["bbox"]
            vertical = min(band[3], ob[3]) - max(band[1], ob[1])
            horizontal = min(b[2], ob[2]) - max(b[0], ob[0])
            if vertical > 0 and horizontal > 0.5 * max(b[2] - b[0], 1):
                near.append((vertical, o))
        it["covered_by"] = [o for _, o in sorted(near, key=lambda t: -t[0])[:FRAGMENT_MAX_NEIGHBOURS]]
    for it in tess_only:
        if it["kind"] != "tesseract_only":
            continue
        b = it["tesseract"]["bbox"]
        own = _area(b)
        for o in matched:
            if own == 0 or not _same_image(it, o):
                continue
            inter = _intersection(b, o["tesseract"]["bbox"])
            if inter / own >= DUPLICATE_OVERLAP_SELF:
                both = inter / max(_area(o["tesseract"]["bbox"]), 1) >= DUPLICATE_OVERLAP_OTHER
                (it["covered_by"] if both else it["maybe_duplicate_of"]).append(o)
        if it["covered_by"]:
            it["kind"] = "duplicate"
            it["maybe_duplicate_of"] = []


def _is_dependent(ch: str) -> bool:
    """สระบน/ล่าง วรรณยุกต์ และ ำ — ต้องติดพยัญชนะ ห้ามขึ้นต้นช่วง"""
    return unicodedata.category(ch) == "Mn" or ch == "ำ"


def _boundaries(segs: list[dict], n: int) -> set[int]:
    """ตำแหน่ง canonical ที่เป็นขอบคำ/บรรทัดของข้อความ Typhoon: ต้น-ท้ายข้อความ ขอบบรรทัด Typhoon และจุดที่ในข้อความแสดงผล
    (``display``) มีช่องว่างคั่นระหว่างตัวอักษรสองตัว (ใช้ ``idx`` แปลงกลับ)"""
    b = {0, n}
    for sg in segs:
        b.add(sg["start"])
        b.add(sg["end"])
        idx = sg["idx"]
        if idx:
            b.update(sg["start"] + a for a in range(1, len(idx)) if idx[a] - idx[a - 1] > 1)
    return b


def _snap_left(s: int, owner: list[int], hay: str, bounds: set[int]) -> tuple[int, bool]:
    """ขยายต้นช่วง s ไปทางซ้ายถึงขอบที่ใกล้ที่สุดภายใน ``SNAP_MAX`` ตัว (เฉพาะตัวที่ยังไม่มีรายการใดครอบ; ชนรายการอื่นหยุดที่ขอบรายการนั้น)
    ไม่เจอขอบเลย: อย่างน้อยไม่ตัดกลางพยางค์ (ดึงพยัญชนะที่สระ/วรรณยุกต์เกาะอยู่เข้ามา) และคืน True = ตัดกลางคำ (cut_mid)"""
    if s in bounds or (s > 0 and owner[s - 1] != -1):
        return s, False
    q, lo = s, max(s - SNAP_MAX, 0)
    while q > lo and owner[q - 1] == -1:
        q -= 1
        if q in bounds:
            return q, False
    if q > 0 and owner[q - 1] != -1:  # ชนรายการอื่น (ก่อนครบ SNAP_MAX หรือพอดี)
        return q, False
    s2 = s
    while s2 > 0 and owner[s2 - 1] == -1 and _is_dependent(hay[s2]):
        s2 -= 1
    return s2, s2 not in bounds


def _snap_right(e: int, owner: list[int], hay: str, bounds: set[int]) -> tuple[int, bool]:
    """ท้ายช่วง e (ไม่รวม) ไปทางขวา — กติกาเดียวกับ ``_snap_left``; ไม่เจอขอบ: ดึงสระ/วรรณยุกต์ที่ตามมา และสระนำ (เ แ โ ใ ไ) ที่ค้างท้าย"""
    n = len(hay)
    if e in bounds or (e < n and owner[e] != -1):
        return e, False
    q, hi = e, min(e + SNAP_MAX, n)
    while q < hi and owner[q] == -1:
        q += 1
        if q in bounds:
            return q, False
    if q < n and owner[q] != -1:
        return q, False
    e2 = e
    while e2 < n and owner[e2] == -1 and _is_dependent(hay[e2]):
        e2 += 1
    if e2 > 0 and e2 < n and owner[e2] == -1 and hay[e2 - 1] in LEADING_VOWELS:
        e2 += 1
    return e2, e2 not in bounds


def _best_span(c: str, hay: str) -> Optional[tuple[float, int, int]]:
    """ช่วงของ hay ที่คล้ายบรรทัด c ที่สุด: (ความคล้าย, s, e) หรือ None ถ้าไม่ถึง ``MATCH_SCORE``"""
    if not hay:
        return None
    if len(c) < MIN_FUZZY_LEN:
        p = hay.find(c)  # สั้นมาก: ต้องเจอตรงตัว
        return (100.0, p, p + len(c)) if p >= 0 else None
    if len(c) > len(hay):
        return None
    al = _align(c, hay)
    if al.score < MATCH_SCORE:
        return None
    r, s, e = _refine(c, hay, al.dest_start, al.dest_end)
    return (r, s, e) if r >= MATCH_SCORE else None


def _longest_free_run(owner: list[int], s: int, e: int) -> tuple[int, int]:
    best, i = (s, s), s
    while i < e:
        if owner[i] != -1:
            i += 1
            continue
        j = i
        while j < e and owner[j] == -1:
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j
    return best


def build_items(file: str, page: int, typhoon: OcrResult, tesseract: OcrResult) -> dict:
    """หน่วยที่ให้คนตัดสิน = บรรทัดที่มองเห็นในภาพ (บรรทัดของ Tesseract) คู่กับช่วงข้อความ Typhoon ที่ตรงกัน

    เหตุผล: Typhoon เขียนย่อหน้ายาวเป็นบรรทัดเดียว (หลายร้อยตัวอักษร) และลำดับอ่านข้ามคอลัมน์ได้ ส่วน Tesseract ให้บรรทัดสั้นพร้อม bbox
    จึงเอาแต่ละบรรทัด Tesseract ไปหาตำแหน่งในข้อความ Typhoon ทั้งหน้า (ไม่ตามลำดับ ไม่ต้องติดกัน) แล้วตัดข้อความ Typhoon ช่วงนั้นมาเทียบ

    ขอบช่วง (``hay[s:e]``) ผ่าน 4 ขั้น (docstring โมดูลข้อ 3):
    1) จับคู่เบื้องต้น (หาตำแหน่ง + ``_refine``) 2) แก้ช่วงที่ซ้อนกัน: บรรทัดที่ได้ความคล้าย x ความยาวมากสุดครอบก่อน ที่เหลือหาตำแหน่งใหม่/ถูกตัดเหลือส่วนที่ว่างยาวสุด
       (เหลือไม่ถึง ``TRIM_MIN_KEEP`` ของช่วงเดิม = ไม่จับคู่ ให้เป็น tesseract_only) 3) snap ขอบไปที่ขอบคำ/บรรทัดของ Typhoon (ไม่เกิน ``SNAP_MAX``
       ตัวต่อข้าง ไม่ทับรายการอื่น) 4) ตัวที่ไม่มีใครครอบ: ยาว >= ``MIN_GAP_CANON`` = typhoon_only; สั้นกว่านั้นรวมเข้ารายการข้างเคียง (บรรทัด
       Typhoon เดียวกัน ถ้าเท่ากันตัวก่อนหน้า) ไม่มีข้างเคียงเลย = typhoon_only; kind คำนวณหลังสุด (agree เฉพาะเมื่อ canonical ตรงกันทุกตัว)
    invariant: ทุกตัวของ ``hay`` ถูกครอบโดยรายการ agree/diff/typhoon_only พอดี 1 ครั้ง (assert; ``invariant.dropped`` = 0)
    """
    ty_lines, figures, page_numbers = typhoon_lines(typhoon.text)

    segs: list[dict] = []
    hay = ""
    for line in ty_lines:
        disp = _display(line)
        c = canon(disp)
        idx = [i for i, ch in enumerate(disp) if not ch.isspace()]
        segs.append({"display": disp, "canon": c, "idx": idx if len(idx) == len(c) else None, "start": len(hay), "end": len(hay) + len(c)})
        hay += c
    n_hay = len(hay)
    bounds = _boundaries(segs, n_hay)

    kept: list[TLine] = []
    skipped: list[dict] = []
    page_no_canon = {str(n) for n in page_numbers}
    parts = tesseract_parts(tesseract)
    for img, part in enumerate(parts):
        for ln in part.get("lines", []):
            c = canon(ln["text"])
            reason = None
            if len(c) < MIN_TESS_CANON:
                reason = "สั้นเกินไป"
            elif ln.get("conf") is not None and ln["conf"] < MIN_TESS_CONF:
                reason = f"conf ต่ำ ({ln['conf']})"
            elif c.isdigit() and c.lstrip("0") in page_no_canon:
                reason = "เลขหน้า (Typhoon ตัดออกโดยตั้งใจ)"
            if reason:
                skipped.append({"text": ln["text"], "reason": reason, "image": img, "bbox": ln["bbox"]})
                continue
            kept.append(TLine(ln["text"], c, img, ln["bbox"], ln.get("conf"), col=ln.get("col")))

    stats = {"relocated": 0, "trimmed": 0, "demoted_by_overlap": 0, "snapped_items": 0, "snapped_chars": 0,
             "gap_merged_runs": 0, "gap_merged_chars": 0, "short_gap_typhoon_only": 0}

    # 1) จับคู่เบื้องต้น  2) แก้ช่วงที่ซ้อนกัน
    spans = [_best_span(t.canon, hay) for t in kept]
    owner = [-1] * n_hay  # index ใน matches; -2 = typhoon_only
    matches: list[dict] = []
    order = sorted((k for k, sp in enumerate(spans) if sp), key=lambda k: (-spans[k][0] * len(kept[k].canon), k))  # ตัวอักษรที่จับคู่ได้มากสุดก่อน (เส้นยาวที่มีบริบทครอบก่อนบรรทัดสั้นที่คลุมเครือ)
    for k in order:
        r, s, e = spans[k]
        rs, re_ = _longest_free_run(owner, s, e)
        if (rs, re_) != (s, e):
            # ซ้อนกับรายการที่ครอบไปแล้ว (เช่น วลีเดียวกันปรากฏสองที่ในข้อความ Typhoon): หาตำแหน่งใหม่ในข้อความที่กันตัวที่ถูกครอบออก
            masked = "".join("\x00" if o != -1 else ch for ch, o in zip(hay, owner))
            alt = _best_span(kept[k].canon, masked)
            if alt:
                ars, are = _longest_free_run(owner, alt[1], alt[2])
                if are - ars > re_ - rs:
                    (r, s, e), (rs, re_) = alt, (ars, are)
                    stats["relocated"] += 1
        if re_ - rs < TRIM_MIN_KEEP * (e - s) or re_ <= rs:
            spans[k] = None
            stats["demoted_by_overlap"] += 1
            continue
        stats["trimmed"] += (rs, re_) != (s, e)
        for i in range(rs, re_):
            owner[i] = len(matches)
        matches.append({"i": len(matches), "k": k, "s": rs, "e": re_, "s0": rs, "e0": re_, "snap": 0, "gap": 0, "cut": []})

    # 3) snap ขอบ
    for m in sorted(matches, key=lambda m: m["s"]):
        mi = m["i"]
        s2, cut_s = _snap_left(m["s"], owner, hay, bounds)
        for i in range(s2, m["s"]):
            owner[i] = mi
        e2, cut_e = _snap_right(m["e"], owner, hay, bounds)
        for i in range(m["e"], e2):
            owner[i] = mi
        m["snap"] = (m["s"] - s2) + (e2 - m["e"])
        m["s"], m["e"] = s2, e2
        m["cut"] = [side for side, flag in (("start", cut_s), ("end", cut_e)) if flag]
        if m["snap"]:
            stats["snapped_items"] += 1
            stats["snapped_chars"] += m["snap"]

    # 4) ตัวที่ไม่มีใครครอบ
    ty_spans: list[tuple[int, int]] = []
    for sg in segs:
        i = sg["start"]
        while i < sg["end"]:
            if owner[i] != -1:
                i += 1
                continue
            j = i
            while j < sg["end"] and owner[j] == -1:
                j += 1
            prev = owner[i - 1] if i > 0 and owner[i - 1] >= 0 else None
            nxt = owner[j] if j < n_hay and owner[j] >= 0 else None
            if j - i >= MIN_GAP_CANON or (prev is None and nxt is None):
                stats["short_gap_typhoon_only"] += j - i < MIN_GAP_CANON
                ty_spans.append((i, j))
                for q in range(i, j):
                    owner[q] = -2
            else:
                same_prev, same_next = prev is not None and i > sg["start"], nxt is not None and j < sg["end"]
                pick = prev if (same_prev or not same_next) and prev is not None else nxt
                mm = matches[pick]
                if pick == prev:
                    mm["e"] = j
                else:
                    mm["s"] = i
                mm["gap"] += j - i
                for q in range(i, j):
                    owner[q] = pick
                stats["gap_merged_runs"] += 1
                stats["gap_merged_chars"] += j - i
            i = j

    # สร้างรายการ
    items: list[dict] = []
    for m in matches:
        t, s, e = kept[m["k"]], m["s"], m["e"]
        c = t.canon
        kind = "agree" if hay[s:e] == c else "diff"
        initial_agree = hay[m["s0"]:m["e0"]] == c
        items.append({"typhoon": {"text": _span_display(segs, s, e)},
                      "tesseract": {"text": t.text, "image": t.image, "bbox": t.bbox, "conf": t.conf, "column": t.col},
                      "region": None, "auto_accept": kind == "agree", "audit": False, "kind": kind,
                      "score": 100.0 if kind == "agree" else round(_ratio(c, hay[s:e]), 1), "pos": s, "end": e, "k": m["k"],
                      "snap": m["snap"], "gap_merged": m["gap"], "cut_mid": bool(m["cut"]), "cut_mid_sides": m["cut"],
                      "_before": {"agree": initial_agree, "typhoon": _span_display(segs, m["s0"], m["e0"])}})
    matched_ks = {m["k"] for m in matches}
    for k, t in enumerate(kept):
        if k not in matched_ks:
            items.append({"typhoon": None, "tesseract": {"text": t.text, "image": t.image, "bbox": t.bbox, "conf": t.conf, "column": t.col},
                          "region": None, "auto_accept": False, "audit": False, "kind": "tesseract_only", "score": 0.0, "pos": None, "k": k,
                          "snap": 0, "gap_merged": 0, "cut_mid": False, "cut_mid_sides": []})

    # typhoon_only: ข้อความ Typhoon ที่ไม่มีบรรทัด Tesseract ใดคลุม (ต่อบรรทัดของ Typhoon)
    matched = sorted((it for it in items if it["pos"] is not None), key=lambda it: it["pos"])
    for i, j in ty_spans:
        prev = next((m for m in reversed(matched) if m["end"] <= i), None)
        nxt = next((m for m in matched if m["pos"] >= j), None)
        region = None
        if prev and nxt and prev["tesseract"]["image"] == nxt["tesseract"]["image"]:
            lo, hi = sorted((prev["tesseract"]["bbox"][1], nxt["tesseract"]["bbox"][3]))
            if hi - lo <= MAX_REGION_HEIGHT:
                region = {"image": prev["tesseract"]["image"], "bbox": [0, max(lo - 2 * CROP_PAD, 0), 10 ** 6, hi + 2 * CROP_PAD]}
        items.append({"typhoon": {"text": _span_display(segs, i, j)}, "tesseract": None, "region": region,
                      "auto_accept": False, "audit": False, "kind": "typhoon_only", "score": 0.0, "pos": i, "end": j,
                      "snap": 0, "gap_merged": 0, "cut_mid": False, "cut_mid_sides": []})

    # invariant: ทุกตัวของ hay อยู่ในรายการ agree/diff/typhoon_only พอดี 1 ครั้ง
    cover = [0] * n_hay
    for it in items:
        if it["pos"] is not None:
            for i in range(it["pos"], it["end"]):
                cover[i] += 1
    dropped, doubled = cover.count(0), sum(1 for v in cover if v > 1)
    assert dropped == 0 and doubled == 0, f"{file} p{page}: ตัวอักษร Typhoon ตก {dropped} / ซ้ำ {doubled}"

    for it in items:
        if it["kind"] == "typhoon_only":
            it["covered_by"], it["maybe_duplicate_of"] = [], []
    classify_noise([it for it in items if it["kind"] != "typhoon_only"])

    # ตำแหน่งของ tesseract_only (รวม fragment/duplicate) ในลำดับของ Typhoon: ต่อท้ายบรรทัด Tesseract ก่อนหน้าที่จับคู่ได้
    last_end = 0
    by_k = sorted(items, key=lambda it: it.get("k", 10 ** 9))
    for it in by_k:
        if it["kind"] in TESS_ONLY_KINDS:
            it["pos"] = last_end - 0.5
        elif it["kind"] in ("agree", "diff"):
            last_end = it["end"]
    for it in items:
        it.pop("k", None)
    items.sort(key=lambda it: (it["pos"], 1 if it["kind"] in TESS_ONLY_KINDS else 0))

    agree = [it for it in items if it["kind"] == "agree"]
    if agree:
        rng = random.Random(f"{AUDIT_SEED}:{file}:{page}")
        for it in rng.sample(agree, max(1, round(AUDIT_FRACTION * len(agree)))):
            it["audit"] = True

    stem = Path(file).stem
    for n, it in enumerate(items, 1):
        it["id"] = f"{stem}_p{page}_{n:03d}"
        it["suggest"] = _suggest(it)
        it["order"] = n - 1
        it["pos_end"] = it.pop("end", it["pos"])  # ท้ายช่วงข้อความ Typhoon (ไม่รวม; tesseract_only/fragment/duplicate = pos)
    agree_to_diff = []
    for it in items:
        before = it.pop("_before", None)
        if before and before["agree"] and it["kind"] == "diff":
            agree_to_diff.append({"id": it["id"], "tesseract": it["tesseract"]["text"], "typhoon_before": before["typhoon"],
                                  "typhoon_after": it["typhoon"]["text"]})
    stats["agree_to_diff"] = agree_to_diff
    stats["cut_mid"] = [it["id"] for it in items if it["cut_mid"]]
    for it in items:
        it["covered_by"] = [o["id"] for o in it["covered_by"]]
        it["maybe_duplicate_of"] = [o["id"] for o in it["maybe_duplicate_of"]]
    counts = {"agree": 0, "agree_audit": 0, "diff": 0, "typhoon_only": 0, "tesseract_only": 0, "fragment": 0, "duplicate": 0}
    for it in items:
        if it["kind"] == "agree":
            counts["agree_audit" if it["audit"] else "agree"] += 1
        else:
            counts[it["kind"]] += 1
    counts["to_judge"] = counts["agree_audit"] + counts["diff"] + counts["typhoon_only"] + counts["tesseract_only"]
    counts["maybe_duplicate"] = sum(1 for it in items if it["maybe_duplicate_of"])  # เซตย่อยของ to_judge (ทับทางเดียว)
    return {
        "file": file, "page": page,
        "typhoon": {"backend": typhoon.backend, "model": typhoon.model, "revision": typhoon.revision,
                    "seconds": typhoon.seconds, "tokens": typhoon.tokens},
        "tesseract": {"backend": tesseract.backend, "model": tesseract.model, "revision": tesseract.revision,
                      "seconds": tesseract.seconds},
        "typhoon_page_numbers": page_numbers,
        "column_split": [(p.get("column_split") or {}).get("x") for p in parts],
        "invariant": {"hay_chars": n_hay, "dropped": dropped, "doubled": doubled},
        "span_stats": stats,
        "figures": figures, "skipped_tesseract": skipped, "counts": counts, "items": items,
    }


def _display(line: str) -> str:
    """ข้อความที่แสดงให้คน/เก็บเป็น "ข้อความที่ Typhoon อ่านได้": ตัดเครื่องหมาย markdown (# * |) แต่คงช่องว่าง"""
    return _WS.sub(" ", _MD_CHARS.sub(" ", line)).strip()


def _suggest(it: dict) -> str:
    """ตัวที่ "ใกล้กว่า" สำหรับใส่ให้คนแก้ต่อ: เทียบสัดส่วนคำไทยที่อยู่ในพจนานุกรม (วัดอย่างเดียว ไม่ใช่ข้อความที่ถูก)
    ตัวใดไม่มีให้ใช้อีกตัว; เสมอ/ไม่มีอักษรไทย -> Typhoon"""
    ty, te = it.get("typhoon"), it.get("tesseract")
    if not ty:
        return "tesseract"
    if not te:
        return "typhoon"
    from app.extraction.quality import dictionary_ratio

    a, b = dictionary_ratio(ty["text"]), dictionary_ratio(te["text"])
    return "tesseract" if (a is not None and b is not None and b > a) else "typhoon"


# ---------- ภาพ + HTML ----------

def _jpeg_b64(im, quality: int = 80) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def attach_images(page_data: dict, images_png: list[bytes]) -> dict:
    """ภาพครอปต่อรายการที่ต้องตัดสิน + ภาพหน้า (ย่อ) -> ใส่ใน page_data (เฉพาะใช้ทำ HTML ไม่เขียนลง items.json)"""
    from PIL import Image

    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in images_png]
    by_id = {it["id"]: it for it in page_data["items"]}
    out_items = []
    for it in page_data["items"]:
        if it["auto_accept"] and not it["audit"]:
            continue
        box = (it.get("tesseract") or {}).get("bbox") if it["kind"] != "typhoon_only" else None
        img_i = (it.get("tesseract") or {}).get("image", 0) if box else None
        if it["kind"] == "typhoon_only" and it["region"]:
            img_i, box = it["region"]["image"], it["region"]["bbox"]
        related = []
        for rid, relation in [(r, "covered_by") for r in it.get("covered_by", [])] + [(r, "maybe_duplicate_of") for r in it.get("maybe_duplicate_of", [])]:
            o = by_id.get(rid)
            if o:
                related.append({"id": rid, "kind": o["kind"], "relation": relation,
                                "tesseract": (o.get("tesseract") or {}).get("text"), "typhoon": (o.get("typhoon") or {}).get("text")})
        if it["kind"] in ("fragment", "duplicate") and box:  # ภาพของรายการที่ถูกกรองออก: รวมรายการที่ทับไว้ในกรอบด้วยให้เห็นบริบท
            for rid in it["covered_by"]:
                o = by_id.get(rid)
                if o and o.get("tesseract") and o["tesseract"]["image"] == img_i:
                    ob = o["tesseract"]["bbox"]
                    box = [min(box[0], ob[0]), min(box[1], ob[1]), max(box[2], ob[2]), max(box[3], ob[3])]
        crop = None
        if box is not None and img_i is not None and img_i < len(imgs):
            im = imgs[img_i]
            x0, y0, x1, y1 = box
            c = im.crop((max(x0 - CROP_PAD, 0), max(y0 - CROP_PAD, 0), min(x1 + CROP_PAD, im.width), min(y1 + CROP_PAD, im.height)))
            if c.width > 0 and c.height > 0:
                crop = _jpeg_b64(c)
        out_items.append({**it, "crop": crop, "related": related})
    thumbs = []
    for im in imgs:
        s = PAGE_THUMB_LONG_SIDE / max(im.size)
        thumbs.append(_jpeg_b64(im.resize((int(im.width * s), int(im.height * s))), 70) if s < 1 else _jpeg_b64(im, 70))
    return {"file": page_data["file"], "page": page_data["page"], "counts": page_data["counts"],
            "items": out_items, "page_images": thumbs, "page_sizes": [list(im.size) for im in imgs]}


def build_html(pages: list[dict]) -> str:
    """รายการที่ถูกกรองออก (fragment/duplicate) อยู่ในหน้าตรวจไว้เปิดดู แต่ไม่นับใน total / dataset_id (ต้องตัดสินแค่รายการอื่น)"""
    to_judge = [[it["id"] for it in p["items"] if it["kind"] in ("agree", "diff", "typhoon_only", "tesseract_only")] for p in pages]
    total = sum(len(ids) for ids in to_judge)
    dataset_id = hashlib.sha256(json.dumps(to_judge).encode()).hexdigest()[:16]
    payload = json.dumps({"dataset_id": dataset_id, "total": total, "pages": pages}, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")  # กัน </script> ในข้อความ
    return TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload)


def _short_name(file: str, page: int) -> str:
    """ONEREPORTKBANKT.PDF หน้า 87 -> kbank87 (ชื่อไฟล์ debug)"""
    stem = Path(file).stem.lower()
    for prefix in ("e_one_report_", "onereport"):
        stem = stem.removeprefix(prefix)
    return f"{stem.removesuffix('t')}{page}"


def save_column_debug(file: str, page: int, tesseract: OcrResult, images_png: list[bytes], out_dir: Path = DEBUG_DIR) -> list[Path]:
    """ภาพ crop ของแต่ละ region ที่ Tesseract แยก (แถบข้าง / เนื้อหา) พร้อมกรอบบรรทัดที่อ่านได้ (พิกัดภาพเดิม แล้วลบ offset ของ region)
    ไว้ตรวจด้วยตาว่าเส้นแบ่งถูกที่และ bbox ที่บวก offset กลับตรงตำแหน่ง — คืน path ที่เขียน (ไม่มีการแยก = ลิสต์ว่าง)"""
    from PIL import Image, ImageDraw

    written: list[Path] = []
    for i, part in enumerate(tesseract_parts(tesseract)):
        split = part.get("column_split")
        if not split or i >= len(images_png):
            continue
        im = Image.open(io.BytesIO(images_png[i])).convert("RGB")
        x = split["x"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for col, (name, x0, x1, color) in enumerate((("side", 0, x, "red"), ("body", x, im.width, "blue"))):
            crop = im.crop((x0, 0, x1, im.height))
            draw = ImageDraw.Draw(crop)
            for ln in part["lines"]:
                if ln.get("col") == col:
                    b = ln["bbox"]
                    draw.rectangle([b[0] - x0, b[1], b[2] - x0, b[3]], outline=color, width=2)
            path = out_dir / f"{_short_name(file, page)}_cols_img{i}_{name}.png"
            crop.save(path)
            written.append(path)
    return written


# ---------- main ----------

def main(argv: list[str] | None = None) -> int:
    import importlib.util

    from app.extraction.ocr import get_backend, recognize_page_cached
    from app.extraction.pipeline import _hash_file
    from app.models import DetectedType

    spec = importlib.util.spec_from_file_location("ocr_pages", Path(__file__).with_name("ocr_pages.py"))
    ocr_pages = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ocr_pages)

    ap = argparse.ArgumentParser(description="สร้างรายการให้คนตัดสินข้อความที่ถูก")
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ty, te = get_backend("typhoon-local"), get_backend("tesseract-local")
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    html_pages = []
    total_to_judge = 0
    for name, page_no in ocr_pages.PAGES:
        path = DATASETS / name
        h = _hash_file(path)[0]
        # ต้องมีในแคชแล้ว (ไม่เรียกโมเดลที่นี่): ใช้ cache อย่างเดียว
        r_ty, pg, hit1 = recognize_page_cached(ty, path, DetectedType.PDF, h, page_no)
        r_te, _, hit2 = recognize_page_cached(te, path, DetectedType.PDF, h, page_no)
        if not (hit1 and hit2):
            print(f"[คำเตือน] {name} p{page_no}: ไม่พบในแคชครบ ({'typhoon' if not hit1 else ''} {'tesseract' if not hit2 else ''}) "
                  "ผลถูกรันใหม่ตอนนี้ — ควรรัน scripts/ocr_pages.py ก่อน", file=sys.stderr)
        data = build_items(name, page_no, r_ty, r_te)
        data["split"] = {"method": pg.split_method, "x": pg.split_x, "dpi": pg.dpi}
        out = TRUTH_DIR / f"{Path(name).stem}_p{page_no}.items.json"
        out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        c = data["counts"]
        total_to_judge += c["to_judge"]
        print(f"{name} p{page_no}: agree(รับอัตโนมัติ)={c['agree']} agree(สุ่มตรวจ)={c['agree_audit']} diff={c['diff']} "
              f"typhoon_only={c['typhoon_only']} tesseract_only={c['tesseract_only']} (อาจซ้ำ {c['maybe_duplicate']}) "
              f"fragment={c['fragment']} duplicate={c['duplicate']} -> ต้องตัดสิน {c['to_judge']}  "
              f"(ข้ามบรรทัดขยะของ Tesseract {len(data['skipped_tesseract'])}, figure {len(data['figures'])})  "
              f"dropped={data['invariant']['dropped']}  snap {data['span_stats']['snapped_items']} ช่วง/{data['span_stats']['snapped_chars']} ตัว  "
              f"-> {out.name}")
        for dbg in save_column_debug(name, page_no, r_te, pg.images):
            print(f"  แยกคอลัมน์ x={data['column_split']} -> {dbg}")
        if not args.no_html:
            html_pages.append(attach_images(data, pg.images))
    print(f"รวมรายการที่คนต้องตัดสิน: {total_to_judge}")
    if not args.no_html:
        REVIEW_HTML.parent.mkdir(parents=True, exist_ok=True)
        REVIEW_HTML.write_text(build_html(html_pages), encoding="utf-8")
        print(f"หน้าตรวจ: {REVIEW_HTML}  ({REVIEW_HTML.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
