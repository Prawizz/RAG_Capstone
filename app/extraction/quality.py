"""ด่านตรวจคุณภาพ (T04): ตัดสิน ``Unit.status`` ว่าใช้ข้อความที่ดึงได้ต่อ (ok) หรือต้องส่ง OCR (needs_ocr)

ใช้กับ unit ทุกประเภท ตัดสินทั้ง unit (ห้ามตัดทิ้งบาง block) ใช้ **จำนวนจุดพัง** ไม่ใช้สัดส่วน
เพราะหน้าที่สะอาดมีจุดพังไม่เกิน 1 จุดต่อหน้า ส่วนหน้าที่พังมีหลายสิบถึงหลายร้อยจุด และ KBANK ฟอนต์พัง
เกือบทั้งไฟล์ เกณฑ์สัดส่วนจะปล่อยผ่านเกือบทั้งไฟล์ ความผิดพลาดที่แย่กว่าคือปล่อยหน้าพังผ่าน
(ระบบตอบจากข้อความพัง) ส่งหน้าดีไป OCR เกินแค่เสียค่าใช้จ่าย

ราย unit (``evaluate_unit``) เรียงตามลำดับ
  b. ตัวอักษร (ไทย+ละติน = gate:letters) < MIN_LETTERS (รวมข้อความว่าง) และ image_coverage >= SCAN_IMAGE_COVERAGE
                                                  -> needs_ocr + scan_like (ไม่มี image_coverage = ข้าม)
     (อยู่ก่อนกฎ a: PDF สแกนไม่มีชั้นข้อความจะได้ข้อความว่างทุกหน้า ต้องไม่หลุดเป็น ok+empty)
  a. text ว่าง (ไม่มีภาพใหญ่)                     -> ok + empty ตามกติกาเดิม
  c. broken_events >= BROKEN_EVENTS_THRESHOLD     -> needs_ocr + flag ตามชนิดที่พบ
  d. letters < MIN_LETTERS                        -> ok + low_text
  e. นอกนั้น                                      -> ok
สไลด์ (type=slide) ใช้เกณฑ์ของตัวเองแทนกฎ b: image_coverage >= SLIDE_IMAGE_COVERAGE (0.25 เพราะแผนภาพกินพื้นที่
25–45% ได้ตามปกติ) และ letters < SLIDE_MIN_LETTERS -> needs_ocr + scan_like; letters >= SLIDE_MIN_LETTERS -> ok +
image_content (ข้อความถูกแต่ไม่ครบ เนื้อหาหลักอยู่ในรูป ค้นด้วยหัวเรื่องได้ทันที OCR รูปทีหลัง) ถัดจากนั้นใช้กฎ a, c, d, e
นับ letters ไม่ใช่นับอักษรไทยอย่างเดียว เพราะหน้า/สไลด์ภาษาอังกฤษไม่ควรถูกตัดสินว่าข้อความน้อย; กฎนับจุดพัง
และกฎระดับไฟล์ยังใช้อักษรไทยเหมือนเดิม

``broken_events`` = norm_sig:fffd_remaining + invalid_mark_sequence + conflicting_tone_marks
+ dangling_mark + จำนวนกลุ่มอักษรละตินที่ติดกับอักษรไทยโดยตรง + จำนวนอักขระในท่อน mojibake (นับจาก ``Unit.text``)
"ปตท. (PTT)" / "รุ่น iPhone 15" ไม่นับ เพราะมีช่องว่างหรือเครื่องหมายคั่น
mojibake = อักขระชุด U+00A1–U+00FF + อักขระพิเศษของ cp1252 ต่อกันตั้งแต่ 3 ตัว (ข้อความไทยที่ถูกถอดผิดเป็น Latin-1
เช่น "¢ŒÒ¾ÃÐ¾Ø·¸à¨ŒÒ") ตัวเดี่ยวๆ ที่ถูกอย่าง © ® ° × é ไม่นับ ท่อนนี้ถูกตัดออกก่อนนับ latin_in_thai (ไม่นับซ้ำ)

ระดับไฟล์ (``apply_document_rule``): ถ้า unit ที่มีอักษรไทย >= MIN_THAI_CHARS ถูกตัดสินด้วยกฎ c เกินครึ่ง
ฟอนต์พังเป็นเรื่องทั้งไฟล์ unit ที่มีอักษรไทยพอและยัง ok จะถูกเปลี่ยนเป็น needs_ocr + document_font_broken
(หน้าที่ดูสะอาดอาจพังแบบที่ตรวจระดับอักขระไม่ได้ เช่น ี ถูก map เป็น ่)

``dictionary_ratio()`` วัดอย่างเดียว ไม่ใช้ตัดสินและ pipeline ไม่เรียก (QualityReport.dictionary_ratio เป็น None)
เพราะแยกหน้าสะอาดกับหน้าพังไม่ได้ แต่ทำให้ ingest ช้า; เรียกเองได้ผ่าน ``scripts/gate_report.py --dictionary``
``QualityReport.score`` ยังเป็น None: ยังไม่มีผู้ใช้ การตัดสินใช้ status + flags
"""

import re
from functools import lru_cache
from typing import Optional

from app.models import QualityFlag, Unit, UnitStatus, UnitType

# ค่าที่ปรับได้ที่เดียว
MIN_THAI_CHARS = 100  # ใช้กับกฎระดับไฟล์ (unit ที่มีอักษรไทยพอ) และ dictionary_ratio
MIN_LETTERS = 100  # "ข้อความน้อย" (low_text) / หน้าภาพ: นับอักษรไทย + ละติน (gate:letters)
SCAN_IMAGE_COVERAGE = 0.5
# สไลด์: แผนภาพกินพื้นที่ 25–45% ได้ตามปกติ จึงใช้เกณฑ์ต่ำกว่า PDF
SLIDE_IMAGE_COVERAGE = 0.25
SLIDE_MIN_LETTERS = 20  # ต่ำกว่านี้ + ภาพใหญ่ = ไม่มีข้อความจริง -> OCR; เท่ากับหรือมากกว่า = image_content
BROKEN_EVENTS_THRESHOLD = 3
DOCUMENT_BROKEN_SHARE = 0.5  # เกินสัดส่วนนี้ (มากกว่า ไม่รวมพอดี) ถือว่าฟอนต์ทั้งไฟล์พัง
MIN_DICT_WORD_LEN = 2

_THAI = "ก-๛"
_LATIN = "A-Za-zÀ-ÖØ-öø-ɏ"  # ไม่รวม × (D7) และ ÷ (F7)
_THAI_CHAR = re.compile(f"[{_THAI}]")
_LETTER = re.compile(f"[{_THAI}A-Za-zÀ-ÖØ-öø-ɏ]")
# กลุ่มอักษรละตินที่ติดอักษรไทยทางใดทางหนึ่งโดยตรง
_LATIN_IN_THAI = re.compile(f"(?<=[{_THAI}])[{_LATIN}]+|[{_LATIN}]+(?=[{_THAI}])")
_MOJIBAKE_CHARS = "\u00a1-\u00ff\u0152\u0153\u0160\u0161\u0178\u0192\u02c6\u02dc\u2020\u2021\u2030\u2039\u203a\u20ac\u2122"
_MOJIBAKE = re.compile(f"[{_MOJIBAKE_CHARS}]{{3,}}")
MOJIBAKE_REASON = "ข้อความเข้ารหัสเพี้ยน"

K_BROKEN = "gate:broken_events"
K_THAI = "gate:thai_chars"
K_LETTERS = "gate:letters"
K_LATIN = "gate:latin_in_thai"
K_MOJIBAKE = "gate:mojibake"


def thai_char_count(text: str) -> int:
    return len(_THAI_CHAR.findall(text))


def letters_count(text: str) -> int:
    """อักษรไทย + ละติน (รวมเครื่องหมายวรรณยุกต์/สระของไทย ไม่รวมตัวเลข เครื่องหมาย ช่องว่าง)"""
    return len(_LETTER.findall(text))


def mojibake_count(text: str) -> int:
    """จำนวนอักขระในท่อน mojibake (อักขระ Latin-1/cp1252 ต่อกัน >= 3 ตัว)"""
    return sum(len(m) for m in _MOJIBAKE.findall(text))


def latin_in_thai_count(text: str) -> int:
    """จำนวนกลุ่มอักษรละตินที่ติดอักษรไทย (ท่อน mojibake ไม่นับที่นี่ เพราะนับแยกเป็นรายอักขระ)"""
    return len(_LATIN_IN_THAI.findall(_MOJIBAKE.sub(" ", text)))


def _sig(signals: dict[str, float], name: str) -> int:
    return int(signals.get(f"norm_sig:{name}", 0))


def _breakdown(signals: dict[str, float]) -> dict[str, int]:
    return {
        "fffd": _sig(signals, "fffd_remaining"),
        "mark_seq": _sig(signals, "invalid_mark_sequence"),
        "tone": _sig(signals, "conflicting_tone_marks"),
        "dangling": _sig(signals, "dangling_mark"),
        "latin": int(signals.get(K_LATIN, 0)),
        "mojibake": int(signals.get(K_MOJIBAKE, 0)),
    }


def _broken_reason(n: int, b: dict[str, int]) -> str:
    labels = [
        ("fffd", "�"), ("mark_seq", "สระซ้อน"), ("tone", "วรรณยุกต์ซ้อน"),
        ("dangling", "เครื่องหมายลอย"), ("latin", "ละตินกลางคำไทย"),
        ("mojibake", MOJIBAKE_REASON),
    ]
    parts = ", ".join(f"{label}: {b[k]}" for k, label in labels if b[k])
    return f"พบจุดพัง {n} จุด ({parts})"


def _updated(unit: Unit, *, status: UnitStatus, reason: Optional[str], flags: list[QualityFlag],
             signals: dict[str, float], dictionary_ratio: Optional[float]) -> Unit:
    data = unit.model_dump()
    data["status"] = status
    data["reason"] = reason
    data["signals"] = signals
    data["quality"]["flags"] = list(unit.quality.flags) + [f for f in flags if f not in unit.quality.flags]
    data["quality"]["dictionary_ratio"] = dictionary_ratio
    return Unit.model_validate(data)


def evaluate_unit(unit: Unit) -> Unit:
    """ตัดสิน unit หนึ่งตัว (คืน Unit ใหม่); unit ที่ status ไม่ใช่ ok (เช่น failed) ไม่แตะ"""
    if unit.status != UnitStatus.OK:
        return unit
    thai = thai_char_count(unit.text)
    letters = letters_count(unit.text)
    latin = latin_in_thai_count(unit.text)
    signals = dict(unit.signals)
    signals[K_THAI] = float(thai)
    signals[K_LETTERS] = float(letters)
    signals[K_LATIN] = float(latin)
    signals[K_MOJIBAKE] = float(mojibake_count(unit.text))
    b = _breakdown(signals)
    broken = sum(b.values())
    signals[K_BROKEN] = float(broken)

    def done(status, reason, flags):
        return _updated(unit, status=status, reason=reason, flags=flags, signals=signals,
                        dictionary_ratio=unit.quality.dictionary_ratio)

    coverage = unit.signals.get("image_coverage")
    is_slide = unit.type == UnitType.SLIDE
    if is_slide:
        big_image = coverage is not None and coverage >= SLIDE_IMAGE_COVERAGE
        scan_like = big_image and letters < SLIDE_MIN_LETTERS
    else:
        big_image = coverage is not None and coverage >= SCAN_IMAGE_COVERAGE
        scan_like = big_image and letters < MIN_LETTERS
    if scan_like:  # b
        return done(
            UnitStatus.NEEDS_OCR,
            f"{'สไลด์' if is_slide else 'หน้า'}เป็นภาพ (ภาพใหญ่สุดครอบ {coverage:.0%} ของ{'สไลด์' if is_slide else 'หน้า'}) "
            f"มีตัวอักษรเพียง {letters} ตัว — ต้อง OCR",
            [QualityFlag.SCAN_LIKE],
        )
    if not unit.text.strip():  # a: กติกา empty เดิม
        return done(UnitStatus.OK, None, [])
    if broken >= BROKEN_EVENTS_THRESHOLD:  # c
        flags = []
        if b["fffd"]:
            flags.append(QualityFlag.REPLACEMENT_CHAR)
        if b["mark_seq"] or b["tone"] or b["dangling"]:  # สระ/วรรณยุกต์ผิดที่ทั้งสามแบบใช้ flag เดียวกัน
            flags.append(QualityFlag.INVALID_MARK_SEQUENCE)
        if b["latin"] or b["mojibake"]:
            flags.append(QualityFlag.LATIN_IN_THAI)
        return done(UnitStatus.NEEDS_OCR, _broken_reason(broken, b), flags)
    if is_slide and big_image:  # สไลด์: มีข้อความ (หัวเรื่อง ฯลฯ) แต่เนื้อหาหลักอยู่ในรูป ค้นด้วยข้อความได้ OCR รูปเพิ่มทีหลัง
        return done(UnitStatus.OK, None, [QualityFlag.IMAGE_CONTENT])
    if letters < MIN_LETTERS:  # d
        return done(UnitStatus.OK, None, [QualityFlag.LOW_TEXT])
    return done(UnitStatus.OK, None, [])  # e


def _has_enough_thai(unit: Unit) -> bool:
    return unit.signals.get(K_THAI, 0) >= MIN_THAI_CHARS


def _broken_by_rule_c(unit: Unit) -> bool:
    return (
        unit.status == UnitStatus.NEEDS_OCR
        and unit.signals.get(K_BROKEN, 0) >= BROKEN_EVENTS_THRESHOLD
        and QualityFlag.SCAN_LIKE not in unit.quality.flags
    )


def apply_document_rule(units: list[Unit]) -> list[Unit]:
    """กฎระดับไฟล์ (ดู docstring ของโมดูล) คืนรายการ unit ใหม่"""
    rich = [u for u in units if _has_enough_thai(u)]
    broken = [u for u in rich if _broken_by_rule_c(u)]
    if not rich or len(broken) <= DOCUMENT_BROKEN_SHARE * len(rich):
        return units
    reason = (
        f"ฟอนต์ของไฟล์พัง: {len(broken)} จาก {len(rich)} หน้าที่มีข้อความไทยพังชัดเจน "
        "หน้านี้ดูสะอาดแต่ไม่ไว้ใจ"
    )
    out = []
    for u in units:
        if _has_enough_thai(u) and u.status == UnitStatus.OK:
            u = _updated(u, status=UnitStatus.NEEDS_OCR, reason=reason,
                         flags=[QualityFlag.DOCUMENT_FONT_BROKEN], signals=dict(u.signals),
                         dictionary_ratio=u.quality.dictionary_ratio)
        out.append(u)
    return out


def apply_gate(units: list[Unit]) -> list[Unit]:
    """ราย unit ก่อน แล้วกฎระดับไฟล์"""
    return apply_document_rule([evaluate_unit(u) for u in units])


# ---------- สัญญาณพจนานุกรม (วัดอย่างเดียว ห้ามใช้ตัดสิน) ----------

@lru_cache(maxsize=1)
def _dictionary() -> frozenset[str]:
    from pythainlp.corpus import thai_words

    return frozenset(thai_words())


def dictionary_ratio(text: str) -> Optional[float]:
    """ตัวอักษรไทยที่อยู่ในคำซึ่งพบในพจนานุกรม (newmm, คำยาว >= 2) ÷ ตัวอักษรไทยทั้งหมด; ไม่มีอักษรไทย = None"""
    from pythainlp.tokenize import word_tokenize

    total = thai_char_count(text)
    if not total:
        return None
    words = _dictionary()
    known = 0
    for tok in word_tokenize(text, engine="newmm", keep_whitespace=False):
        if len(tok) >= MIN_DICT_WORD_LEN and tok in words:
            known += thai_char_count(tok)
    return min(known / total, 1.0)


# ---------- สรุปผลสำหรับ CLI / รายงาน ----------

GATE_KEYS = (
    "ok",
    "ok:low_text",
    "ok:image_content",
    "ok:empty",
    "needs_ocr:scan_like",
    "needs_ocr:broken",
    "needs_ocr:document_font_broken",
)


def gate_key(unit: Unit) -> str:
    """หมวดผลของด่านตรวจของ unit หนึ่งตัว (ค่าอยู่ใน GATE_KEYS; unit failed/rejected คืนชื่อสถานะ)"""
    flags = set(unit.quality.flags)
    if unit.status == UnitStatus.OK:
        if QualityFlag.EMPTY in flags:
            return "ok:empty"
        if QualityFlag.IMAGE_CONTENT in flags:
            return "ok:image_content"
        return "ok:low_text" if QualityFlag.LOW_TEXT in flags else "ok"
    if unit.status == UnitStatus.NEEDS_OCR:
        if QualityFlag.DOCUMENT_FONT_BROKEN in flags:
            return "needs_ocr:document_font_broken"
        return "needs_ocr:scan_like" if QualityFlag.SCAN_LIKE in flags else "needs_ocr:broken"
    return unit.status.value


def gate_summary(units: list[Unit]) -> dict[str, int]:
    out = {k: 0 for k in GATE_KEYS}
    for u in units:
        k = gate_key(u)
        out[k] = out.get(k, 0) + 1
    return {k: v for k, v in out.items() if v or k in ("ok",)}
