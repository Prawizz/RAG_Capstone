"""Normalizer กลาง v1 (ตัวเดียวของระบบ — CLAUDE.md ข้อ 1)

ข้อความทุกส่วนต้องผ่านที่นี่ ทั้งตอน ingest (``Unit.text`` / ``Block.text``) และตอนรับคำถามตอน retrieve
ห้ามเขียน normalizer ตัวที่สอง

ขอบเขต: **normalize เท่านั้น ไม่ repair** (CLAUDE.md ข้อ 5) ทุกกฎเป็นกฎตายตัวที่ไม่ต้องเดาความหมาย
ไม่ใช้พจนานุกรม ไม่ลบเนื้อหาที่ไม่แน่ใจ สิ่งที่เห็นแต่ไม่แก้จะนับเป็น ``signals`` ให้ด่านตรวจคุณภาพ (T04) ใช้

API
- ``normalize_with_report(text) -> NormalizeResult``: ``text`` + ``fixes`` (จำนวนที่แก้รายกฎ) + ``signals``
- ``normalize(text) -> str``: คืนแค่ text (ใช้กับคำถามตอน retrieve)

กฎ (เรียงตามนี้; ``RULES``)
 1. nfc               unicodedata NFC (นับเป็นจำนวน "ก้อน" ที่คั่นด้วยช่องว่างซึ่งเปลี่ยน)
 2. zero_width        ลบ U+200B U+200C U+200D U+FEFF U+00AD
 3. line_endings      "\\r\\n" -> "\\n" แล้ว "\\r" ที่เหลือ -> "\\n" (ต้องมาก่อนกฎ 4-5 ไม่งั้น \\r กลายเป็น U+FFFD
                      ในคำถามที่มาจาก Windows/Excel)
 4. control_after_mark ลบอักขระควบคุม (Cc ยกเว้น \\n \\t) ที่ตามหลัง "ำ"/"ํ" หรือเครื่องหมายตัวเดียวกันสองตัวติดกัน
                      (รวมกรณีตามหลังผ่านอักขระควบคุม/U+FFFD ที่อยู่ติดกัน) = ขยะของฟอนต์ (PTT ~10k จุด)
 5. control_to_fffd   อักขระควบคุมที่เหลือทั้งหมด (ยกเว้น \\n \\t) -> U+FFFD ตัวละหนึ่งตัว ห้ามลบ
                      เพราะมักเป็นตัวอักษรจริงที่ map ผิด (PTT "สร้\\x1cง" = สร้าง, KBANK "เกณ\\x92์" = เกณฑ์)
                      ต้องเหลือหลักฐานในข้อความ นับต่อใน signal fffd_remaining
 6. marker            ลบ "#MARKER:XXX" (CPALL: TOC, PART_n, PAGE_PART_n, ATTACHMENT, PAGE_ATTACHMENTS)
 7. thai_digits       ๐–๙ -> 0–9
 8. sara_am           "ํา" / "าํ" (มีวรรณยุกต์คั่นได้) -> ำ
 9. mark_order        วรรณยุกต์/ทัณฑฆาตที่มาก่อนสระบน/ล่าง -> สลับเป็นลำดับมาตรฐาน
10. fffd_after_mark   ลบ U+FFFD ที่ตามหลัง "ำ"/"ํ" หรือเครื่องหมายตัวเดียวกันสองตัวติดกัน (เงื่อนไขเดียวกับกฎ 4)
11. dup_mark          ยุบเครื่องหมาย/สระตัวเดียวกันที่ติดกัน และ "เเ" -> "แ" (ไม่ยุบพยัญชนะ ตัวเลข ๆ ฯ)
12. space_before_mark ลบช่องว่างเดี่ยวระหว่างพยัญชนะ (ที่ไม่มีสระนำหน้า) กับเครื่องหมายที่ตามมา
13. whitespace        Unicode space/tab -> space, ยุบช่องว่างซ้อน, ตัดช่องว่างหัว/ท้ายบรรทัด, \\n{3,} -> \\n\\n
                      และตัดช่องว่าง/ขึ้นบรรทัดหัวท้ายของทั้งข้อความ

ทำไมกฎ 4 และ 10 ต้องดูว่าเครื่องหมายซ้ำหรือเป็น ำ/ํ: ฟอนต์ที่พัง (Thongterm) พ่นเครื่องหมายซ้ำแล้วตามด้วย
อักขระควบคุม/U+FFFD เป็นขยะ แต่เครื่องหมายเดี่ยวๆ ตามด้วยอักขระควบคุมอาจเป็นตัวอักษรจริง
("สร้\\x1cง": \\x1c = า ตามหลัง ร้) การลบทุกตัวที่ตามหลังเครื่องหมายจะทำให้ได้ "สร้ง" ซึ่งไม่เหลือหลักฐาน
(วัดแล้ว PTT: ตามหลังเครื่องหมายซ้ำ 9,533, ตามหลัง ำ/ํ 494, ตามหลังเครื่องหมายเดี่ยวอื่น 12)
เหตุผลที่กฎ 10 อยู่ก่อนกฎ 11 (ย้ายจากลำดับเดิม): ต้องเห็นเครื่องหมายซ้ำก่อนถูกยุบ ไม่งั้น U+FFFD ที่แปลงมา
จากกฎ 5 ตามหลัง "ร้" จะถูกกฎ 10 ลบ และ normalize รอบสองจะไม่ idempotent

ลำดับเป็นไปตามด้านบน และเนื่องจากบางกฎสร้างเงื่อนไขให้กฎก่อนหน้า (เช่น กฎ 12 ลบช่องว่างแล้วเกิด
"ก่ิ" ที่กฎ 9 ต้องแก้ หรือ NBSP ที่กฎ 13 เพิ่งแปลงเป็นช่องว่างให้กฎ 12 เห็น) จึงรันทั้งชุดซ้ำจนข้อความ
ไม่เปลี่ยน (สูงสุด ``MAX_PASSES``) เพื่อรับประกัน idempotent: normalize(normalize(x)) == normalize(x)
ตัวนับ ``fixes`` รวมทุกรอบ (รอบที่ 2 เป็นต้นไปปกติไม่มีอะไรแก้ จึงไม่กระทบตัวเลข)

signals (นับอย่างเดียว ห้ามแก้ — การแก้คือการเดา)
- nbsp_between_thai        NBSP ระหว่างอักษรไทยสองตัว (นับก่อนกฎ whitespace) อาจเป็น ฟ ที่ map ผิด
- conflicting_tone_marks   วรรณยุกต์ต่างชนิดซ้อนกัน ("ก่้")
- dangling_mark            สระบน/ล่าง/วรรณยุกต์ที่ไม่มีอักษรไทยนำหน้า (ต้นข้อความ หลังช่องว่าง/ตัวที่ไม่ใช่ไทย)
- invalid_mark_sequence    สระบน/ล่าง (ั ิ ี ึ ื ็ ุ ู ฺ) ตามด้วยสระบน/ล่างอีกตัวบนพยัญชนะเดียวกัน โดยมีวรรณยุกต์/์ คั่นได้
                           ("ชีุ้", "ลีั่", "ปฏฺิ") ภาษาไทยไม่มีลำดับนี้ = ฟอนต์พัง ไม่แก้เพราะไม่รู้ว่าตัวไหนผิด (T04)
- fffd_remaining           U+FFFD ที่ยังเหลือ
- space_between_thai       ช่องว่างระหว่างอักษรไทยสองตัว (เก็บไว้ตัดสินทีหลัง)
(ห้าตัวหลังนับจากข้อความผลลัพธ์; fffd_remaining รวม U+FFFD ที่กฎ 5 แปลงมาด้วย)

ที่ตั้งใจไม่ทำ: ไม่เรียก ``pythainlp.util.normalize()`` ทั้งก้อน (ลบเนื้อหาเงียบๆ), ไม่แปลง ๅ -> า,
ไม่ลบช่องว่างระหว่างคำไทย, ไม่แก้อักษรละตินกลางคำไทย (repair/T04)
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

NORMALIZER_VERSION = "1.1"
MAX_PASSES = 5

# ---------- ชุดอักขระ ----------
_CONS = "\u0e01-\u0e2e"
_ABOVE_V = "\u0e31\u0e34\u0e35\u0e36\u0e37"  # ั ิ ี ึ ื
_BELOW_V = "\u0e38\u0e39"  # ุ ู
_TONES = "\u0e48\u0e49\u0e4a\u0e4b"  # ่ ้ ๊ ๋
_THANTHAKHAT = "\u0e4c"
# สระ/เครื่องหมายที่ห้อยบนหรือล่างพยัญชนะ (รวม ฺ ็ ํ ๎) ใช้กับกฎ space_before_mark, dup_mark และ signal dangling
_DANGLING = _ABOVE_V + _BELOW_V + _TONES + "\u0e3a\u0e4c\u0e4d\u0e4e"
# ตัวที่ FFFD ของฟอนต์มักตามหลัง (กฎ 9): สระบน, ็, วรรณยุกต์, ์, ำ, ํ
_BEFORE_FFFD = _ABOVE_V + "\u0e47" + _TONES + "\u0e4c\u0e33\u0e4d"
_REPEATABLE = _ABOVE_V + _BELOW_V + "\u0e3a\u0e47" + _TONES + "\u0e4c\u0e4d\u0e4e\u0e33"
_THAI_LETTER = "\u0e01-\u0e4e"
# สระ (ใช้เช็คเงื่อนไข "หน้าพยัญชนะไม่มีสระ" ของกฎ 10 เหมือน pythainlp)
_VOWELS = "\u0e24\u0e26\u0e30-\u0e3a\u0e40-\u0e45\u0e47\u0e4d"

_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\ufeff\u00ad]")
_CTRL = "\x00-\x08\x0b-\x1f\x7f-\x9f"  # Cc ยกเว้น \t(\x09) \n(\x0a)
_CONTROL = re.compile(f"[{_CTRL}]")
_CRLF = re.compile("\r\n")
_CR = re.compile("\r")
# ตัวทริกเกอร์ของขยะฟอนต์: เครื่องหมายตัวเดียวกันสองตัวติดกัน หรือ ำ/ํ  (group 1 = ทริกเกอร์, 2 = เครื่องหมายที่ซ้ำ)
_JUNK_TRIGGER = f"((?:([{_BEFORE_FFFD}])\\2|[\u0e33\u0e4d]))"
_CONTROL_AFTER_MARK = re.compile(f"{_JUNK_TRIGGER}([{_CTRL}\ufffd]+)")
_MARKER = re.compile(r"#MARKER:[A-Za-z0-9_\-]*")
_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_THAI_DIGIT_RE = re.compile("[\u0e50-\u0e59]")

_TONE_CLASS = f"[{_TONES}]"
# ํา / าํ โดยมีวรรณยุกต์คั่นได้ (วรรณยุกต์ย้ายไปอยู่หน้า ำ ตามลำดับมาตรฐาน)
_SARA_AM_1 = re.compile(f"\u0e4d({_TONE_CLASS}*)\u0e32")
_SARA_AM_2 = re.compile(f"\u0e32({_TONE_CLASS}*)\u0e4d")
_MARK_ORDER = re.compile(f"([{_TONES}{_THANTHAKHAT}]+)([{_ABOVE_V}{_BELOW_V}]+)")
_DUP_MARK = re.compile(f"([{_REPEATABLE}])\\1+")
_DUP_SARA_E = re.compile("\u0e40\u0e40")
_FFFD_AFTER_MARK = re.compile(f"{_JUNK_TRIGGER}(\ufffd+)")
# พยัญชนะ + ช่องว่างเดี่ยว + เครื่องหมาย โดยหน้าพยัญชนะต้องไม่มีสระ (ไม่งั้น "ภาพ ุ่" จะพัง)
_SPACE_BEFORE_MARK = re.compile(
    f"([{_CONS}])(?<![{_VOWELS}][{_CONS}]) ([{_DANGLING}])"
)
_UNICODE_SPACE = re.compile("[\t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
_MULTI_SPACE = re.compile(" {2,}")
_LINE_EDGE_SPACE = re.compile(r"(?m)^ +| +$")
_MANY_NEWLINES = re.compile("\n{3,}")

_NBSP_BETWEEN_THAI = re.compile(f"(?<=[{_THAI_LETTER}])\u00a0(?=[{_THAI_LETTER}])")
_CONFLICTING_TONES = re.compile(f"([{_TONES}])(?!\\1)[{_TONES}]")
_DANGLING_MARK = re.compile(f"(?<![{_THAI_LETTER}])[{_ABOVE_V}{_BELOW_V}{_TONES}]")
_STACKED_V = _ABOVE_V + _BELOW_V + "ฺ็"  # สระบน/ล่าง รวม ฺ ็
_INVALID_MARK_SEQ = re.compile(f"[{_STACKED_V}][{_TONES}{_THANTHAKHAT}]*[{_STACKED_V}]")
_SPACE_BETWEEN_THAI = re.compile(f"(?<=[{_THAI_LETTER}]) (?=[{_THAI_LETTER}])")

RULE_NAMES = (
    "nfc", "zero_width", "line_endings", "control_after_mark", "control_to_fffd", "marker", "thai_digits",
    "sara_am", "mark_order", "fffd_after_mark", "dup_mark", "space_before_mark", "whitespace",
)
SIGNAL_NAMES = (
    "nbsp_between_thai", "conflicting_tone_marks", "dangling_mark", "invalid_mark_sequence",
    "fffd_remaining", "space_between_thai",
)


@dataclass
class NormalizeResult:
    text: str
    fixes: dict[str, int] = field(default_factory=lambda: dict.fromkeys(RULE_NAMES, 0))
    signals: dict[str, int] = field(default_factory=lambda: dict.fromkeys(SIGNAL_NAMES, 0))

    @property
    def total_fixes(self) -> int:
        return sum(self.fixes.values())


# ---------- กฎ: แต่ละกฎคือ text -> (text ใหม่, จำนวนที่แก้) ----------

def _subn(pattern: re.Pattern, repl, text: str) -> tuple[str, int]:
    return pattern.subn(repl, text)


def _nfc(text: str) -> tuple[str, int]:
    if unicodedata.is_normalized("NFC", text):
        return text, 0
    count = 0

    def fix(m: re.Match) -> str:
        nonlocal count
        new = unicodedata.normalize("NFC", m.group(0))
        if new != m.group(0):
            count += 1
        return new

    return re.sub(r"\S+", fix, text), count


def _zero_width(text: str) -> tuple[str, int]:
    return _subn(_ZERO_WIDTH, "", text)


def _line_endings(text: str) -> tuple[str, int]:
    text, n1 = _subn(_CRLF, "\n", text)
    text, n2 = _subn(_CR, "\n", text)
    return text, n1 + n2


def _control_after_mark(text: str) -> tuple[str, int]:
    removed = 0

    def fix(m: re.Match) -> str:
        nonlocal removed
        run = m.group(3)
        kept = _CONTROL.sub("", run)  # ลบเฉพาะอักขระควบคุม U+FFFD ในช่วงเดียวกันคงไว้ให้กฎ 9
        removed += len(run) - len(kept)
        return m.group(1) + kept

    return _CONTROL_AFTER_MARK.sub(fix, text), removed


def _control_to_fffd(text: str) -> tuple[str, int]:
    return _subn(_CONTROL, "\ufffd", text)


def _marker(text: str) -> tuple[str, int]:
    return _subn(_MARKER, "", text)


def _thai_digits(text: str) -> tuple[str, int]:
    n = len(_THAI_DIGIT_RE.findall(text))
    return (text.translate(_THAI_DIGITS), n) if n else (text, 0)


def _sara_am(text: str) -> tuple[str, int]:
    text, n1 = _subn(_SARA_AM_1, "\\1\u0e33", text)
    text, n2 = _subn(_SARA_AM_2, "\\1\u0e33", text)
    return text, n1 + n2


def _mark_order(text: str) -> tuple[str, int]:
    total = 0
    while True:  # สลับจนนิ่ง (เช่น "ก่่ิ" ต้องสลับสองครั้ง)
        text, n = _subn(_MARK_ORDER, "\\2\\1", text)
        if not n:
            return text, total
        total += n


def _dup_mark(text: str) -> tuple[str, int]:
    text, n1 = _subn(_DUP_MARK, "\\1", text)
    text, n2 = _subn(_DUP_SARA_E, "\u0e41", text)
    return text, n1 + n2


def _fffd_after_mark(text: str) -> tuple[str, int]:
    removed = 0

    def fix(m: re.Match) -> str:
        nonlocal removed
        removed += len(m.group(3))
        return m.group(1)

    return _FFFD_AFTER_MARK.sub(fix, text), removed


def _space_before_mark(text: str) -> tuple[str, int]:
    return _subn(_SPACE_BEFORE_MARK, "\\1\\2", text)


def _whitespace(text: str) -> tuple[str, int]:
    text, n1 = _subn(_UNICODE_SPACE, " ", text)
    text, n2 = _subn(_MULTI_SPACE, " ", text)
    text, n3 = _subn(_LINE_EDGE_SPACE, "", text)
    text, n4 = _subn(_MANY_NEWLINES, "\n\n", text)
    stripped = text.strip()
    n5 = int(stripped != text)  # ช่องว่าง/ขึ้นบรรทัดหัวท้ายของทั้งข้อความ
    return stripped, n1 + n2 + n3 + n4 + n5


RULES: tuple[tuple[str, Callable[[str], tuple[str, int]]], ...] = (
    ("nfc", _nfc),
    ("zero_width", _zero_width),
    ("line_endings", _line_endings),
    ("control_after_mark", _control_after_mark),
    ("control_to_fffd", _control_to_fffd),
    ("marker", _marker),
    ("thai_digits", _thai_digits),
    ("sara_am", _sara_am),
    ("mark_order", _mark_order),
    ("fffd_after_mark", _fffd_after_mark),
    ("dup_mark", _dup_mark),
    ("space_before_mark", _space_before_mark),
    ("whitespace", _whitespace),
)
assert tuple(n for n, _ in RULES) == RULE_NAMES


def _count_final_signals(text: str, signals: dict[str, int]) -> None:
    signals["conflicting_tone_marks"] += len(_CONFLICTING_TONES.findall(text))
    signals["dangling_mark"] += len(_DANGLING_MARK.findall(text))
    signals["invalid_mark_sequence"] += len(_INVALID_MARK_SEQ.findall(text))
    signals["fffd_remaining"] += text.count("\ufffd")
    signals["space_between_thai"] += len(_SPACE_BETWEEN_THAI.findall(text))


def normalize_with_report(text: str) -> NormalizeResult:
    result = NormalizeResult(text=text)
    for pass_no in range(MAX_PASSES):
        before = text
        for name, rule in RULES:
            if name == "whitespace" and pass_no == 0:
                # นับก่อนกฎ 11 จะแปลง NBSP เป็นช่องว่าง
                result.signals["nbsp_between_thai"] += len(_NBSP_BETWEEN_THAI.findall(text))
            text, n = rule(text)
            result.fixes[name] += n
        if text == before:
            break
    result.text = text
    _count_final_signals(text, result.signals)
    return result


def normalize(text: str) -> str:
    return normalize_with_report(text).text
