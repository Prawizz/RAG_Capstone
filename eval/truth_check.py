"""เทียบผล normalize กับ truth ที่คนพิมพ์จากภาพใน PDF (ใช้ทั้งใน test และ scripts/check_truth.py)

truth = ข้อความที่ถูกตามที่ตาเห็น ไม่เกี่ยวกับกฎของ normalizer จึงเป็น ground truth ที่ไม่ได้มาจากโค้ดเรา
(CLAUDE.md ข้อ 6) ผลการเทียบ:

- exact        out == normalize(truth)
- evidence     out เท่ากับ normalize(truth) เมื่อให้ U+FFFD แต่ละตัวแทนอักขระได้หนึ่งตัวพอดี
               (ข้อความยังพังอยู่ แต่ไม่ได้ทำให้ดูเหมือนถูก และเหลือหลักฐานให้ T04 ซ่อม)
- needs_repair ไม่ตรง แต่ out มีร่องรอย (U+FFFD, อักษรละตินติดอักษรไทย, conflicting_tone_marks,
               dangling_mark) = งาน repair/T04 ไม่ใช่ความผิดของ normalizer
- fail         ไม่ตรงและไม่มีร่องรอยเลย = ข้อความผิดแต่ดูสะอาด (CLAUDE.md ข้อ 5)

ก่อนเทียบจะตัดช่องว่างและ \\n ทั้งหมดออกจากทั้งสองข้าง (คนพิมพ์ truth จากภาพ เว้นวรรค/ขึ้นบรรทัดไม่ตรง input)
ไม่ใช้ fuzzy ratio: matcher ตรงตัวทุกอักขระยกเว้นตำแหน่ง U+FFFD
"""

import re
from typing import Optional

from app.thai.normalizer import normalize, normalize_with_report

EXACT = "exact"
EVIDENCE = "evidence"
NEEDS_REPAIR = "needs_repair"
FAIL = "fail"
NO_TRUTH = "no_truth"

FAIL_MESSAGE = "ข้อความผิดแต่ดูสะอาด (CLAUDE.md ข้อ 5)"

_LATIN = "A-Za-zÀ-ɏ"
_THAI = "ก-๎"
_LATIN_NEXT_TO_THAI = re.compile(f"[{_THAI}][{_LATIN}]|[{_LATIN}][{_THAI}]")
_WS = re.compile(r"\s+")


def strip_ws(text: str) -> str:
    return _WS.sub("", text)


def evidence_match(out: str, truth_norm: str) -> bool:
    """out ตรงกับ truth เมื่อ U+FFFD แต่ละตัวแทนอักขระหนึ่งตัวพอดี (ต้องมี FFFD อย่างน้อยหนึ่งตัว)"""
    if "�" not in out:
        return False
    pattern = ".".join(re.escape(part) for part in out.split("�"))
    return re.fullmatch(pattern, truth_norm, re.DOTALL) is not None


def traces(out: str) -> list[str]:
    """ร่องรอยว่าข้อความยังพังอยู่ (หลักฐานให้ T04)"""
    found = []
    if "�" in out:
        found.append("U+FFFD")
    if _LATIN_NEXT_TO_THAI.search(out):
        found.append("อักษรละตินติดอักษรไทย")
    signals = normalize_with_report(out).signals
    if signals["conflicting_tone_marks"] > 0:
        found.append("conflicting_tone_marks")
    if signals["dangling_mark"] > 0:
        found.append("dangling_mark")
    return found


def classify(out: str, truth: str) -> str:
    out_cmp = strip_ws(out)
    truth_cmp = strip_ws(normalize(truth))
    if out_cmp == truth_cmp:
        return EXACT
    if evidence_match(out_cmp, truth_cmp):
        return EVIDENCE
    return NEEDS_REPAIR if traces(out) else FAIL


def classify_case(case: dict) -> tuple[str, str]:
    """(ผล, out) ของคู่ทดสอบหนึ่งคู่; ไม่มี truth (null/ว่าง) = NO_TRUTH"""
    out = normalize(case["input"])
    truth: Optional[str] = case.get("truth")
    if not truth:
        return NO_TRUTH, out
    return classify(out, truth), out
