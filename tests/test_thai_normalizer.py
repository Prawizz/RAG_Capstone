"""Thai normalizer v1

คู่ทดสอบอยู่ที่ eval/datasets/thai_normalization.jsonl — expected ทุกคู่เขียนด้วยมือจากการไล่กฎ
ไม่ได้มาจากการรัน normalizer (CLAUDE.md ข้อ 6) และ verified=false จนกว่าทีมจะตรวจ
"""

import json
from pathlib import Path

import pymupdf
import pytest

from app.extraction.pdf import raw_blocks
from app.thai.normalizer import (
    NORMALIZER_VERSION,
    RULE_NAMES,
    SIGNAL_NAMES,
    normalize,
    normalize_with_report,
)

ROOT = Path(__file__).parent.parent
CASES_PATH = ROOT / "eval" / "datasets" / "thai_normalization.jsonl"
# แยกบรรทัดด้วย "\n" เท่านั้น (splitlines จะตัดที่ \x85 /   ที่อยู่ในข้อมูลด้วย)
CASES = [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").split("\n") if line.strip()]
ONE_REPORTS = ["E_ONE_REPORT_CPALLT.PDF", "ONEREPORTKBANKT.PDF", "ONEREPORTPTTT.PDF"]


def nonzero(d: dict) -> dict:
    return {k: v for k, v in d.items() if v}


def case_id(c: dict) -> str:
    return f"{c['rule']}:{c['input'][:20]!r}"


def test_version():
    assert NORMALIZER_VERSION == "1.0"


def test_dataset_shape_and_coverage():
    assert len(CASES) >= 60
    for c in CASES:
        assert {"input", "expected", "rule", "source", "verified"} <= c.keys()
        assert c["verified"] is False
    rules = {c["rule"] for c in CASES}
    assert set(RULE_NAMES) <= rules
    assert {f"signal:{s}" for s in SIGNAL_NAMES} <= rules
    assert "negative" in rules


@pytest.mark.parametrize("case", CASES, ids=[case_id(c) for c in CASES])
def test_pairs(case):
    res = normalize_with_report(case["input"])
    assert res.text == case["expected"]
    if "fixes" in case:
        assert nonzero(res.fixes) == case["fixes"]
    if "signals" in case:
        assert nonzero(res.signals) == case["signals"]


@pytest.mark.parametrize("case", CASES, ids=[case_id(c) for c in CASES])
def test_pairs_idempotent(case):
    once = normalize(case["input"])
    assert normalize(once) == once


def test_normalize_is_text_of_report():
    for c in CASES:
        assert normalize(c["input"]) == normalize_with_report(c["input"]).text


def test_query_matches_document_text():
    """คำถามที่พิมพ์ตรงกับข้อความในเอกสาร ต้อง normalize ได้ผลเหมือนกัน"""
    doc_text = "บริษัท ซีพี ออลล์ ดําเนินธุรกิจ ปีี ๒๕๖๘"
    query = "บริษัท ซีพี ออลล์ ดำเนินธุรกิจ ปี 2568"
    assert normalize(doc_text) == normalize(query) == query


def test_report_has_all_keys_and_counts_every_rule():
    res = normalize_with_report("")
    assert tuple(res.fixes) == RULE_NAMES and tuple(res.signals) == SIGNAL_NAMES
    assert not any(res.fixes.values()) and not any(res.signals.values())


def test_does_not_delete_content_like_pythainlp():
    # pythainlp.util.normalize ลบเนื้อหาเงียบๆ ในเคสเหล่านี้ เราต้องคงไว้ (นับเป็น signal แทน)
    for text in ["ภาพ ุ่ม", "ั้งแต่", "ราคา า", "ก่้า"]:
        assert normalize(text) == text


# ---------- ข้อความทั้งหน้าของ PDF จริง (ช้า) ----------

@pytest.mark.slow
@pytest.mark.parametrize("name", ONE_REPORTS)
def test_idempotent_on_real_pages(name):
    with pymupdf.open(str(ROOT / "eval" / "datasets" / name)) as doc:
        for page in doc:
            raw = "\n".join(b.text for b in raw_blocks(page))
            once = normalize(raw)
            assert normalize(once) == once, f"{name} หน้า {page.number + 1} ไม่ idempotent"


@pytest.mark.parametrize("sep", ["\r\n", "\r"])
def test_windows_line_endings_equal_unix(sep):
    """คำถามจาก Windows/Excel (\r\n, \r) ต้องได้ผลเหมือนใช้ \n และไม่มี U+FFFD ติดมา"""
    unix = "ค่าธรรมเนียม\nบัตรเครดิต\nปีี ๒๕๖๘"
    other = unix.replace("\n", sep)
    assert normalize(other) == normalize(unix) == "ค่าธรรมเนียม\nบัตรเครดิต\nปี 2568"
    assert "\ufffd" not in normalize(other)
    assert normalize_with_report(other).signals["fffd_remaining"] == 0
