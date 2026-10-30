"""Thai normalizer v1

คู่ทดสอบอยู่ที่ eval/datasets/thai_normalization.jsonl
- expected เป็น regression test ของกฎ เขียนด้วยมือจากการไล่กฎ ไม่ได้มาจากการรัน normalizer (CLAUDE.md ข้อ 6)
- truth (คู่ที่ source ไม่ใช่ synthetic) คือข้อความที่ถูกตามที่ตาเห็นใน PDF คนเติมเอง ยังว่าง (null) = ข้ามการเทียบ
"""

import json
from pathlib import Path

import pymupdf
import pytest

from app.extraction.pdf import raw_blocks
from eval.truth_check import (
    EVIDENCE,
    EXACT,
    FAIL,
    FAIL_MESSAGE,
    NEEDS_REPAIR,
    NO_TRUTH,
    classify,
    classify_case,
    evidence_match,
)
from app.thai.normalizer import (
    NORMALIZER_VERSION,
    RULE_NAMES,
    SIGNAL_NAMES,
    normalize,
    normalize_with_report,
)

ROOT = Path(__file__).parent.parent
CASES_PATH = ROOT / "eval" / "datasets" / "thai_normalization.jsonl"
# แยกบรรทัดด้วย "\n" เท่านั้น (splitlines จะตัดที่ \x85 / \u2028 ที่อยู่ในข้อมูลด้วย)
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
        assert {"input", "expected", "rule", "source"} <= c.keys()
        assert "verified" not in c
        if c["source"] == "synthetic":
            assert "truth" not in c
        else:
            assert "truth" in c  # null จนกว่าคนจะเติม
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


# ---------- เทียบกับ truth ที่คนพิมพ์จากภาพ (ข้ามคู่ที่ truth เป็น null) ----------

TRUTH_CASES = [c for c in CASES if c.get("truth")]


def test_truth_field_counts():
    with_field = [c for c in CASES if "truth" in c]
    assert len(with_field) == len([c for c in CASES if c["source"] != "synthetic"])
    # ตอนนี้ truth ยังว่างทุกคู่ (คนจะเติมเอง); เติมแล้ว test ด้านล่างจะเริ่มทำงาน
    assert all(c["truth"] is None or isinstance(c["truth"], str) for c in with_field)


@pytest.mark.parametrize("case", TRUTH_CASES, ids=[case_id(c) for c in TRUTH_CASES])
def test_against_truth(case):
    status, out = classify_case(case)
    assert status != FAIL, (
        f"{FAIL_MESSAGE}\n  source: {case['source']}\n  input: {case['input']!r}\n"
        f"  out:   {out!r}\n  truth: {case['truth']!r}"
    )
    assert status in (EXACT, EVIDENCE, NEEDS_REPAIR)  # needs_repair = งาน T04 ไม่ fail


def test_no_truth_is_reported_as_such():
    assert classify_case({"input": "ก", "truth": None})[0] == NO_TRUTH
    assert classify_case({"input": "ก", "truth": ""})[0] == NO_TRUTH
    assert classify_case({"input": "ก"})[0] == NO_TRUTH


# ---------- ทดสอบ matcher ด้วยคู่สังเคราะห์ ----------

def run(inp: str, truth: str) -> str:
    return classify_case({"input": inp, "truth": truth})[0]


@pytest.mark.parametrize(
    "inp,truth",
    [
        ("ธุุรกิิจ", "ธุรกิจ"),  # แก้ด้วยกฎ ได้ตรงกับที่ตาเห็น
        ("ปีี ๒๕๖๘", "ปี 2568"),  # truth ถูก normalize ก่อนเทียบ (เลขไทย)
    ],
)
def test_matcher_exact(inp, truth):
    assert run(inp, truth) == EXACT


@pytest.mark.parametrize(
    "inp,truth",
    [
        ("สร้\x1cงคว\x1cม", "สร้างความ"),  # \x1c = า (PTT หน้า 43)
        ("หลักเกณ\x92์", "หลักเกณฑ์"),  # \x92 = ฑ (KBANK หน้า 9)
    ],
)
def test_matcher_evidence(inp, truth):
    assert run(inp, truth) == EVIDENCE


@pytest.mark.parametrize(
    "inp,truth",
    [
        ("เน่Çอหาประกาศ", "เนื้อหาประกาศ"),  # ละตินติดไทย (KBANK)
        ("แกmผ่้ถ่อหุ้น", "แก่ผู้ถือหุ้น"),  # ละติน + วรรณยุกต์ซ้อน
        ("ก\x01ข", "กข"),  # FFFD แต่ truth ไม่มีตัวให้แทน -> ไม่ใช่ evidence แต่ยังมีร่องรอย
    ],
)
def test_matcher_needs_repair(inp, truth):
    assert run(inp, truth) == NEEDS_REPAIR


@pytest.mark.parametrize(
    "inp,truth",
    [
        ("บริษัทปตท", "บริษัท ปตท.ไทย"),  # ผิดแต่ดูสะอาด
        ("ประธานกรรมการ", "ประธานกรรมกร"),
    ],
)
def test_matcher_fail(inp, truth):
    assert run(inp, truth) == FAIL


def test_matcher_ignores_whitespace_and_newlines_on_both_sides():
    assert run("ค่าธรรมเนียม\nบัตรเครดิต", "ค่าธรรมเนียม บัตรเครดิต") == EXACT
    assert run("ค่าธรรมเนียมบัตรเครดิต", " ค่าธรรม เนียม\n\nบัตร เครดิต ") == EXACT
    assert run("สร้\x1cง ความ", "สร้า ง\nความ") == EVIDENCE  # evidence ก็ไม่สนช่องว่าง
    assert run("บริษัทปตท", "บริษัท\nปตท.ไทย") == FAIL  # ต่างที่ตัวอักษรจริง ยัง fail


def test_evidence_matcher_is_exact_not_fuzzy():
    assert evidence_match("ก\ufffdค", "กขค")
    assert not evidence_match("ก\ufffdค", "กขขค")  # FFFD แทนได้ตัวเดียวพอดี
    assert not evidence_match("ก\ufffdค", "กค")
    assert not evidence_match("ก\ufffdค", "กขง")  # ตัวอื่นต้องตรงทุกตัว
    assert not evidence_match("กขค", "กขค")  # ไม่มี FFFD ไม่ใช่ evidence
    assert evidence_match("\ufffd\ufffd", "ขค")
    assert classify("ก\ufffdค", "กขค") == EVIDENCE
