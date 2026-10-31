"""Test ช้า: ใช้ไฟล์จริง รันเฉพาะเมื่อสั่ง ``pytest -m slow``"""

import difflib
import re
from pathlib import Path

import pymupdf
import pytest

from app.extraction.pipeline import extract_file
from app.models import DocumentStatus

ROOT = Path(__file__).parent.parent
DATASETS = ROOT / "eval" / "datasets"
ONE_REPORTS = ["E_ONE_REPORT_CPALLT.PDF", "ONEREPORTKBANKT.PDF", "ONEREPORTPTTT.PDF"]

pytestmark = pytest.mark.slow


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def test_mock_corpus_matches_corpus_txt():
    # ground truth = corpus.txt (ไฟล์ต้นทางที่คนสร้าง PDF) ไม่ได้มาจาก extractor/normalizer ของเรา
    doc = extract_file(ROOT / "mock_corpus.pdf")
    assert doc.error is None
    extracted = _squash("\n".join(u.text for u in doc.units))
    truth = _squash((ROOT / "corpus.txt").read_text(encoding="utf-8"))
    # marker "--- เอกสารที่ N ---" ใน corpus.txt มีอยู่ใน PDF ด้วย จึงไม่ต้องตัดออกก่อนเทียบ
    ratio = difflib.SequenceMatcher(None, truth, extracted, autojunk=False).ratio()
    print(f"\nmock_corpus.pdf vs corpus.txt similarity = {ratio:.4f}")
    assert ratio >= 0.99


@pytest.mark.parametrize("name", ONE_REPORTS)
def test_one_report(name):
    path = DATASETS / name
    with pymupdf.open(str(path)) as d:
        pages = d.page_count
    doc = extract_file(path)
    assert doc.error is None
    # KBANK ฟอนต์พังทั้งไฟล์ ทุกหน้า needs_ocr: ไฟล์รอ OCR ไม่ใช่พัง
    expected = DocumentStatus.NEEDS_OCR if name == "ONEREPORTKBANKT.PDF" else None
    if expected:
        assert doc.status == expected
    else:
        assert doc.status != DocumentStatus.FAILED
    assert doc.unit_count == pages


# ---------- ด่านตรวจคุณภาพบน PDF จริง ----------

@pytest.fixture(scope="module")
def gated():
    return {n: extract_file(DATASETS / n) for n in ONE_REPORTS}


def test_gate_cpall_clean_pages_are_ok(gated):
    units = gated["E_ONE_REPORT_CPALLT.PDF"].units
    for page in (298, 302):  # คำพิมพ์ผิดในต้นฉบับ 1 จุด ไม่ใช่ฟอนต์พัง
        assert units[page - 1].status.value == "ok", units[page - 1].reason


def test_gate_cpall_scans_go_to_ocr(gated):
    from app.extraction.quality import gate_summary

    assert gate_summary(gated["E_ONE_REPORT_CPALLT.PDF"].units)["needs_ocr:scan_like"] >= 150


def test_gate_ptt_page_43_needs_ocr(gated):
    u = gated["ONEREPORTPTTT.PDF"].units[42]
    assert u.status.value == "needs_ocr" and "พบจุดพัง" in u.reason


def test_gate_kbank_almost_all_needs_ocr(gated):
    units = gated["ONEREPORTKBANKT.PDF"].units
    share = sum(u.status.value == "needs_ocr" for u in units) / len(units)
    assert share >= 0.9


# ---------- ล็อกผลด่านตรวจ PDF (หลังเปลี่ยนเป็นนับ letters: CPALL 241 / PTT 1, 460, 461 ไม่ใช่ low_text, PTT 462 ไม่ใช่ scan_like) ----------

LOCKED_GATE = {
    "E_ONE_REPORT_CPALLT.PDF": {"ok": 368, "ok:low_text": 22, "needs_ocr:scan_like": 156},
    "ONEREPORTKBANKT.PDF": {"needs_ocr:broken": 171, "needs_ocr:document_font_broken": 9},
    "ONEREPORTPTTT.PDF": {"ok": 458, "ok:low_text": 2, "needs_ocr:scan_like": 2, "needs_ocr:broken": 2},
}


@pytest.mark.parametrize("name", ONE_REPORTS)
def test_pdf_gate_results_are_locked(gated, name):
    from collections import Counter

    from app.extraction.quality import gate_key

    assert dict(Counter(gate_key(u) for u in gated[name].units)) == LOCKED_GATE[name]


def test_pdf_gate_matches_human_labels(gated):
    import json

    path = DATASETS / "ocr_pages.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    rows = [r for r in rows if r.get("label") in ("ok", "broken")]
    assert len(rows) == 36
    for r in rows:
        status = gated[r["file"]].units[r["page"] - 1].status.value
        assert (status == "needs_ocr") == (r["label"] == "broken"), (r["file"], r["page"], r["label"], status)
