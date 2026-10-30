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


@pytest.mark.xfail(
    strict=True,
    reason="รอ T03: mock_corpus.pdf มีเครื่องหมายซ้ำ 386 จุด ต้อง normalize ก่อนเทียบ",
)
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
    assert doc.status != DocumentStatus.FAILED
    assert doc.unit_count == pages
