"""Test ช้า: ไฟล์ XLSX จริง (eval/datasets/files/ ไม่อยู่ใน git — ไม่มีไฟล์ก็ข้าม) รันด้วย ``pytest -m slow``

ตัวเลขที่ assert มาจากต้นแบบที่ทดลองในแชทก่อนเขียนโค้ด (ไม่ได้มาจาก extractor นี้)
test นี้ไม่พิมพ์/ไม่เก็บข้อมูลรายแถวของ confirmed-cases (มีข้อมูลรายบุคคล)
"""

import time
from pathlib import Path

import pytest

from app.extraction.pipeline import extract_file
from app.models import BlockType, DocumentStatus, ErrorCode, UnitStatus

FILES = Path(__file__).parent.parent / "eval" / "datasets" / "files"
MUNI = FILES / "municipal_staffing_140568.xlsx"
TIS = FILES / "tis_mask_17042569.xlsx"
CASES = FILES / "confirmed-cases-since-280265.xlsx"

pytestmark = pytest.mark.slow

MUNI_HEADER = ["ลำดับที่", "ชื่อสายงาน", "ระดับตำแหน่ง", "จำนวนทั้งหมด(คน)", "จำนวนที่มีอยู่ปัจจุบัน(คน)", "หมายเหตุ"]
needs_muni = pytest.mark.skipif(not MUNI.exists(), reason="ไม่มี municipal_staffing_140568.xlsx")
needs_tis = pytest.mark.skipif(not TIS.exists(), reason="ไม่มี tis_mask_17042569.xlsx")
needs_cases = pytest.mark.skipif(not CASES.exists(), reason="ไม่มี confirmed-cases-since-280265.xlsx")


@pytest.fixture(scope="module")
def muni():
    d = extract_file(MUNI)
    assert d.error is None
    return d


@pytest.fixture(scope="module")
def tis():
    d = extract_file(TIS)
    assert d.error is None
    return d


@needs_muni
def test_municipal_columns_match_header(muni):
    tables = [b for u in muni.units for b in u.blocks if b.type == BlockType.TABLE]
    assert tables and all(b.table.rows[0] == MUNI_HEADER for b in tables)


@needs_muni
def test_municipal_counts(muni):
    assert sum(u.signals["xlsx:data_rows"] for u in muni.units) == 116
    assert sum(u.signals.get("xlsx:repeated_header_rows", 0) for u in muni.units) == 9
    assert all(u.position["range"].startswith("A") and u.position["range"].split(":")[1].startswith("F") for u in muni.units)
    assert muni.units[-1].position["range"].endswith("F171")  # ไม่ใช่ Z1000 ตาม dimension ที่ Excel ประกาศ


@needs_muni
def test_municipal_total_row_note_and_dash(muni):
    all_lines = [ln for u in muni.units for ln in u.text.split("\n")]
    totals = [ln for ln in all_lines if "ชื่อสายงาน: รวม" in ln]
    assert totals == ["ชื่อสายงาน: รวม | จำนวนทั้งหมด(คน): 157 | จำนวนที่มีอยู่ปัจจุบัน(คน): 136"]
    assert "หมายเหตุ: ข้อมูล ณ วันที่ 14 พ.ค.68" in all_lines
    lawyer = [ln for ln in all_lines if "ชื่อสายงาน: นิติกร" in ln]
    assert lawyer and lawyer[0].endswith("จำนวนที่มีอยู่ปัจจุบัน(คน): -")


@needs_muni
def test_municipal_every_unit_after_first_has_previous_labels_line(muni):
    first, rest = muni.units[0], muni.units[1:]
    assert rest
    assert not any(ln.startswith("[ป้ายก่อนหน้า") for ln in first.text.split("\n"))
    for u in rest:
        assert any(ln.startswith("[ป้ายก่อนหน้า: ") for ln in u.text.split("\n")), u.label
    assert all(not ln.startswith("## ") for u in muni.units for ln in u.text.split("\n")[-1:])


@needs_muni
def test_municipal_gate_and_status(muni):
    assert muni.status == DocumentStatus.DONE
    assert all(u.status == UnitStatus.OK and not u.quality.flags for u in muni.units)
    assert all(u.signals["xlsx:group_label_max_repeat"] >= 2 for u in muni.units)


@needs_tis
def test_tis_flat_table(tis):
    tables = [b for u in tis.units for b in u.blocks if b.type == BlockType.TABLE]
    assert all(len(b.table.rows[0]) == 17 for b in tables)
    assert sum(u.signals["xlsx:data_rows"] for u in tis.units) == 7
    assert tis.status == DocumentStatus.DONE


@needs_cases
def test_confirmed_cases_is_rejected_quickly():
    t0 = time.perf_counter()
    d = extract_file(CASES)
    elapsed = time.perf_counter() - t0
    assert d.status == DocumentStatus.FAILED and d.error.code == ErrorCode.TOO_LARGE
    assert not d.units
    assert elapsed < 5


@needs_muni
def test_municipal_context_lines_collapse_adjacent_duplicates(muni):
    ctx = [ln for u in muni.units[1:] for ln in u.text.split(chr(10)) if ln.startswith("[ป้ายก่อนหน้า: ")]
    assert len(ctx) == 3
    for ln in ctx:
        labels = ln[len("[ป้ายก่อนหน้า: "):-1].split(" → ")
        assert 1 <= len(labels) <= 3
        assert all(a != b for a, b in zip(labels, labels[1:]))
    assert "กองคลัง (04) → กองคลัง (04)" not in ctx[0]
