"""Test ช้า: ไฟล์รุ่นเก่าจริงผ่าน LibreOffice จริง (eval/datasets/files/legacy/ ไม่อยู่ใน git)
ข้ามถ้าไม่มีไฟล์หรือไม่มี LibreOffice; รันด้วย ``pytest -m slow``

ตัวเลขที่ assert มาจากการทดลองบน Linux ก่อนเขียนโค้ด (ไม่ได้มาจาก extractor นี้)
"""

import re
import tempfile
from pathlib import Path

import pytest

from app.extraction.convert import find_soffice
from app.extraction.pipeline import extract_file
from app.models import BlockType, DetectedType, DocumentStatus, ErrorCode, ExtractionMethod

LEGACY = Path(__file__).parent.parent / "eval" / "datasets" / "files" / "legacy"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not LEGACY.exists(), reason="ไม่มี eval/datasets/files/legacy/"),
    pytest.mark.skipif(find_soffice() is None, reason="ไม่มี LibreOffice"),
]

THAI = re.compile("[ก-๛]")
LATIN1 = re.compile("[À-ÿ]")


def need(name):
    f = LEGACY / name
    if not f.exists():
        pytest.skip(f"ไม่มี {name}")
    return f


def total_text(d):
    return "".join(u.text for u in d.units)


@pytest.fixture(scope="module")
def results():
    """แปลงแต่ละไฟล์ครั้งเดียว และเช็กว่าไม่มีขยะค้าง (ทั้งในโฟลเดอร์ทดสอบและใน temp)"""
    before = sorted(p.name for p in LEGACY.rglob("*"))
    temp_before = {p.name for p in Path(tempfile.gettempdir()).glob("rag-convert-*")}
    out = {}
    for name in ["thesis_manual_th.doc", "municipal_staffing_140568.xls", "official_letter_writing_th.ppt",
                 "real_msword_20210318.doc", "real_mspowerpoint_esg_2558.ppt", "real_msexcel_form55.xls"]:
        if (LEGACY / name).exists():
            out[name] = extract_file(LEGACY / name)
    assert sorted(p.name for p in LEGACY.rglob("*")) == before
    assert {p.name for p in Path(tempfile.gettempdir()).glob("rag-convert-*")} == temp_before
    return out


def get(results, name):
    if name not in results:
        pytest.skip(f"ไม่มี {name}")
    d = results[name]
    assert d.error is None, d.error
    return d


def test_converted_documents_are_marked(results):
    for name, d in results.items():
        assert d.error is None and d.converted_from == d.detected_type.value
        assert d.detected_type in (DetectedType.DOC, DetectedType.XLS, DetectedType.PPT)
        assert all(u.method == ExtractionMethod.CONVERTED for u in d.units)


def test_thesis_doc_roundtrip(results):
    d = get(results, "thesis_manual_th.doc")
    assert len(d.units) == 60 and len(total_text(d)) == 65362
    headings = {b.text for u in d.units for b in u.blocks if b.type == BlockType.HEADING}
    from tests.test_docx_slow import WORD_CONFIRMED_NUMBERING

    for h in WORD_CONFIRMED_NUMBERING:  # เลขอัตโนมัติ 6 จุดที่คนยืนยันกับ Word ยังอยู่หลังแปลง
        assert h in headings


def test_municipal_xls_roundtrip(results):
    d = get(results, "municipal_staffing_140568.xls")
    assert len(d.units) == 4 and len(total_text(d)) == 16037


def test_official_letter_ppt_roundtrip(results):
    d = get(results, "official_letter_writing_th.ppt")
    assert len(d.units) == 35 and len(total_text(d)) == 8080


def test_real_word_doc(results):
    d = get(results, "real_msword_20210318.doc")
    # บรรทัดบริบท "[แถวแรกของตาราง: ...]" ที่เพิ่มตอนแบ่งตารางไม่ใช่ข้อความในไฟล์ จึงหักออกก่อนเทียบกับตัวเลขเดิม
    ctx = [b.text for u in d.units for b in u.blocks if b.text.startswith("[แถวแรกของตาราง")]
    text = total_text(d)
    plain_len = len(text) - sum(len(c) + 1 for c in ctx)  # +1 = ตัวขึ้นบรรทัดคั่นใน unit.text
    plain_thai = len(THAI.findall(text)) - sum(len(THAI.findall(c)) for c in ctx)
    assert d.status == DocumentStatus.DONE
    assert plain_len == 50620 and plain_thai == 38423
    assert text.count(chr(0xFFFD)) == 0 and not LATIN1.findall(text)
    assert max(len(u.text) for u in d.units) <= 4000 + 200  # ข้อ B1: ตารางบัญชีโครงการถูกแบ่งแล้ว (บริบทไม่นับในเพดาน)



def test_real_powerpoint_ppt(results):
    d = get(results, "real_mspowerpoint_esg_2558.ppt")
    text = total_text(d)
    assert d.status == DocumentStatus.DONE and len(d.units) == 34
    assert len(THAI.findall(text)) == 7248
    assert text.count("�") == 0 and not LATIN1.findall(text)


def test_real_excel_xls_converts_and_thai_is_clean(results):
    d = get(results, "real_msexcel_form55.xls")  # ไม่ล็อกจำนวน unit
    text = total_text(d)
    assert text.count("�") == 0 and not LATIN1.findall(text)
    assert len(THAI.findall(text)) > 20000


def test_real_excel_instruction_sheet_keeps_all_thai_cell_text(results):
    """อักษรไทยของชีต "คำชี้แจง" ต้องเท่ากับอักษรไทยในเซลล์จริง (นับอิสระด้วย openpyxl จากไฟล์ที่ LibreOffice แปลงไว้)
    ก่อนแก้ B2 ตัวเลขสูงกว่านี้เพราะแถว "1 | ลำดับที่ | ..." ถูกเข้าใจผิดเป็นหัวตารางแล้วชื่อคอลัมน์ซ้ำทุกแถว"""
    import openpyxl

    ref = LEGACY / "converted_by_libreoffice" / "real_msexcel_form55.xlsx"
    if not ref.exists():
        pytest.skip("ไม่มีไฟล์เทียบ")
    d = get(results, "real_msexcel_form55.xls")
    ws = openpyxl.load_workbook(ref, data_only=True).worksheets[0]
    raw = sum(len(THAI.findall(str(c.value))) for row in ws.iter_rows() for c in row if c.value is not None)
    ours = sum(len(THAI.findall(u.text)) for u in d.units if u.position["sheet"] == ws.title)
    assert abs(ours - raw) <= 5


def test_real_excel_instruction_sheet_has_no_guessed_header_or_page_numbers(results):
    d = get(results, "real_msexcel_form55.xls")
    first = d.units[0]
    assert first.position["sheet"] == "คำชี้แจง"
    assert first.text.split("\n")[1].startswith("A: 1 | B: ลำดับที่ | C: ")
    assert "## -" not in first.text and "-2-" not in first.text
    assert not any(ln.startswith("หมายเหตุ: ") and len(ln) < 4 for ln in first.text.split("\n"))


def test_conversion_matches_libreoffice_converted_on_linux(results):
    conv = LEGACY / "converted_by_libreoffice"
    for legacy, modern in [("real_msword_20210318.doc", "real_msword_20210318.docx"),
                           ("real_msexcel_form55.xls", "real_msexcel_form55.xlsx")]:
        if not (conv / modern).exists() or legacy not in results:
            pytest.skip("ไม่มีไฟล์เทียบ")
        direct = extract_file(conv / modern)
        ours = results[legacy]
        assert len(direct.units) == len(ours.units)
        assert len(THAI.findall(total_text(direct))) == len(THAI.findall(total_text(ours)))
        assert [u.text for u in direct.units] == [u.text for u in ours.units]


def test_nonexistent_encryption_not_flagged(results):
    assert all(d.error is None or d.error.code != ErrorCode.ENCRYPTED for d in results.values())


def test_real_word_doc_page_equivalent_is_45(results):
    assert get(results, "real_msword_20210318.doc").page_equivalent == 45


def test_real_word_split_table_later_parts_have_first_row_context():
    ref = LEGACY / "converted_by_libreoffice" / "real_msword_20210318.docx"
    if not ref.exists():
        pytest.skip("ไม่มีไฟล์เทียบ")
    d = extract_file(ref)
    ctx = [(u.index, b.text) for u in d.units for b in u.blocks if b.text.startswith("[แถวแรกของตาราง")]
    assert ctx and all(t.startswith("[แถวแรกของตาราง: ยุทธศาสตร์ / แนวทาง | จำนวนโครงการ") for _, t in ctx)
    assert max(len(u.text) for u in d.units) <= 4000 + 200  # บริบทไม่นับในเพดาน แต่ต้องไม่ทำให้ unit บวมผิดปกติ
