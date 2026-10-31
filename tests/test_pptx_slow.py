"""Test ช้า: PPTX จริง (eval/datasets/files/ ไม่อยู่ใน git — ไม่มีไฟล์ก็ข้าม) รันด้วย ``pytest -m slow``"""

from pathlib import Path

import pytest

from app.extraction.pipeline import extract_file
from app.extraction.quality import gate_key
from app.models import BlockType

FILES = Path(__file__).parent.parent / "eval" / "datasets" / "files"
TH = FILES / "official_letter_writing_th.pptx"
EN = FILES / "os_ch4_multithreaded_en.pptx"

pytestmark = pytest.mark.slow
needs_th = pytest.mark.skipif(not TH.exists(), reason="ไม่มี official_letter_writing_th.pptx")
needs_en = pytest.mark.skipif(not EN.exists(), reason="ไม่มี os_ch4_multithreaded_en.pptx")


@pytest.fixture(scope="module")
def th():
    d = extract_file(TH)
    assert d.error is None
    return d


@pytest.fixture(scope="module")
def en():
    d = extract_file(EN)
    assert d.error is None
    return d


def block_texts(doc, slide):
    return [b.text for b in doc.units[slide - 1].blocks]


@needs_th
def test_th_shape(th):
    assert th.unit_count == 35 and th.detected_type.value == "pptx"
    assert [u.position["slide"] for u in th.units] == list(range(1, 36))


@needs_th
def test_th_slide_20_reading_order(th):
    # ลำดับที่คาดคำนวณจากพิกัดในไฟล์ (ไม่ได้มาจากโค้ดเรา): กล่องที่ top ใกล้กันอยู่แถวเดียวกัน เรียงซ้ายไปขวา
    assert block_texts(th, 20) == [
        "คำขึ้นต้น", "คำลงท้าย",
        "กราบเรียน", "ขอแสดงความนับถืออย่างยิ่ง",
        "เรียน", "ขอแสดงความนับถือ",
        "ขอประทานกราบทูล", "ควรมิควรแล้วแต่จะโปรดเกล้าโปรดกระหม่อม",
        "กราบทูล", "ควรมิควรแล้วแต่จะโปรด",
        "นมัสการ", "ขอนมัสการด้วยความเคารพ",
    ]


@needs_th
def test_th_slide_25_reading_order(th):
    got = block_texts(th, 25)
    assert got[:7] == ["การใช้ “ตาม ตามที่”", "ตาม", "คำนาม", "นั้น", "ตามที่", "ประโยค", "นั้น"]
    assert got[7].startswith("ตาม/ตามที่ เรื่องที่เคยมีการติดต่อกันมา")


@needs_th
@pytest.mark.parametrize("n", [4, 5, 10])
def test_th_picture_only_slides_need_ocr(th, n):
    u = th.units[n - 1]
    assert u.status.value == "needs_ocr" and "scan_like" in {f.value for f in u.quality.flags}
    assert u.signals["image_coverage"] >= 0.25


@needs_en
@pytest.mark.parametrize("n", [5, 6, 7, 10, 14])
def test_en_diagram_slides_are_ok_with_image_content(en, n):
    u = en.units[n - 1]
    assert u.status.value == "ok" and gate_key(u) == "ok:image_content"
    if n != 7:  # สไลด์ 7 ไม่มีหัวเรื่อง (มีแต่ข้อความสั้นๆ ข้างรูป)
        assert u.blocks[0].type == BlockType.HEADING  # ค้นด้วยหัวเรื่องได้
    assert u.text.strip()


@needs_en
def test_en_slide_11_has_numbers_from_chart(en):
    text = en.units[10].text
    assert "3633.8" in text and "กราฟ" in text


@needs_en
def test_en_slide_15_chart_data(en):
    text = en.units[14].text
    assert "กราฟ: Time to sum 1,000,000 integers" in text and "1731.6667" in text


@needs_en
def test_en_text_rich_slides_are_never_low_text(en):
    for u in en.units:
        if u.signals["gate:letters"] >= 100:
            assert "low_text" not in {f.value for f in u.quality.flags}, u.label
