"""run_ocr ด้วย backend ปลอม (ไม่ต้องมี GPU)"""

import hashlib

import pymupdf
import pytest

from app.extraction import ocr as ocr_mod
from app.extraction.detect import detect
from app.extraction.ocr import FakeBackend, OcrError, OcrNotAllowed, render_page_images, run_ocr, text_layer_agreement
from app.extraction.ocr_markdown import parse_ocr_markdown, parse_table
from app.extraction.pipeline import extract_file
from app.models import DetectedType, DocumentStatus, ErrorCode, ExtractionMethod, UnitStatus

THAI_LONG = "ข้อความภาษาไทยสำหรับทดสอบการอ่านเอกสารที่ยาวพอสมควรเพื่อให้ผ่านด่านตรวจคุณภาพ " * 4


def make_pdf(path, pages):
    """pages: list ของข้อความ ('' = หน้าเป็นภาพล้วน ไม่มีชั้นข้อความ)"""
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), False)
    pix.set_rect(pix.irect, (200, 200, 200))
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 100), text, fontname="helv", fontsize=12)
        else:
            page.insert_image(page.rect, pixmap=pix)  # ภาพเต็มหน้า -> scan_like
    doc.save(str(path))
    return path


@pytest.fixture
def scan_pdf(tmp_path):
    return make_pdf(tmp_path / "scan.pdf", ["Hello text layer page " * 10, "", ""])


def png_bytes():
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 30), False)
    pix.set_rect(pix.irect, (10, 20, 30))
    return pix.tobytes("png"), pix.tobytes("jpeg")


def test_only_needs_ocr_units_are_touched(scan_pdf):
    doc = extract_file(scan_pdf)
    assert [u.status for u in doc.units] == [UnitStatus.OK, UnitStatus.NEEDS_OCR, UnitStatus.NEEDS_OCR]
    out = run_ocr(doc, scan_pdf, FakeBackend(THAI_LONG))
    assert out.units[0] is doc.units[0]  # ไม่ใช่ needs_ocr -> วัตถุเดิม
    assert [u.index for u in out.units] == [0, 1, 2]
    assert [u.method for u in out.units] == [ExtractionMethod.TEXT_LAYER, ExtractionMethod.OCR, ExtractionMethod.OCR]
    assert doc.units[1].status == UnitStatus.NEEDS_OCR  # Document เดิมไม่ถูกแก้
    assert out.status == DocumentStatus.DONE


def test_extract_file_does_not_call_ocr(scan_pdf, monkeypatch):
    monkeypatch.setattr(ocr_mod, "run_ocr", lambda *a, **k: pytest.fail("extract_file เรียก OCR"))
    extract_file(scan_pdf)


def test_ocr_unit_goes_through_normalizer_and_gate(scan_pdf):
    doc = extract_file(scan_pdf)
    out = run_ocr(doc, scan_pdf, FakeBackend(THAI_LONG + " ทํา"))  # ํา (นิคหิต+สระอา) ต้องถูก normalize เป็น ำ
    u = out.units[1]
    assert u.status == UnitStatus.OK and u.method == ExtractionMethod.OCR
    assert "ทำ" in u.text and "ํา" not in u.text
    assert u.signals["gate:letters"] >= 100 and u.signals["ocr:seconds"] > 0
    assert u.text_hash == hashlib.sha256(u.text.encode()).hexdigest()
    assert u.reason == "OCR: fake · fake-model@fake-rev"
    assert "image_coverage" not in u.signals


def test_ocr_result_failing_gate_stays_needs_ocr(scan_pdf):
    doc = extract_file(scan_pdf)
    broken = "สวัสดี���� ทดสอบ"
    out = run_ocr(doc, scan_pdf, FakeBackend(broken))
    u = out.units[1]
    assert u.status == UnitStatus.NEEDS_OCR and u.method == ExtractionMethod.OCR
    assert "OCR: fake" in u.reason and "พบจุดพัง" in u.reason


def test_ocr_little_text_is_low_text_not_scan_like(scan_pdf):
    out = run_ocr(extract_file(scan_pdf), scan_pdf, FakeBackend("สั้นๆ"))
    assert out.units[1].status == UnitStatus.OK
    assert [f.value for f in out.units[1].quality.flags] == ["low_text"]


def test_page_failure_does_not_fail_file(scan_pdf):
    doc = extract_file(scan_pdf)
    backend = FakeBackend([RuntimeError("CUDA out of memory"), THAI_LONG])
    out = run_ocr(doc, scan_pdf, backend)
    bad, good = out.units[1], out.units[2]
    assert bad.status == UnitStatus.FAILED and "CUDA out of memory" in bad.reason and bad.method == ExtractionMethod.OCR
    assert good.status == UnitStatus.OK
    assert out.status == DocumentStatus.DONE_WITH_GAPS


def test_offsite_backend_rejected_unless_allowed(scan_pdf, monkeypatch):
    monkeypatch.delenv("OCR_ALLOW_OFFSITE", raising=False)
    doc = extract_file(scan_pdf)
    backend = FakeBackend(THAI_LONG, sends_data_offsite=True)
    with pytest.raises(OcrNotAllowed):
        run_ocr(doc, scan_pdf, backend)
    assert backend.calls == 0
    assert run_ocr(doc, scan_pdf, backend, allow_offsite=True).units[1].status == UnitStatus.OK
    monkeypatch.setenv("OCR_ALLOW_OFFSITE", "1")
    assert run_ocr(doc, scan_pdf, backend).units[1].status == UnitStatus.OK


def test_offsite_rejected_even_with_nothing_to_ocr(tmp_path):
    f = make_pdf(tmp_path / "t.pdf", ["Hello text layer page " * 10])
    with pytest.raises(OcrNotAllowed):
        run_ocr(extract_file(f), f, FakeBackend(sends_data_offsite=True), allow_offsite=False)


def test_default_config_is_local(monkeypatch):
    monkeypatch.delenv("OCR_BACKEND", raising=False)
    assert ocr_mod.DEFAULT_BACKEND == "typhoon-local"
    from app.extraction.ocr_typhoon import TyphoonLocalBackend

    assert TyphoonLocalBackend.sends_data_offsite is False


def test_unknown_backend_name():
    with pytest.raises(ocr_mod.OcrUnavailable):
        ocr_mod.get_backend("nope")


def test_wrong_source_file_rejected(scan_pdf, tmp_path):
    doc = extract_file(scan_pdf)
    other = make_pdf(tmp_path / "other.pdf", ["", ""])
    with pytest.raises(OcrError):
        run_ocr(doc, other, FakeBackend())


def test_nothing_to_do_returns_same_document(tmp_path):
    f = make_pdf(tmp_path / "t.pdf", ["Hello text layer page " * 10])
    doc = extract_file(f)
    backend = FakeBackend()
    assert run_ocr(doc, f, backend) is doc and backend.calls == 0


def test_backend_receives_png_of_the_right_page(scan_pdf):
    backend = FakeBackend(THAI_LONG)
    run_ocr(extract_file(scan_pdf), scan_pdf, backend)
    assert len(backend.images) == 2 and all(b.startswith(b"\x89PNG") for b in backend.images)


def test_render_dpi_is_one_constant(scan_pdf, monkeypatch):
    backend = FakeBackend(THAI_LONG)
    doc = extract_file(scan_pdf)
    run_ocr(doc, scan_pdf, backend)
    w200 = pymupdf.Pixmap(backend.images[0]).width
    monkeypatch.setattr(ocr_mod, "OCR_RENDER_DPI", 100)
    b2 = FakeBackend(THAI_LONG)
    run_ocr(doc, scan_pdf, b2)
    assert pymupdf.Pixmap(b2.images[0]).width < w200


# ---------- ไฟล์ภาพ ----------

@pytest.mark.parametrize("which,kind", [(0, DetectedType.PNG), (1, DetectedType.JPG)])
def test_image_file_is_one_page_needs_ocr(tmp_path, which, kind):
    f = tmp_path / "a.bin"  # นามสกุลไม่เกี่ยว ตรวจจาก magic bytes
    f.write_bytes(png_bytes()[which])
    assert detect(f) == kind
    doc = extract_file(f)
    assert doc.detected_type == kind and doc.error is None and doc.page_equivalent == 1
    (u,) = doc.units
    assert u.type.value == "page" and u.position == {"page": 1} and u.status == UnitStatus.NEEDS_OCR
    assert doc.status == DocumentStatus.NEEDS_OCR
    out = run_ocr(doc, f, FakeBackend(THAI_LONG))
    assert out.units[0].status == UnitStatus.OK and out.units[0].method == ExtractionMethod.OCR
    assert "ocr:text_layer_recall" not in out.units[0].signals  # ภาพไม่มีชั้นข้อความ
    assert out.status == DocumentStatus.DONE


def test_corrupt_image_is_corrupt(tmp_path):
    f = tmp_path / "bad.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"junk" * 10)
    d = extract_file(f)
    assert d.error.code == ErrorCode.CORRUPT


def test_unsupported_type_with_needs_ocr_is_error(scan_pdf):
    doc = extract_file(scan_pdf).model_copy(update={"detected_type": DetectedType.PPTX})
    with pytest.raises(OcrError):
        run_ocr(doc, scan_pdf, FakeBackend())


# ---------- markdown -> blocks ----------

NL = chr(10)
TABLE_HTML = (
    '<table><thead><tr><th rowspan="2">รายการ</th><th colspan="2">ปี</th></tr>'
    "<tr><th>2567</th><th>2568</th></tr></thead>"
    "<tbody><tr><td>รายได้ &amp; อื่นๆ</td><td>1,000</td><td>2,000</td></tr></tbody></table>"
)
FULL_MD = NL.join([
    "# งบการเงินรวม", "", "## หมายเหตุ 1", "ข้อความย่อหน้าแรก &amp; ต่อ", "บรรทัดสอง", "",
    "* รายการหนึ่ง", "* รายการสอง", "1. ข้อหนึ่ง", "", TABLE_HTML, "",
    "<figure>", "แผนภูมิแท่งแสดงรายได้", "</figure>", "", "ปิดท้าย", "<page_number>๑๒</page_number>",
])


def test_markdown_to_blocks_full(scan_pdf):
    u = run_ocr(extract_file(scan_pdf), scan_pdf, FakeBackend(FULL_MD)).units[1]
    kinds = [(b.type.value, b.level) for b in u.blocks]
    assert kinds == [("heading", 1), ("heading", 2), ("paragraph", None), ("list", None), ("table", None),
                     ("image", None), ("paragraph", None)]
    assert [b.order for b in u.blocks] == list(range(7))
    assert u.blocks[2].text == "ข้อความย่อหน้าแรก & ต่อ" + NL + "บรรทัดสอง"  # entity ถูกถอด
    assert u.blocks[3].text == NL.join(["* รายการหนึ่ง", "* รายการสอง", "1. ข้อหนึ่ง"])
    assert u.blocks[0].text == "งบการเงินรวม"  # ไม่มี # ในข้อความ
    assert "page_number" not in u.text and "<" not in u.text and "&amp;" not in u.text
    assert u.signals["ocr:page_number"] == 12.0  # เลขไทยอ่านเป็นตัวเลข
    assert u.signals["ocr:figure_count"] == 1.0


def test_figure_is_block_but_not_in_unit_text(scan_pdf):
    u = run_ocr(extract_file(scan_pdf), scan_pdf, FakeBackend(FULL_MD)).units[1]
    img = [b for b in u.blocks if b.type.value == "image"]
    assert len(img) == 1 and img[0].text == "แผนภูมิแท่งแสดงรายได้"
    assert "แผนภูมิแท่ง" not in u.text


def test_table_block_rows_header_and_text(scan_pdf):
    u = run_ocr(extract_file(scan_pdf), scan_pdf, FakeBackend(FULL_MD)).units[1]
    (b,) = [b for b in u.blocks if b.type.value == "table"]
    assert b.table.rows == [["รายการ", "ปี", ""], ["", "2567", "2568"], ["รายได้ & อื่นๆ", "1,000", "2,000"]]
    assert b.table.header_rows == 2
    assert {(m.row, m.col, m.rowspan, m.colspan) for m in b.table.merged_cells} == {(0, 0, 2, 1), (0, 1, 1, 2)}
    assert b.text == NL.join(["รายการ | ปี", "2567 | 2568", "รายได้ & อื่นๆ | 1,000 | 2,000"])
    assert b.text in u.text


def test_table_without_thead_uses_leading_th_rows_and_rowspan_shifts_columns():
    t = parse_table("<table><tr><th>ก</th><th>ข</th></tr><tr><td rowspan=2>x</td><td>1</td></tr><tr><td>2</td></tr></table>")
    assert t.header_rows == 1
    assert t.rows == [["ก", "ข"], ["x", "1"], ["", "2"]]  # แถวสามเลื่อนไปคอลัมน์สอง เพราะ x คลุมลงมา
    t2 = parse_table("<table><tr><td>a</td></tr></table>")
    assert t2.header_rows == 0 and t2.rows == [["a"]]


def test_table_unclosed_and_br():
    t = parse_ocr_markdown("<table><tr><td>บรรทัด<br>สอง</td><td>b</td>").blocks[0].table
    assert t.rows == [["บรรทัด สอง", "b"]]


def test_page_number_variants():
    p = parse_ocr_markdown("ก <page_number>iv</page_number> ข <page_number>041</page_number>")
    assert p.page_numbers == [41] and [b.text for b in p.blocks] == ["ก  ข"]
    assert parse_ocr_markdown("<page_number>14</page_number>").blocks == []


def test_figure_does_not_count_toward_gate(scan_pdf):
    md = "<figure>" + THAI_LONG + "</figure>"  # ตัวอักษรจากคำบรรยายภาพไม่นับเป็นข้อความของหน้า
    u = run_ocr(extract_file(scan_pdf), scan_pdf, FakeBackend(md)).units[1]
    assert u.text == "" and [f.value for f in u.quality.flags] == ["empty"] and u.status == UnitStatus.OK


# ---------- agreement สองทาง ----------

def test_recall_and_precision():
    layer = "งบการเงินรวม สินทรัพย์ หนี้สิน"  # 4 คำ: งบการเงิน รวม สินทรัพย์ หนี้สิน
    assert text_layer_agreement(layer, "งบการเงินรวม สินทรัพย์ หนี้สิน") == (1.0, 1.0)
    r, p = text_layer_agreement(layer, "งบการเงินรวม สินทรัพย์ หนี้สิน และกำไรสุทธิประจำปี")
    assert r == 1.0 and p < 0.7  # OCR เติมคำเอง -> precision ตก recall ยังเต็ม
    r, p = text_layer_agreement(layer, "<p>งบการเงินรวม</p> สินทรัพย์")
    assert r == pytest.approx(0.75, abs=0.01) and p == 1.0  # OCR ตกหล่น -> recall ตก precision เต็ม
    assert text_layer_agreement(layer, "ฟหกดเ่าสว") == (0.0, 0.0)
    assert text_layer_agreement(layer, "") == (0.0, 0.0)


def test_agreement_none_without_text_layer():
    assert text_layer_agreement("", "อะไรก็ได้") is None
    assert text_layer_agreement("  " + NL + " ", "อะไรก็ได้") is None
    assert text_layer_agreement("- , .", "x") is None


def test_agreement_counts_multiset():
    r, p = text_layer_agreement("abc abc abc", "abc")
    assert r == pytest.approx(1 / 3) and p == 1.0


def test_agreement_signals_only_when_text_layer_exists(tmp_path):
    f = make_pdf(tmp_path / "t.pdf", ["", ""])
    out = run_ocr(extract_file(f), f, FakeBackend(THAI_LONG))
    assert all(k not in u.signals for u in out.units for k in ("ocr:text_layer_recall", "ocr:text_layer_precision"))


def test_agreement_signals_with_text_layer(tmp_path):
    f = make_pdf(tmp_path / "t.pdf", ["alpha beta gamma delta " * 8, ""])
    doc = extract_file(f)
    forced = [u.model_copy(update={"status": UnitStatus.NEEDS_OCR, "reason": "ทดสอบ"}) if u.index == 0 else u for u in doc.units]
    doc = doc.model_copy(update={"units": forced})
    out = run_ocr(doc, f, FakeBackend("alpha beta gamma " * 8 + THAI_LONG))
    s = out.units[0].signals
    assert s["ocr:text_layer_recall"] == pytest.approx(0.75, abs=0.01)
    assert s["ocr:text_layer_precision"] < 0.5  # ภาษาไทยที่ OCR เติมมา
    assert "ocr:text_layer_recall" not in out.units[1].signals


# ---------- หน้ากว้าง ----------

def _gray():
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 50), False)
    pix.set_rect(pix.irect, (210, 210, 210))
    return pix


def wide_pdf(path, w, h):
    doc = pymupdf.open()
    page = doc.new_page(width=w, height=h)
    page.insert_image(page.rect, pixmap=_gray())
    doc.save(str(path))
    return path


def test_wide_page_is_split_left_then_right(tmp_path):
    f = wide_pdf(tmp_path / "w.pdf", 800, 400)
    doc = extract_file(f)
    backend = FakeBackend([THAI_LONG + " ซ้าย", THAI_LONG + " ขวา"])
    u = run_ocr(doc, f, backend).units[0]
    assert backend.calls == 2
    pix = [pymupdf.Pixmap(b) for b in backend.images]
    assert pix[0].height == pix[1].height
    assert abs(pix[0].width - pix[1].width) <= 1
    assert u.text.index("ซ้าย") < u.text.index("ขวา")
    assert u.signals["ocr:seconds"] == pytest.approx(0.02)  # รวมสองครึ่ง


def test_page_at_or_below_ratio_not_split(tmp_path):
    f = wide_pdf(tmp_path / "n.pdf", 650, 500)  # 1.3 พอดี ไม่เกิน
    assert len(render_page_images(f, DetectedType.PDF, 1)) == 1
    g = wide_pdf(tmp_path / "m.pdf", 660, 500)  # 1.32
    assert len(render_page_images(g, DetectedType.PDF, 1)) == 2
    assert ocr_mod.WIDE_PAGE_RATIO == 1.3


def test_portrait_page_not_split(scan_pdf):
    assert len(render_page_images(scan_pdf, DetectedType.PDF, 2)) == 1


def test_wide_image_file_is_split_too(tmp_path):
    from PIL import Image

    f = tmp_path / "wide.png"
    Image.new("RGB", (900, 300), (200, 200, 200)).save(f)
    assert len(render_page_images(f, DetectedType.PNG, 1)) == 2


# ---------- EXIF ----------

def test_exif_orientation_applied_before_ocr(tmp_path):
    from PIL import Image

    im = Image.new("RGB", (60, 20), (120, 120, 120))
    exif = Image.Exif()
    exif[0x0112] = 6  # ภาพจริงต้องหมุน 90° -> สูงกว่ากว้าง
    f = tmp_path / "photo.jpg"
    im.save(f, "JPEG", exif=exif)
    doc = extract_file(f)
    assert doc.units[0].signals["image:width"] == 20 and doc.units[0].signals["image:height"] == 60
    backend = FakeBackend(THAI_LONG)
    run_ocr(doc, f, backend)
    assert len(backend.images) == 1  # ถ้าไม่หมุน 60x20 กว้างเกินเกณฑ์และจะถูกตัดครึ่งผิดๆ
    pix = pymupdf.Pixmap(backend.images[0])
    assert (pix.width, pix.height) == (20, 60)


def test_image_without_exif_unchanged(tmp_path):
    from PIL import Image

    f = tmp_path / "p.png"
    Image.new("RGB", (30, 50)).save(f)
    backend = FakeBackend(THAI_LONG)
    run_ocr(extract_file(f), f, backend)
    pix = pymupdf.Pixmap(backend.images[0])
    assert (pix.width, pix.height) == (30, 50)
