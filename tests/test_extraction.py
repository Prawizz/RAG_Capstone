"""PDF extractor + pipeline: ใช้ PDF เล็กที่สร้างใน test เอง"""

import hashlib

import pymupdf
import pytest

import app.extraction.pdf as pdf_mod
from app.extraction.pdf import join_lines
from app.extraction.pipeline import extract_file
from app.models import (
    DetectedType,
    DocumentStatus,
    ErrorCode,
    QualityFlag,
    UnitStatus,
)
from app.thai.normalizer import NORMALIZER_VERSION


def make_pdf(path, pages, **save_kw):
    """pages: list ของข้อความ ('' = หน้าว่าง)"""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 100), text, fontsize=14)
    doc.save(str(path), **save_kw)
    doc.close()
    return path


# ---------- join_lines ----------

@pytest.mark.parametrize(
    "lines,expected",
    [
        (["พนักงาน", "ประจำ"], "พนักงานประจำ"),  # ไทย+ไทย ไม่แทรก
        (["พนักงาน", "HR"], "พนักงาน HR"),  # ไทย+อังกฤษ แทรก
        (["HR", "พนักงาน"], "HR พนักงาน"),
        (["hello", "world"], "hello world"),  # อังกฤษ+อังกฤษ แทรก
        (["hello ", "world"], "hello world"),  # มีช่องว่างอยู่แล้ว ไม่แทรกเพิ่ม
        (["hello", " world"], "hello world"),
        (["พนักงาน ", "ประจำ"], "พนักงาน ประจำ"),
        (["a", "b", "c"], "a b c"),
        ([], ""),
        (["เดียว"], "เดียว"),
        (["ยึดมั่", "\x0eนในหลัก"], "ยึดมั่\x0eนในหลัก"),  # Cc ที่รอยต่อ ไม่แทรก (เก็บอักขระไว้)
        (["ผู้นำ\ufffd", "ในตลาด"], "ผู้นำ\ufffdในตลาด"),  # U+FFFD
        (["ผู้นำ", "\u200bในตลาด"], "ผู้นำ\u200bในตลาด"),  # Cf (zero-width space)
        (["hello\x0e", "\ufffdworld"], "hello\x0e \ufffdworld"),  # ข้าม junk แล้วเป็นอังกฤษ ยังแทรก
        (["พนักงาน\x0e ", "ประจำ"], "พนักงาน\x0e ประจำ"),  # ช่องว่างหลัง junk ยังนับ
        (["\x0e", "ประจำ"], "\x0eประจำ"),  # บรรทัดที่เป็น junk ล้วน
    ],
)
def test_join_lines(lines, expected):
    assert join_lines(lines) == expected


# ---------- ระดับไฟล์ ----------

def test_basic_pdf(tmp_path):
    f = make_pdf(tmp_path / "a.pdf", ["Hello first page", "Second page"])
    doc = extract_file(f)
    assert doc.status == DocumentStatus.DONE
    assert doc.detected_type == DetectedType.PDF
    assert doc.unit_count == 2 and doc.page_equivalent == 2
    assert doc.size_bytes == f.stat().st_size
    assert doc.content_hash == hashlib.sha256(f.read_bytes()).hexdigest()
    assert len(doc.file_id) == 32
    assert doc.normalizer_version == NORMALIZER_VERSION
    u = doc.units[0]
    assert (u.index, u.position, u.label, u.method.value) == (0, {"page": 1}, "หน้า 1", "text_layer")
    assert u.text == "Hello first page"
    assert u.text_hash == hashlib.sha256(u.text.encode("utf-8")).hexdigest()
    assert u.quality.score is None
    assert set(u.signals) >= {"char_count", "block_count", "image_coverage"}
    assert u.signals["char_count"] == len("Hello first page")
    assert u.blocks[0].font_size == 14.0 and u.blocks[0].bbox is not None
    assert u.blocks[0].bold is False


def test_file_id_passthrough(tmp_path):
    f = make_pdf(tmp_path / "a.pdf", ["x"])
    assert extract_file(f, file_id="abc").file_id == "abc"


def test_page_range(tmp_path):
    f = make_pdf(tmp_path / "a.pdf", ["p1", "p2", "p3", "p4"])
    doc = extract_file(f, pages=(2, 3))
    assert [u.position["page"] for u in doc.units] == [2, 3]
    assert [u.index for u in doc.units] == [0, 1]
    assert doc.page_equivalent == 4


def test_encrypted_pdf(tmp_path):
    f = make_pdf(
        tmp_path / "e.pdf", ["secret"],
        encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="user", owner_pw="owner",
    )
    doc = extract_file(f)
    assert doc.status == DocumentStatus.FAILED
    assert doc.error.code == ErrorCode.ENCRYPTED and doc.error.retriable is False
    assert doc.units == []


def test_owner_password_only_is_readable(tmp_path):
    f = make_pdf(
        tmp_path / "o.pdf", ["readable"],
        encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="", owner_pw="owner",
    )
    doc = extract_file(f)
    assert doc.error is None and doc.units[0].text == "readable"


def test_corrupt_pdf(tmp_path):
    f = tmp_path / "bad.pdf"
    f.write_bytes(b"%PDF-1.7\nthis is not a pdf at all\n")
    doc = extract_file(f)
    assert doc.status == DocumentStatus.FAILED
    assert doc.error.code == ErrorCode.CORRUPT
    assert doc.detected_type == DetectedType.PDF


def test_empty_page(tmp_path):
    doc = extract_file(make_pdf(tmp_path / "a.pdf", ["", "text"]))
    u = doc.units[0]
    assert u.status == UnitStatus.OK and u.text == ""
    assert QualityFlag.EMPTY in u.quality.flags and u.quality.score is None
    assert u.signals["char_count"] == 0 and u.signals["block_count"] == 0
    assert doc.status == DocumentStatus.DONE


def test_one_failing_page_does_not_fail_file(tmp_path, monkeypatch):
    f = make_pdf(tmp_path / "a.pdf", ["one", "two", "three"])
    real = pdf_mod._extract_page

    def flaky(page, index):
        if page.number == 1:
            raise RuntimeError("boom")
        return real(page, index)

    monkeypatch.setattr(pdf_mod, "_extract_page", flaky)
    doc = extract_file(f)
    assert [u.status for u in doc.units] == [UnitStatus.OK, UnitStatus.FAILED, UnitStatus.OK]
    assert "boom" in doc.units[1].reason and "หน้า 2" in doc.units[1].reason
    assert doc.status == DocumentStatus.DONE_WITH_GAPS
    assert doc.unit_summary["failed"] == 1


def test_page_label(tmp_path):
    f = tmp_path / "l.pdf"
    d = pymupdf.open()
    for i in range(3):
        d.new_page().insert_text((72, 100), f"p{i}")
    d.set_page_labels([{"startpage": 1, "prefix": "A-", "style": "D", "firstpagenum": 1}])
    d.save(str(f))
    d.close()
    doc = extract_file(f)
    assert "page_label" not in doc.units[0].position
    assert doc.units[1].position == {"page": 2, "page_label": "A-1"}
    assert doc.units[2].position["page_label"] == "A-2"


def test_outline_and_producer(tmp_path):
    f = tmp_path / "o.pdf"
    d = pymupdf.open()
    d.new_page().insert_text((72, 100), "x")
    d.set_toc([[1, "Intro", 1]])
    d.set_metadata({"producer": "UnitTest"})
    d.save(str(f))
    d.close()
    doc = extract_file(f)
    assert doc.has_outline is True and doc.producer == "UnitTest"
    plain = extract_file(make_pdf(tmp_path / "p.pdf", ["x"]))
    assert plain.has_outline is False


def test_image_coverage(tmp_path):
    f = tmp_path / "i.pdf"
    d = pymupdf.open()
    page = d.new_page(width=200, height=200)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
    pix.set_rect(pix.irect, (200, 0, 0))
    page.insert_image(pymupdf.Rect(0, 0, 100, 100), pixmap=pix)  # 1/4 ของหน้า
    d.save(str(f))
    d.close()
    assert extract_file(f).units[0].signals["image_coverage"] == pytest.approx(0.25, abs=0.01)


def test_unsupported_type_documents(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    doc = extract_file(f)
    assert doc.status == DocumentStatus.FAILED
    assert doc.error.code == ErrorCode.UNSUPPORTED_TYPE
    assert doc.detected_type is None


def test_recognised_but_not_implemented_type(tmp_path, monkeypatch):
    from tests.test_detect import ole_file

    f = ole_file(tmp_path, monkeypatch, [["Workbook"]])
    doc = extract_file(f)
    assert doc.detected_type == DetectedType.XLS
    assert doc.error.code == ErrorCode.UNSUPPORTED_TYPE
    assert doc.error.message == "ยังไม่รองรับใน v1"


# ---------- ต่อ normalizer เข้า pipeline ----------

def test_normalizer_results_flow_into_unit(tmp_path, monkeypatch):
    f = make_pdf(tmp_path / "n.pdf", ["x"])
    messy = [
        pdf_mod.RawBlock("ธุุรกิิจ ดําเนินงาน", (0, 0, 1, 1), 12.0, False),
        pdf_mod.RawBlock("#MARKER:PART_1", (0, 0, 1, 1), 12.0, False),  # block ว่างหลัง normalize -> ไม่สร้าง block
        pdf_mod.RawBlock("เรื่่�อง", (0, 0, 1, 1), 12.0, False),
    ]
    monkeypatch.setattr(pdf_mod, "raw_blocks", lambda page: messy)
    u = extract_file(f).units[0]
    assert [b.text for b in u.blocks] == ["ธุรกิจ ดำเนินงาน", "เรื่อง"]
    assert [b.order for b in u.blocks] == [0, 1]
    assert u.text == "ธุรกิจ ดำเนินงาน\nเรื่อง"
    assert u.text_hash == hashlib.sha256(u.text.encode("utf-8")).hexdigest()
    # ผลนับรวมจากทุก block รวม block ที่ถูกลบจนว่าง
    expect = {
        "norm_fix:dup_mark": 3, "norm_fix:sara_am": 1, "norm_fix:whitespace": 1,
        "norm_fix:marker": 1, "norm_fix:fffd_after_mark": 1,
    }
    assert {k: v for k, v in u.signals.items() if k.startswith("norm_fix:")} == {k: float(v) for k, v in expect.items()}
    assert u.quality.normalization_changes == sum(expect.values())
    assert all(isinstance(v, float) for v in u.signals.values())
    assert u.signals["char_count"] == len(u.text.strip())


def test_clean_text_has_no_norm_keys(tmp_path):
    u = extract_file(make_pdf(tmp_path / "c.pdf", ["Hello world"])).units[0]
    assert not [k for k in u.signals if k.startswith("norm_")]
    assert u.quality.normalization_changes == 0
