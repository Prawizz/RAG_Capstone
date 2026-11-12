"""ชนเพดานโทเคน -> ตัดลูป + flag ocr_incomplete และ OCR ซ้ำแบบแบ่งบน/ล่าง (backend ปลอม ไม่ต้องมี GPU)"""

import io
import json

import numpy as np
import pymupdf
import pytest
from PIL import Image, ImageDraw

from app.extraction import ocr as ocr_mod
from app.extraction.ocr import (
    PART_SEPARATOR,
    FakeBackend,
    OcrResult,
    find_row_gap,
    hits_ceiling,
    recognize_page_auto,
    result_parts,
    run_ocr,
)
from app.extraction.ocr_cache import OcrCache, make_key_fields
from app.extraction.pipeline import extract_file
from app.models import DetectedType, QualityFlag, UnitStatus

THAI_LONG = "ข้อความภาษาไทยสำหรับทดสอบการอ่านเอกสารที่ยาวพอสมควรเพื่อให้ผ่านด่านตรวจคุณภาพ " * 4
LOOP = "ประโยคที่โมเดลพิมพ์วนซ้ำไปเรื่อยๆ " * 40
CAP = 100  # เพดานโทเคนของ backend ปลอม


# ---------- ภาพทดสอบ ----------

def dashed_rows(im, y0, y1, x0, x1, gray=20):
    """แถบหมึกเป็นจุด 3x3 px ทุก 6 px แทนบรรทัดข้อความ: หมึกไม่ถึงครึ่งแถว (ค่ากลางของแถวเป็นสีพื้น) และไม่มีเส้นแนวตั้ง
    ยาวต่อเนื่อง (ไม่ถูกมองเป็นเส้นตาราง); ทุกแถวในช่วงมีหมึก"""
    d = ImageDraw.Draw(im)
    for y in range(y0, y1, 3):
        shift = 0 if (y - y0) % 6 == 0 else 3
        for x in range(x0 + shift, x1, 6):
            d.rectangle([x, y, min(x + 2, x1 - 1), min(y + 2, y1 - 1)], fill=(gray, gray, gray))


def gray_of(im):
    return np.asarray(im.convert("L"))


def lined_image(w, h, x_ranges=((0.05, 0.95),), pitch=40, line=24):
    """บรรทัดสูง ``line`` px ทุก ``pitch`` px (ช่องว่างระหว่างบรรทัด pitch - line px)"""
    im = Image.new("RGB", (w, h), (255, 255, 255))
    for y in range(20, h - 20, pitch):
        for a, b in x_ranges:
            dashed_rows(im, y, y + line, int(a * w), int(b * w))
    return im


def save(tmp_path, im, name="p.png"):
    f = tmp_path / name
    im.save(f)
    return f


def img_height(png):
    return Image.open(io.BytesIO(png)).height


# ---------- find_row_gap ----------

def test_row_gap_nearest_middle():
    im = Image.new("RGB", (600, 800), (255, 255, 255))
    dashed_rows(im, 100, 300, 30, 570)
    dashed_rows(im, 330, 450, 30, 570)
    dashed_rows(im, 460, 700, 30, 570)
    # ช่องว่าง: 300-330 (กลาง 315) และ 450-460 (กลาง 455) -> 455 ใกล้กลาง (400) กว่า
    assert find_row_gap(gray_of(im)) == 455


def test_row_gap_none_when_gap_too_small_or_outside_window():
    im = Image.new("RGB", (600, 800), (255, 255, 255))
    dashed_rows(im, 100, 400, 30, 570)
    dashed_rows(im, 403, 700, 30, 570)  # ช่อง 3 px < ROW_GAP_MIN_PX
    assert find_row_gap(gray_of(im)) is None
    im2 = Image.new("RGB", (600, 800), (255, 255, 255))
    dashed_rows(im2, 100, 700, 30, 570)  # ช่องว่างมีแต่ขอบบน/ล่าง (นอก window)
    assert find_row_gap(gray_of(im2)) is None


def test_thin_vertical_table_rules_do_not_block_gap():
    im = Image.new("RGB", (600, 800), (255, 255, 255))
    dashed_rows(im, 100, 380, 60, 540)
    dashed_rows(im, 420, 700, 60, 540)
    d = ImageDraw.Draw(im)
    for x in (40, 300, 560):  # เส้นตารางแนวตั้งกว้าง 2 px ยาวทั้งตาราง
        d.rectangle([x, 80, x + 1, 720], fill=(0, 0, 0))
    assert find_row_gap(gray_of(im)) == 400


def test_thick_vertical_stroke_still_counts_as_ink():
    im = Image.new("RGB", (600, 800), (255, 255, 255))
    dashed_rows(im, 100, 380, 60, 540)
    dashed_rows(im, 420, 700, 60, 540)
    ImageDraw.Draw(im).rectangle([300, 80, 311, 720], fill=(0, 0, 0))  # กว้าง 12 px = ขีดตัวอักษร/ภาพ ไม่ใช่เส้นตาราง
    assert find_row_gap(gray_of(im)) is None


def test_uniform_colour_band_is_not_ink():
    """แถบพื้นสีเรียบเต็มแถว (ไม่มีตัวอักษร) ตัดผ่านได้"""
    im = Image.new("RGB", (600, 800), (255, 255, 255))
    dashed_rows(im, 100, 380, 60, 540)
    dashed_rows(im, 420, 700, 60, 540)
    ImageDraw.Draw(im).rectangle([0, 380, 599, 419], fill=(0, 120, 60))
    assert find_row_gap(gray_of(im)) is not None


# ---------- แยกผลรายชิ้น ----------

def test_result_parts_uses_part_stats_and_falls_back_for_old_cache():
    res = OcrResult(text="ก" + PART_SEPARATOR + "ขค", backend="b", model="m", revision="r", seconds=1, tokens=7,
                    part_stats=[{"tokens": 3, "chars": 1}, {"tokens": 4, "chars": 2}])
    assert result_parts(res) == [("ก", 3), ("ขค", 4)]
    assert hits_ceiling(res, 4) == 1 and hits_ceiling(res, 3) == 2 and hits_ceiling(res, None) == 0
    old = OcrResult(text="ก" + PART_SEPARATOR + "ขค", backend="b", model="m", revision="r", seconds=1, tokens=7)
    assert result_parts(old) == [(old.text, 7)]
    assert hits_ceiling(old, 7) == 1  # ไม่รู้รายชิ้น: ใช้โทเคนรวม (ประมาณข้างมาก)
    bad = OcrResult(text="กข", backend="b", model="m", revision="r", seconds=1, tokens=7,
                    part_stats=[{"tokens": 3, "chars": 5}])
    assert result_parts(bad) == [("กข", 7)]


# ---------- ตัดลูป + flag (ไม่ resplit) ----------

def test_loop_in_left_half_is_trimmed_per_part_and_flagged(tmp_path):
    f = save(tmp_path, lined_image(1600, 800, x_ranges=((0.03, 0.45), (0.55, 0.97))))
    backend = FakeBackend([(THAI_LONG + " ซ้าย " + LOOP, CAP), (THAI_LONG + " ขวา", 10)], max_new_tokens=CAP)
    u = run_ocr(extract_file(f), f, backend, resplit=False).units[0]
    assert backend.calls == 2
    assert u.status == UnitStatus.OK and QualityFlag.OCR_INCOMPLETE in u.quality.flags
    assert u.signals["ocr:hit_max_tokens"] == 1.0
    assert u.signals["ocr:loop_trimmed_chars"] > 0
    assert u.text.count("ประโยคที่โมเดลพิมพ์วนซ้ำไปเรื่อยๆ") == 1  # เก็บรอบแรกไว้หนึ่งรอบ
    assert u.text.rstrip().endswith("ขวา")  # ครึ่งขวาที่อยู่หลังลูปยังอยู่ครบ


def test_hit_ceiling_without_loop_is_flagged_but_text_kept(tmp_path):
    f = save(tmp_path, lined_image(600, 800))
    backend = FakeBackend([(THAI_LONG + " ถูกตัดท้าย", CAP)], max_new_tokens=CAP)
    u = run_ocr(extract_file(f), f, backend, resplit=False).units[0]
    assert QualityFlag.OCR_INCOMPLETE in u.quality.flags and u.status == UnitStatus.OK
    assert u.signals["ocr:hit_max_tokens"] == 1.0 and u.signals["ocr:loop_trimmed_chars"] == 0.0
    assert "ถูกตัดท้าย" in u.text


def test_clean_result_has_no_flag(tmp_path):
    f = save(tmp_path, lined_image(600, 800))
    u = run_ocr(extract_file(f), f, FakeBackend([(THAI_LONG, 10)], max_new_tokens=CAP)).units[0]
    assert QualityFlag.OCR_INCOMPLETE not in u.quality.flags
    assert u.signals["ocr:hit_max_tokens"] == 0.0 and u.signals["ocr:loop_trimmed_chars"] == 0.0
    assert "ocr:resplit" not in u.signals


def test_backend_without_cap_still_trims_loops(tmp_path):
    f = save(tmp_path, lined_image(600, 800))
    u = run_ocr(extract_file(f), f, FakeBackend(THAI_LONG + " " + LOOP)).units[0]
    assert "ocr:hit_max_tokens" not in u.signals  # ไม่มีเพดาน = ไม่รู้ ไม่ใส่
    assert u.signals["ocr:loop_trimmed_chars"] > 0 and QualityFlag.OCR_INCOMPLETE in u.quality.flags


# ---------- resplit ----------

def test_single_page_resplit_top_then_bottom(tmp_path):
    f = save(tmp_path, lined_image(600, 800))
    backend = FakeBackend([(THAI_LONG + LOOP, CAP), (THAI_LONG + " บน", 10), (THAI_LONG + " ล่าง", 10)],
                          max_new_tokens=CAP)
    u = run_ocr(extract_file(f), f, backend).units[0]
    assert backend.calls == 3
    assert img_height(backend.images[1]) + img_height(backend.images[2]) == 800
    assert u.text.index("บน") < u.text.index("ล่าง") and "พิมพ์วน" not in u.text
    assert u.signals["ocr:resplit"] == 1.0 and u.signals["ocr:pieces"] == 2.0
    assert u.signals["ocr:first_pass_tokens"] == float(CAP) and u.signals["ocr:tokens"] == 20.0
    assert u.signals["ocr:hit_max_tokens"] == 0.0 and u.signals["ocr:resplit_no_gap"] == 0.0
    assert u.signals["ocr:resplit_rotated"] == 0.0
    assert QualityFlag.OCR_INCOMPLETE not in u.quality.flags


def test_wide_page_resplit_four_pieces_in_reading_order(tmp_path):
    f = save(tmp_path, lined_image(1600, 800, x_ranges=((0.03, 0.45), (0.55, 0.97))))
    labels = ["ซ้ายบน", "ซ้ายล่าง", "ขวาบน", "ขวาล่าง"]
    first = [(THAI_LONG + LOOP, CAP), (THAI_LONG, 10)]
    backend = FakeBackend(lambda n: first[n] if n < 2 else (THAI_LONG + " " + labels[n - 2], 10), max_new_tokens=CAP)
    u = run_ocr(extract_file(f), f, backend).units[0]
    assert backend.calls == 6 and u.signals["ocr:pieces"] == 4.0
    pos = [u.text.index(x) for x in labels]
    assert pos == sorted(pos)
    left_top, left_bottom = (Image.open(io.BytesIO(b)) for b in backend.images[2:4])
    assert left_top.width == left_bottom.width and left_top.height + left_bottom.height == 800


def test_piece_still_at_cap_is_split_once_more_then_flagged(tmp_path):
    f = save(tmp_path, lined_image(600, 1600))
    seq = [
        (LOOP, CAP),  # รอบแรก (ทั้งหน้า)
        (LOOP, CAP),  # บน: ยังชน -> แบ่งอีกระดับ
        (THAI_LONG + " บนบน " + LOOP, CAP),  # บน-บน: ชนอีก (ระดับสุดท้าย) -> ตัดลูป + flag
        (THAI_LONG + " บนล่าง", 10),
        (THAI_LONG + " ชิ้นล่างสุด", 10),
    ]
    backend = FakeBackend(lambda n: seq[n], max_new_tokens=CAP)
    u = run_ocr(extract_file(f), f, backend).units[0]
    assert backend.calls == 5
    assert u.signals["ocr:pieces"] == 3.0 and u.signals["ocr:hit_max_tokens"] == 1.0
    assert u.signals["ocr:loop_trimmed_chars"] > 0 and QualityFlag.OCR_INCOMPLETE in u.quality.flags
    pos = [u.text.index(x) for x in ("บนบน", "บนล่าง", "ชิ้นล่างสุด")]
    assert pos == sorted(pos)


def test_no_gap_reuses_first_pass_and_flags(tmp_path):
    im = Image.new("RGB", (600, 800), (255, 255, 255))
    dashed_rows(im, 20, 780, 30, 570)  # ไม่มีช่องว่างแนวนอน
    f = save(tmp_path, im)
    backend = FakeBackend([(THAI_LONG + " " + LOOP, CAP)], max_new_tokens=CAP)
    u = run_ocr(extract_file(f), f, backend).units[0]
    assert backend.calls == 1  # ภาพเดียวกับรอบแรก + รู้ผลรายชิ้น -> ใช้ผลเดิม ไม่ OCR ซ้ำ
    assert u.signals["ocr:resplit_no_gap"] == 1.0 and u.signals["ocr:pieces"] == 1.0
    assert QualityFlag.OCR_INCOMPLETE in u.quality.flags and u.signals["ocr:hit_max_tokens"] == 1.0


def test_resplit_pieces_are_cached_and_first_pass_key_unchanged(tmp_path):
    f = save(tmp_path, lined_image(600, 800))
    doc = extract_file(f)
    seq = [(THAI_LONG + LOOP, CAP), (THAI_LONG + " บน", 10), (THAI_LONG + " ล่าง", 10)]
    b1 = FakeBackend(seq, max_new_tokens=CAP)  # list (ไม่ใช่ lambda): repr เหมือนกัน -> cache key เดียวกัน
    u1 = run_ocr(doc, f, b1).units[0]
    b2 = FakeBackend(seq, max_new_tokens=CAP)
    u2 = run_ocr(doc, f, b2).units[0]
    assert b2.calls == 0 and u2.text == u1.text and u2.signals["ocr:cache_hit"] == 1.0
    store = OcrCache()
    fields = [json.loads(p.read_text(encoding="utf-8"))["fields"] for p in store.dir.glob("*.json")]
    mine = [x for x in fields if x["content_hash"] == doc.content_hash]
    assert sorted(x["split_method"] for x in mine) == ["none", "none+rows", "none+rows"]
    assert all("region" not in x for x in mine if x["split_method"] == "none")


def test_key_fields_without_region_have_no_region_key():
    base = dict(content_hash="h", page=1, backend="b", revision="r", dpi=200, split_method="gutter", split_x=0.5,
                params={})
    assert "region" not in make_key_fields(**base)
    assert make_key_fields(**base, region=(0, 0.123456, 1, 1))["region"] == [0, 0.1235, 1, 1]


def test_resplit_off_keeps_first_pass(tmp_path):
    f = save(tmp_path, lined_image(600, 800))
    backend = FakeBackend([(THAI_LONG + " " + LOOP, CAP)], max_new_tokens=CAP)
    u = run_ocr(extract_file(f), f, backend, resplit=False).units[0]
    assert backend.calls == 1 and "ocr:resplit" not in u.signals


# ---------- ข้อความหมุน (PDF) ----------

def lined_pdf(path, rotate):
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    if rotate:
        for x in range(60, 560, 30):  # บรรทัดแนวตั้ง (หมุน 90°) เรียงเป็นคอลัมน์
            page.insert_text((x, 700), "Rotated table cell text " * 2, fontsize=10, rotate=90)
    else:
        for y in list(range(80, 360, 20)) + list(range(440, 740, 20)):
            page.insert_text((40, y), "Horizontal body text line for the test " * 2, fontsize=10)
    doc.save(str(path))
    return path


def sha(f):
    import hashlib

    return hashlib.sha256(f.read_bytes()).hexdigest()


def size_of(png):
    return Image.open(io.BytesIO(png)).size


def test_flat_text_is_not_rotated(tmp_path):
    f = lined_pdf(tmp_path / "r.pdf", False)
    backend = FakeBackend(lambda n: (LOOP, CAP) if n == 0 else ("ok", 10), max_new_tokens=CAP)
    _res, _page, _hit, info = recognize_page_auto(backend, f, DetectedType.PDF, sha(f), 1)
    assert info.pieces == 2 and info.rotated == 0
    w0, h0 = size_of(backend.images[0])
    assert all(size_of(b)[0] == w0 for b in backend.images[1:])  # ไม่หมุน: ชิ้นบน/ล่างกว้างเท่าหน้า
    assert sum(size_of(b)[1] for b in backend.images[1:]) == h0


def test_rotated_region_is_turned_upright_then_split(tmp_path):
    """ข้อความหมุน 90° (อ่านจากล่างขึ้นบน) -> หมุนภาพตามเข็ม 90° แล้วแบ่งบน/ล่างของภาพตั้งตรง
    = แบ่งซ้าย/ขวาของหน้า โดยชิ้นบน (ส่งก่อน) คือซ้ายของหน้า"""
    f = lined_pdf(tmp_path / "r.pdf", True)
    backend = FakeBackend(lambda n: (LOOP, CAP) if n == 0 else (f"ok{n}", 10), max_new_tokens=CAP)
    res, page, _hit, info = recognize_page_auto(backend, f, DetectedType.PDF, sha(f), 1)
    assert info.rotated == 1 and info.no_gap == 0 and info.pieces == 2
    w0, h0 = size_of(backend.images[0])  # รอบแรก: หน้าตั้ง
    (wa, ha), (wb, hb) = size_of(backend.images[1]), size_of(backend.images[2])
    assert w0 < h0 and wa == wb == h0 and ha + hb == w0  # ภาพที่ส่งเป็นแนวนอน (หมุนแล้ว) และต่อกันได้ครบ
    (a, b) = info.regions
    assert a[0] == 0 and a[2] == b[0] and b[2] == 1 and a[1] == b[1] == 0 and a[3] == b[3] == 1
    assert res.text.index("ok1") < res.text.index("ok2")


def test_rotated_pieces_have_their_own_cache_keys(tmp_path):
    f = lined_pdf(tmp_path / "r.pdf", True)
    seq = [(LOOP, CAP), ("บน", 10), ("ล่าง", 10)]
    recognize_page_auto(FakeBackend(seq, max_new_tokens=CAP), f, DetectedType.PDF, sha(f), 1)
    again = FakeBackend(seq, max_new_tokens=CAP)
    _res, _page, hit, _info = recognize_page_auto(again, f, DetectedType.PDF, sha(f), 1)
    assert hit and again.calls == 0
    fields = [json.loads(p.read_text(encoding="utf-8"))["fields"] for p in OcrCache().dir.glob("*.json")]
    mine = [x for x in fields if x["content_hash"] == sha(f)]
    assert sorted(x.get("rotation", 0) for x in mine) == [0, 90, 90]
    assert all("rotation" not in x for x in mine if x["split_method"] == "none")  # key รอบแรกไม่เปลี่ยน


def test_rotated_region_without_gap_is_ocred_upright_not_reused(tmp_path):
    f = lined_pdf(tmp_path / "r.pdf", True)
    backend = FakeBackend(lambda n: (LOOP, CAP) if n == 0 else ("ตั้งตรง", 10), max_new_tokens=CAP)
    orig = ocr_mod.find_row_gap
    ocr_mod.find_row_gap = lambda gray: None
    try:
        res, _page, _hit, info = recognize_page_auto(backend, f, DetectedType.PDF, sha(f), 1)
    finally:
        ocr_mod.find_row_gap = orig
    assert info.rotated == 1 and info.no_gap == 1 and info.pieces == 1
    assert backend.calls == 2 and res.text == "ตั้งตรง"  # ไม่ใช้ผลรอบแรก (ภาพตะแคง)
    w, h = size_of(backend.images[1])
    assert w > h


def test_text_orientation_gives_share_and_direction(tmp_path):
    flat = lined_pdf(tmp_path / "a.pdf", False)
    rot = lined_pdf(tmp_path / "b.pdf", True)
    assert ocr_mod.text_orientation(flat, 1, (0, 0, 1, 1)) == (0.0, 0)
    assert ocr_mod.text_orientation(rot, 1, (0, 0, 1, 1)) == (1.0, 90)  # อ่านล่างขึ้นบน -> หมุนตามเข็ม 90°
    assert ocr_mod.rotated_text_share(rot, 1, (0, 0, 1, 1)) == 1.0
    assert ocr_mod.text_orientation(flat, 1, (0, 0.99, 1, 1)) is None  # ไม่มีบรรทัดในช่วงนี้

    down = tmp_path / "c.pdf"  # อ่านบนลงล่าง -> หมุนทวนเข็ม (= ตามเข็ม 270°)
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    for x in range(60, 560, 30):
        page.insert_text((x, 100), "Rotated table cell text " * 2, fontsize=10, rotate=-90)
    doc.save(str(down))
    assert ocr_mod.text_orientation(down, 1, (0, 0, 1, 1)) == (1.0, 270)


def test_rotate_piece_turns_clockwise():
    im = Image.new("RGB", (40, 20), (255, 255, 255))
    im.putpixel((0, 0), (0, 0, 0))  # มุมซ้ายบน
    out = ocr_mod.rotate_piece(ocr_mod.Piece(ocr_mod._png(im), (0, 0, 1, 1), 0), 90)
    r = Image.open(io.BytesIO(out.png)).convert("RGB")
    assert r.size == (20, 40) and out.rotation == 90
    assert r.getpixel((19, 0)) == (0, 0, 0)  # หมุนตามเข็ม: ซ้ายบน -> ขวาบน


@pytest.mark.parametrize("rotation, top, bottom", [
    (0, (0.2, 0.0, 0.6, 0.25), (0.2, 0.25, 0.6, 1.0)),
    (90, (0.2, 0.0, 0.3, 1.0), (0.3, 0.0, 0.6, 1.0)),  # บนของภาพตั้งตรง = ซ้ายของหน้า
    (270, (0.5, 0.0, 0.6, 1.0), (0.2, 0.0, 0.5, 1.0)),  # บน = ขวาของหน้า
    (180, (0.2, 0.75, 0.6, 1.0), (0.2, 0.0, 0.6, 0.75)),  # บน = ล่างของหน้า
])
def test_sub_region_maps_upright_rows_back_to_page(rotation, top, bottom):
    region = (0.2, 0.0, 0.6, 1.0)
    got = [ocr_mod._sub_region(region, rotation, 0.0, 0.25), ocr_mod._sub_region(region, rotation, 0.25, 1.0)]
    assert [tuple(round(v, 6) for v in r) for r in got] == [top, bottom]


def test_resplit_result_failing_gate_is_failed_and_keeps_incomplete_flag(tmp_path):
    """แบบหน้า 169: resplit แล้วยังชนเพดาน ตัดลูปแล้วข้อความไม่ผ่านด่านตรวจ -> failed (ไม่ใช่ needs_ocr)"""
    f = save(tmp_path, lined_image(600, 800))
    bad = "สวัสดี���� ทดสอบ"
    seq = [(LOOP, CAP), (bad + " " + LOOP, CAP), (bad, 10)]
    u = run_ocr(extract_file(f), f, FakeBackend(seq, max_new_tokens=CAP)).units[0]
    assert u.status == UnitStatus.FAILED
    assert u.reason.startswith("OCR แล้วไม่ผ่านด่านตรวจ: replacement_char")
    assert "ocr_incomplete" not in u.reason.split("—")[0]  # flag ในเหตุผล = ของด่านตรวจเท่านั้น
    assert QualityFlag.OCR_INCOMPLETE in u.quality.flags and u.signals["ocr:resplit"] == 1.0


# ---------- batch script ----------

def test_batch_page_runs_resplit_and_notes_it(tmp_path):
    import hashlib
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "ocr_batch_resplit", Path(__file__).resolve().parent.parent / "scripts" / "ocr_batch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    f = lined_pdf(tmp_path / "b.pdf", False)
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    backend = FakeBackend([(LOOP, CAP), ("บน", 10), ("ล่าง", 10)], max_new_tokens=CAP)
    seen = []
    rec = mod.ocr_one_page(backend, f, h, 1, mod.CheckedCache(tmp_path / "c"),
                           on_piece=lambda piece, res, hit: seen.append((piece.depth, res.tokens, hit)))
    assert rec["status"] == mod.OK and rec["tokens"] == 20
    assert rec["resplit"] == {"first_pass_tokens": CAP, "pieces": 2, "hit_pieces": 0, "no_gap": 0, "rotated": 0}
    assert rec["error"].startswith("resplit:") and "เพดาน" not in rec["error"]
    assert seen == [(1, 10, False), (1, 10, False)]
