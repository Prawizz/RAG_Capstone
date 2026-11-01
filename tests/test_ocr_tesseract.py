"""backend Tesseract: ตัวแปลงผล TSV / การตั้งค่า (ไม่ต้องติดตั้ง Tesseract) + test จริง (slow, ข้ามถ้าไม่มี)"""

import pytest

from app.extraction import ocr_tesseract as T
from app.extraction.ocr import OcrUnavailable, get_backend

HEAD = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"


def row(level, b, p, l, w, x, y, wd, h, conf, text):
    return "\t".join(map(str, [level, 1, b, p, l, w, x, y, wd, h, conf, text]))


TSV = "\n".join([
    HEAD,
    row(1, 0, 0, 0, 0, 0, 0, 500, 300, -1, ""),
    row(4, 1, 1, 1, 0, 10, 10, 200, 20, -1, ""),
    row(5, 1, 1, 1, 1, 10, 10, 60, 20, 96.5, "ธนาคาร"),
    row(5, 1, 1, 1, 2, 75, 12, 80, 18, 91.0, "กสิกรไทย"),
    row(5, 1, 1, 1, 3, 160, 10, 50, 20, 88.0, "2568"),
    row(5, 1, 1, 2, 1, 10, 40, 40, 20, 70.0, "Net"),
    row(5, 1, 1, 2, 2, 55, 40, 40, 20, 80.0, "Zero"),
    row(5, 1, 2, 1, 1, 10, 100, 90, 25, 60.0, "ย่อหน้าใหม่"),
    row(5, 1, 2, 1, 2, 110, 100, 10, 25, -1, "   "),  # ว่าง ข้าม
])


def test_parse_tsv_lines_bbox_conf():
    lines = T.parse_tsv(TSV)
    assert [l["text"] for l in lines] == ["ธนาคารกสิกรไทย 2568", "Net Zero", "ย่อหน้าใหม่"]
    assert lines[0]["bbox"] == [10, 10, 210, 30]
    assert lines[0]["conf"] == pytest.approx(91.8, abs=0.1)
    assert lines[1]["bbox"] == [10, 40, 95, 60]


def test_text_has_blank_line_between_paragraphs_not_between_lines():
    assert T.lines_to_text(T.parse_tsv(TSV)) == "ธนาคารกสิกรไทย 2568\nNet Zero\n\nย่อหน้าใหม่"


def test_join_words_thai_thai_joined_others_spaced():
    assert T.join_words(["ธนาคาร", "กสิกร", "ไทย"]) == "ธนาคารกสิกรไทย"
    assert T.join_words(["ปี", "2568", "ของ", "PTT", "กลุ่ม"]) == "ปี 2568 ของ PTT กลุ่ม"
    assert T.join_words(["abc", "def"]) == "abc def"


def test_parse_tsv_ignores_garbage_rows():
    assert T.parse_tsv(HEAD + "\nnot a row\n5\tx") == []
    assert T.parse_tsv("") == []


def test_unavailable_without_binary(monkeypatch):
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(T.shutil, "which", lambda _: None)
    monkeypatch.setattr(T, "DEFAULT_LOCATIONS", ())
    with pytest.raises(OcrUnavailable):
        T.TesseractLocalBackend()


def test_unavailable_without_language_files(monkeypatch, tmp_path):
    exe = tmp_path / "tesseract.exe"
    exe.write_text("x")
    (tmp_path / "tessdata").mkdir()
    (tmp_path / "tessdata" / "eng.traineddata").write_bytes(b"e")  # ไม่มี tha
    monkeypatch.setenv("TESSERACT_CMD", str(exe))
    monkeypatch.delenv("OCR_TESSDATA_DIR", raising=False)
    with pytest.raises(OcrUnavailable) as e:
        T.TesseractLocalBackend()
    assert "tha.traineddata" in str(e.value)


def fake_install(monkeypatch, tmp_path, tha=b"tha-best"):
    exe = tmp_path / "tesseract.exe"
    exe.write_text("x")
    d = tmp_path / "tessdata"
    d.mkdir(exist_ok=True)
    (d / "tha.traineddata").write_bytes(tha)
    (d / "eng.traineddata").write_bytes(b"eng-best")
    monkeypatch.setenv("TESSERACT_CMD", str(exe))
    monkeypatch.delenv("OCR_TESSDATA_DIR", raising=False)
    calls = []

    def fake_run(cmd, timeout):
        calls.append(cmd)
        return (0, "tesseract v5.4.0.20240606\n", "") if "--version" in cmd else (0, TSV, "")

    monkeypatch.setattr(T, "_run", fake_run)
    return calls


def test_backend_with_fake_run(monkeypatch, tmp_path):
    from PIL import Image
    import io

    calls = fake_install(monkeypatch, tmp_path)
    b = T.TesseractLocalBackend()
    assert b.name == "tesseract-local" and b.sends_data_offsite is False and b.model == "tesseract-5.4.0.20240606"
    buf = io.BytesIO()
    Image.new("RGB", (500, 300), "white").save(buf, "PNG")
    r = b.recognize(buf.getvalue())
    assert r.text.startswith("ธนาคารกสิกรไทย 2568") and r.extra["size"] == [500, 300] and len(r.extra["lines"]) == 3
    cmd = calls[-1]
    assert cmd[2] == "stdout" and "-l" in cmd and cmd[cmd.index("-l") + 1] == "tha+eng" and "tessedit_create_tsv=1" in cmd
    assert "--tessdata-dir" in cmd and "--dpi" in cmd


def test_revision_changes_with_traineddata(monkeypatch, tmp_path):
    fake_install(monkeypatch, tmp_path, tha=b"tha-best")
    r1 = T.TesseractLocalBackend().revision
    fake_install(monkeypatch, tmp_path, tha=b"tha-fast")
    assert T.TesseractLocalBackend().revision != r1


def test_nonzero_exit_raises(monkeypatch, tmp_path):
    fake_install(monkeypatch, tmp_path)
    b = T.TesseractLocalBackend()
    monkeypatch.setattr(T, "_run", lambda cmd, timeout: (1, "", "Error opening data file"))
    with pytest.raises(RuntimeError):
        b.recognize(b"\x89PNG\r\n\x1a\n")


def test_registered_in_get_backend(monkeypatch, tmp_path):
    fake_install(monkeypatch, tmp_path)
    assert get_backend("tesseract-local").name == "tesseract-local"


@pytest.mark.slow
def test_real_tesseract_reads_thai(tmp_path):
    if T.find_tesseract() is None:
        pytest.skip("ไม่ได้ติดตั้ง Tesseract")
    import pymupdf

    try:
        b = T.TesseractLocalBackend()
    except OcrUnavailable as e:
        pytest.skip(str(e))
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((20, 60), "Net Zero 2050 Tesseract", fontsize=24)
    png = page.get_pixmap(dpi=200).tobytes("png")
    assert "Zero" in b.recognize(png).text


# ---------- แยกแถบข้างสีพื้น (side band) ----------

def band_image(width=1000, height=600, band=200, tint=(212, 233, 217)):
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, band - 1, height], fill=tint)
    for y in range(40, height, 60):  # ตัวหนังสือ (ink) ทั้งสองฝั่ง — median ตามแนวตั้งต้องไม่หลงไปตามหมึก
        d.rectangle([20, y, max(band - 30, 21), y + 14], fill=(0, 90, 40))
        d.rectangle([band + 40, y, width - 40, y + 14], fill=(30, 30, 30))
    return im


def test_find_side_band_edge_at_color_step():
    assert T.find_side_band_edge(band_image(band=200)) == 200
    assert T.find_side_band_edge(band_image(band=330)) == 330


def test_no_edge_on_white_page_or_weak_tint_or_outside_window():
    from PIL import Image

    assert T.find_side_band_edge(Image.new("RGB", (1000, 600), "white")) is None
    assert T.find_side_band_edge(band_image(tint=(250, 250, 250))) is None  # สีจางเกินเกณฑ์
    assert T.find_side_band_edge(band_image(band=30)) is None  # แคบกว่าช่วงค้นหา (< 5%)
    assert T.find_side_band_edge(band_image(band=600)) is None  # กว้างเกิน 40%


def test_gradient_without_step_is_not_a_band():
    from PIL import Image

    im = Image.new("RGB", (1000, 400), "white")
    px = im.load()
    for x in range(1000):
        v = int(255 - 60 * x / 1000)  # ไล่สีทีละน้อย ไม่มีขอบฉับพลัน
        for y in range(400):
            px[x, y] = (v, 255, v)
    assert T.find_side_band_edge(im) is None


def test_crossing_lines_need_margin_on_both_sides():
    lines = [{"bbox": [10, 0, 600, 20]}, {"bbox": [10, 0, 210, 20]}, {"bbox": [300, 0, 700, 20]}, {"bbox": [190, 0, 400, 20]}]
    assert [ln["bbox"] for ln in T.crossing_lines(lines, 200)] == [[10, 0, 600, 20]]  # 190..400 ล้ำซ้ายแค่ 10 px < margin


def png_of(im):
    import io

    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def side_band_fake(monkeypatch, tmp_path, crossing):
    """คืน TSV ตามความกว้างของภาพที่ส่งให้ (ภาพเต็ม = บรรทัดคร่อม `crossing` บรรทัด, ซีกซ้าย/ขวา = บรรทัดเดียว)"""
    from PIL import Image

    fake_install(monkeypatch, tmp_path)
    widths = []

    def fake_run(cmd, timeout):
        if "--version" in cmd:
            return 0, "tesseract v5.4.0.20240606\n", ""
        w = Image.open(cmd[1]).size[0]
        widths.append(w)
        rows = [HEAD]
        if w == 1000:
            for i in range(crossing):
                rows.append(row(5, 1, 1, i + 1, 1, 10, 30 * i, 900, 20, 80.0, f"คร่อม{i}"))
        elif w == 200:
            rows.append(row(5, 1, 1, 1, 1, 15, 50, 120, 20, 90.0, "แถบ"))
        else:
            rows.append(row(5, 1, 1, 1, 1, 20, 50, 300, 20, 95.0, "เนื้อหา"))
        return 0, "\n".join(rows), ""

    monkeypatch.setattr(T, "_run", fake_run)
    return widths


def test_side_band_page_is_ocrd_per_region_with_offsets(monkeypatch, tmp_path):
    widths = side_band_fake(monkeypatch, tmp_path, crossing=6)
    r = T.TesseractLocalBackend().recognize(png_of(band_image(band=200)))
    assert widths == [1000, 200, 800]  # เต็มภาพก่อน แล้วแถบข้าง / เนื้อหา
    assert r.extra["column_split"] == {"x": 200, "method": "side-band-v1", "crossing_lines_before": 6}
    by = {ln["text"]: ln for ln in r.extra["lines"]}
    assert by["แถบ"]["bbox"] == [15, 50, 135, 70] and by["แถบ"]["col"] == 0
    assert by["เนื้อหา"]["bbox"] == [220, 50, 520, 70] and by["เนื้อหา"]["col"] == 1  # บวก offset ซีกขวาคืนพิกัดภาพเดิม
    assert by["แถบ"]["block"] != by["เนื้อหา"]["block"] and set(by) == {"แถบ", "เนื้อหา"}
    assert "\n\n" in r.text  # คนละ block = คนละย่อหน้า


def test_side_band_not_split_when_too_few_lines_cross(monkeypatch, tmp_path):
    widths = side_band_fake(monkeypatch, tmp_path, crossing=4)  # ขอบชัดแต่คร่อมแค่ 4 < 5
    r = T.TesseractLocalBackend().recognize(png_of(band_image(band=200)))
    assert widths == [1000] and r.extra["column_split"] is None and len(r.extra["lines"]) == 4


def test_side_band_not_split_without_clear_edge(monkeypatch, tmp_path):
    from PIL import Image

    widths = side_band_fake(monkeypatch, tmp_path, crossing=9)
    r = T.TesseractLocalBackend().recognize(png_of(Image.new("RGB", (1000, 600), "white")))
    assert widths == [1000] and r.extra["column_split"] is None


def test_cache_key_params_include_column_split_method(monkeypatch, tmp_path):
    fake_install(monkeypatch, tmp_path)
    p = T.TesseractLocalBackend().cache_params()
    assert p["column_split"][0] == "side-band-v1" and p["langs"] == "tha+eng"
