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
