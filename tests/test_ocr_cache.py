"""แคชผล OCR (app/extraction/ocr_cache.py) — ไม่ต้องมี GPU"""

import pytest

from app.extraction import ocr as ocr_mod
from app.extraction.ocr import FakeBackend, run_ocr
from app.extraction.ocr_cache import OcrCache, cache_dir, make_key, make_key_fields
from app.extraction.pipeline import extract_file
from app.models import DetectedType

from tests.test_ocr import THAI_LONG, ink_image, make_pdf, save_png

BASE = dict(content_hash="h", page=3, backend="typhoon-local", revision="abc", dpi=200,
            split_method="gutter", split_x=0.5854, params={"do_sample": False, "max_new_tokens": 10000})


def key(**override):
    return make_key(make_key_fields(**{**BASE, **override}))


def test_key_is_stable_and_hex():
    assert key() == key() and len(key()) == 64


@pytest.mark.parametrize("override", [
    {"content_hash": "h2"}, {"page": 4}, {"backend": "tesseract-local"}, {"revision": "abd"}, {"dpi": 300},
    {"split_method": "center"}, {"split_x": 0.5}, {"params": {"do_sample": True, "max_new_tokens": 10000}},
    {"params": {"do_sample": False, "max_new_tokens": 5000}},
])
def test_key_changes_when_any_field_changes(override):
    assert key(**override) != key()


def test_split_x_rounded_and_param_order_irrelevant():
    assert key(split_x=0.58540001) == key(split_x=0.5854)
    assert key(params={"b": 1, "a": 2}) == key(params={"a": 2, "b": 1})


def test_cache_dir_env_and_default(monkeypatch, tmp_path):
    monkeypatch.setenv("OCR_CACHE_DIR", str(tmp_path / "x"))
    assert cache_dir() == tmp_path / "x"
    monkeypatch.delenv("OCR_CACHE_DIR")
    assert cache_dir().parts[-2:] == ("data", "ocr_cache")  # ใน .gitignore (data/)


def test_cache_hit_does_not_call_backend(scan_pdf_cache):
    f, doc = scan_pdf_cache
    b1 = FakeBackend(THAI_LONG)
    out1 = run_ocr(doc, f, b1)
    calls = b1.calls
    assert calls == 2 and "ocr:cache_hit" not in out1.units[1].signals
    b2 = FakeBackend(THAI_LONG)  # backend ใหม่ พารามิเตอร์เดียวกัน
    out2 = run_ocr(doc, f, b2)
    assert b2.calls == 0
    assert [u.text for u in out2.units] == [u.text for u in out1.units]
    assert out2.units[1].signals["ocr:cache_hit"] == 1.0
    assert out2.units[1].signals["ocr:seconds"] == out1.units[1].signals["ocr:seconds"]  # เวลาเดิมตอนรันจริง


def test_cache_skip_and_refresh(scan_pdf_cache):
    f, doc = scan_pdf_cache
    run_ocr(doc, f, FakeBackend(THAI_LONG))
    b = FakeBackend(THAI_LONG)
    run_ocr(doc, f, b, use_cache=False)  # ข้ามแคช: เรียก backend
    assert b.calls == 2
    b2 = FakeBackend(THAI_LONG)
    run_ocr(doc, f, b2, refresh_cache=True)  # ไม่อ่านแต่เขียนทับ
    assert b2.calls == 2
    b3 = FakeBackend(THAI_LONG)
    run_ocr(doc, f, b3)
    assert b3.calls == 0


def test_use_cache_false_writes_nothing(scan_pdf_cache, tmp_path):
    f, doc = scan_pdf_cache
    run_ocr(doc, f, FakeBackend(THAI_LONG), use_cache=False)
    assert not list(cache_dir().glob("*.json"))


def test_different_backend_params_do_not_share_cache(scan_pdf_cache):
    f, doc = scan_pdf_cache
    run_ocr(doc, f, FakeBackend("ก" * 150))
    b = FakeBackend("ข" * 150)  # cache_params ต่าง (ข้อความที่ตั้งต่าง)
    out = run_ocr(doc, f, b)
    assert b.calls == 2 and "ข" in out.units[1].text


def test_dpi_change_changes_key(scan_pdf_cache, monkeypatch):
    f, doc = scan_pdf_cache
    run_ocr(doc, f, FakeBackend(THAI_LONG))
    monkeypatch.setattr(ocr_mod, "OCR_RENDER_DPI", 150)
    b = FakeBackend(THAI_LONG)
    run_ocr(doc, f, b)
    assert b.calls == 2


def test_cache_keeps_extra_tokens_and_split(tmp_path):
    f = save_png(tmp_path, ink_image(2000, 800, [(0.02, 0.56), (0.64, 0.98)]))  # หน้ากว้าง: ตัดสองภาพ
    doc = extract_file(f)

    class Rich(FakeBackend):
        def recognize(self, image_png):
            r = super().recognize(image_png)
            r.tokens, r.extra = 7, {"lines": [{"text": "x", "bbox": [1, 2, 3, 4]}]}
            return r

    out1 = run_ocr(doc, f, Rich(THAI_LONG))
    res, page, hit = ocr_mod.recognize_page_cached(Rich(THAI_LONG), f, DetectedType.PNG, doc.content_hash, 1)
    assert hit and res.tokens == 14 and len(res.extra["parts"]) == 2 and page.split_method == "gutter"
    assert res.extra["parts"][0]["lines"][0]["bbox"] == [1, 2, 3, 4]
    assert out1.units[0].signals["ocr:split_x"] == pytest.approx(0.6, abs=0.01)


def test_corrupt_cache_file_is_ignored(scan_pdf_cache):
    f, doc = scan_pdf_cache
    run_ocr(doc, f, FakeBackend(THAI_LONG))
    for p in cache_dir().glob("*.json"):
        p.write_text("{not json", encoding="utf-8")
    b = FakeBackend(THAI_LONG)
    run_ocr(doc, f, b)
    assert b.calls == 2


def test_cache_put_get_roundtrip(tmp_path):
    c = OcrCache(tmp_path)
    r = ocr_mod.OcrResult(text="สวัสดี", backend="b", model="m", revision="r", seconds=1.5, tokens=9, extra={"a": 1})
    c.put("k" * 64, {"x": 1}, r)
    got = c.get("k" * 64)
    assert got["text"] == "สวัสดี" and got["tokens"] == 9 and got["extra"] == {"a": 1} and got["fields"] == {"x": 1}
    assert c.get("z" * 64) is None


@pytest.fixture
def scan_pdf_cache(tmp_path):
    f = make_pdf(tmp_path / "scan.pdf", ["Hello text layer page " * 10, "", ""])
    return f, extract_file(f)
