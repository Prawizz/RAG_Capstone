"""ตั้งค่ากลางของ test: แยกแคช OCR ต่อ test (ห้าม test เขียนลง data/ocr_cache จริง)"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_ocr_cache(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("OCR_CACHE_DIR", str(tmp_path_factory.mktemp("ocr_cache")))
