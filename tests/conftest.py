"""ตั้งค่ากลางของ test: แยกแคช OCR ต่อ test (ห้าม test เขียนลง data/ocr_cache จริง)"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_ocr_cache(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("OCR_CACHE_DIR", str(tmp_path_factory.mktemp("ocr_cache")))
    monkeypatch.setenv("DATA_DIR", str(tmp_path_factory.mktemp("data")))  # ห้าม test เขียน rag.db / files ลง data/ จริง


def pytest_configure(config):
    """test ปกติใช้ hash vector แทน BGE-M3 (ไม่โหลดโมเดล 2 GB) — เปิดโหมด test ชัดเจนผ่าน env ตามที่ ``app/embedding.py`` กำหนด
    test ที่ต้องใช้โมเดลจริงสร้าง ``BGEM3Embedder`` ตรงๆ (mark ``slow``)"""
    import os

    os.environ.setdefault("EMBEDDING_TEST_MODE", "1")
