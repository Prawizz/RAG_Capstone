"""ค่าตั้งของระบบ อ่านจาก environment (และไฟล์ ``.env`` ถ้ามี)

อ่านค่าที่เดียวแล้วส่งต่อเป็น ``Settings`` ห้ามโมดูลอื่นอ่าน ``os.environ`` ของค่าพวกนี้เอง
(ไม่งั้นค่าตั้งจะกระจายและ test override ไม่ได้)

- ชื่อ env ตรงกับ ``.env.example``
- ``vector_size`` ต้องตรงกับโมเดล embedding ที่ใช้จริง (BGE-M3 = 1024) เปลี่ยนโมเดลที่มีมิติต่างไป
  ต้องสร้าง collection ใหม่แล้ว index ซ้ำ
- ``embedding_test_mode`` = เปิดใช้ hash vector แทน BGE-M3 **ต้องระบุชัดเจน** (EMBEDDING_TEST_MODE=1) — ค่าเริ่มต้นปิด
  ถ้าโหลด BGE-M3 ไม่ได้ ``app/embedding.py`` จะ raise ไม่ fallback เงียบๆ
- collection แยก public / private ตาม PLAN.md ข้อ 8 (Public/Private อยู่คนละ collection)
- ``embedding_device`` (EMBEDDING_DEVICE = auto|cpu|cuda) และ ``ocr_backend`` (OCR_BACKEND = auto|typhoon-local|tesseract-local)
  ตัดสินค่าจริงที่ ``app/runtime.py`` (log ทุกครั้ง ไม่ fallback เงียบ) — ค่าอื่นนอกจากนี้ = error ตอนอ่านค่าตั้ง
- ``service_key`` (SERVICE_KEY) = ค่าที่ Gateway ส่งใน header ``X-Service-Key`` — placeholder ของ service auth
  (ของจริงจะเป็น mTLS หรือ signed token) ไม่ตั้ง = ทุก request ที่ต้อง auth ได้ 401 (fail closed)
- ``system_admin_users`` (SYSTEM_ADMIN_USERS คั่นด้วยจุลภาค) = user_id ที่สร้างกล่องได้ (role admin ระดับระบบ)
- ``retrieve_min_score`` (RETRIEVE_MIN_SCORE) = เกณฑ์ no_relevant_context — ค่า placeholder จนกว่าจะ calibrate

``.env`` ไม่เข้า git (ดู .gitignore) ใช้ ``.env.example`` เป็นแม่แบบ
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

# ค่าเริ่มต้น: แก้ที่นี่ที่เดียว
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_VECTOR_SIZE = 1024  # BGE-M3
DEFAULT_CHUNK_TARGET_CHARS = 1000  # ขนาด chunk (parent) เป้าหมายตอน index ลง Qdrant
DEFAULT_EMBEDDING_DEVICE = "auto"  # auto = cuda ถ้ามี ไม่งั้น cpu (CPU ใช้ fp32) — ตัดสินที่ app/runtime.py
DEFAULT_OCR_BACKEND = "auto"  # auto = typhoon-local เมื่อมี GPU, tesseract-local เมื่อไม่มี — ตัดสินที่ app/runtime.py
DEFAULT_EMBEDDING_MAX_LENGTH = 2048  # token ต่อ chunk (chunk ~1,500 ตัวอักษรไทยยังไม่ถึงเพดาน)
DEFAULT_EMBEDDING_BATCH_SIZE = 16
DEFAULT_SEARCH_MODE = "dense+sparse_bge"  # dense | dense+sparse_bge | dense+sparse_bm25 | all (sparse_bm25 index ไว้แต่ไม่ใช่ค่าเริ่มต้น)
DEFAULT_COLLECTION_PUBLIC = "rag_public"
DEFAULT_COLLECTION_PRIVATE = "rag_private"
DEFAULT_QDRANT_TIMEOUT = 30.0
# API (contract v1)
DEFAULT_RETRIEVE_MIN_SCORE = 0.3  # placeholder จนกว่าจะ calibrate: top_score ต่ำกว่านี้ -> no_relevant_context
DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

ENV_FILE = ".env"


def load_env_file(path: str | Path = ENV_FILE) -> dict[str, str]:
    """อ่าน ``.env`` แบบง่าย (KEY=VALUE บรรทัดละตัว) — ไม่เพิ่ม dependency python-dotenv

    ข้ามบรรทัดว่างและบรรทัดที่ขึ้นต้นด้วย ``#`` ตัดเครื่องหมายคำพูดรอบค่าออก
    ไม่รองรับค่าหลายบรรทัดหรือการอ้างตัวแปรอื่น (ไม่จำเป็นสำหรับค่าที่เราใช้)
    """
    file = Path(path)
    if not file.is_file():
        return {}
    out: dict[str, str] = {}
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip("\"'")
    return out


class Settings(BaseModel):
    """ค่าตั้งทั้งระบบ (immutable) สร้างด้วย ``get_settings()`` หรือ ``Settings()`` ใน test"""

    model_config = {"frozen": True}

    qdrant_url: str = DEFAULT_QDRANT_URL
    qdrant_api_key: Optional[str] = None
    qdrant_timeout: float = Field(default=DEFAULT_QDRANT_TIMEOUT, gt=0)
    collection_public: str = DEFAULT_COLLECTION_PUBLIC
    collection_private: str = DEFAULT_COLLECTION_PRIVATE
    vector_size: int = Field(default=DEFAULT_VECTOR_SIZE, gt=0)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_target_chars: int = Field(default=DEFAULT_CHUNK_TARGET_CHARS, gt=0)
    embedding_device: Literal["auto", "cpu", "cuda"] = DEFAULT_EMBEDDING_DEVICE
    embedding_dtype: Literal["bf16", "fp16", "fp32"] = "bf16"  # ชนิดตัวเลขบน GPU (CPU ใช้ fp32 เสมอ)
    embedding_max_length: int = Field(default=DEFAULT_EMBEDDING_MAX_LENGTH, gt=0)
    embedding_batch_size: int = Field(default=DEFAULT_EMBEDDING_BATCH_SIZE, gt=0)
    embedding_test_mode: bool = False
    search_mode: str = DEFAULT_SEARCH_MODE
    ocr_backend: Literal["auto", "typhoon-local", "tesseract-local"] = DEFAULT_OCR_BACKEND
    service_key: Optional[str] = Field(default=None, repr=False)
    system_admin_users: frozenset[str] = frozenset()
    retrieve_min_score: float = DEFAULT_RETRIEVE_MIN_SCORE
    max_upload_bytes: int = Field(default=DEFAULT_MAX_UPLOAD_BYTES, gt=0)


def _env(source: dict[str, str], key: str) -> Optional[str]:
    """environment จริงมาก่อน ``.env`` (ให้ override ตอนรันได้)"""
    value = os.environ.get(key, source.get(key))
    return value if value not in (None, "") else None


def build_settings(env_file: str | Path = ENV_FILE) -> Settings:
    """ประกอบ ``Settings`` จาก environment + ``.env`` (ไม่ cache — ใช้ใน test ได้)"""
    source = load_env_file(env_file)
    fields: dict[str, object] = {}
    for name, key in (
        ("qdrant_url", "QDRANT_URL"),
        ("qdrant_api_key", "QDRANT_API_KEY"),
        ("qdrant_timeout", "QDRANT_TIMEOUT"),
        ("collection_public", "COLLECTION_PUBLIC"),
        ("collection_private", "COLLECTION_PRIVATE"),
        ("vector_size", "VECTOR_SIZE"),
        ("embedding_model", "EMBEDDING_MODEL"),
        ("chunk_target_chars", "CHUNK_TARGET_CHARS"),
        ("embedding_device", "EMBEDDING_DEVICE"),
        ("ocr_backend", "OCR_BACKEND"),
        ("embedding_dtype", "EMBEDDING_DTYPE"),
        ("embedding_max_length", "EMBEDDING_MAX_LENGTH"),
        ("embedding_batch_size", "EMBEDDING_BATCH_SIZE"),
        ("embedding_test_mode", "EMBEDDING_TEST_MODE"),
        ("search_mode", "SEARCH_MODE"),
        ("service_key", "SERVICE_KEY"),
        ("retrieve_min_score", "RETRIEVE_MIN_SCORE"),
        ("max_upload_bytes", "MAX_UPLOAD_BYTES"),
    ):
        value = _env(source, key)
        if value is not None:
            fields[name] = value
    admins = _env(source, "SYSTEM_ADMIN_USERS")
    if admins is not None:
        fields["system_admin_users"] = frozenset(a.strip() for a in admins.split(",") if a.strip())
    return Settings(**fields)  # pydantic แปลงชนิดและตรวจค่าให้


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """ค่าตั้งของ process (cache) — test ที่อยากเปลี่ยนค่าให้เรียก ``get_settings.cache_clear()``"""
    return build_settings()
