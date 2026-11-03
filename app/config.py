"""ค่าตั้งของระบบ อ่านจาก environment (และไฟล์ ``.env`` ถ้ามี)

อ่านค่าที่เดียวแล้วส่งต่อเป็น ``Settings`` ห้ามโมดูลอื่นอ่าน ``os.environ`` ของค่าพวกนี้เอง
(ไม่งั้นค่าตั้งจะกระจายและ test override ไม่ได้)

- ชื่อ env ตรงกับ ``.env.example``
- ``vector_size`` ต้องตรงกับโมเดล embedding ที่ใช้จริง (BGE-M3 = 1024)
  **ยังไม่มีโมเดลจริงในระบบ** (ดู ``app/embedding.py``) ตอนนี้ขนาดเวกเตอร์เป็นแค่ขนาด schema
  ของ collection ถ้าเปลี่ยนโมเดลที่มีมิติต่างไป ต้องสร้าง collection ใหม่แล้ว index ซ้ำ
- collection แยก public / private ตาม PLAN.md ข้อ 8 (Public/Private อยู่คนละ collection)

``.env`` ไม่เข้า git (ดู .gitignore) ใช้ ``.env.example`` เป็นแม่แบบ
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

# ค่าเริ่มต้น: แก้ที่นี่ที่เดียว
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_VECTOR_SIZE = 1024  # BGE-M3
DEFAULT_CHUNK_MAX_CHARS = 1200
DEFAULT_COLLECTION_PUBLIC = "rag_public"
DEFAULT_COLLECTION_PRIVATE = "rag_private"
DEFAULT_QDRANT_TIMEOUT = 30.0

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
    chunk_max_chars: int = Field(default=DEFAULT_CHUNK_MAX_CHARS, gt=0)


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
        ("chunk_max_chars", "CHUNK_MAX_CHARS"),
    ):
        value = _env(source, key)
        if value is not None:
            fields[name] = value
    return Settings(**fields)  # pydantic แปลงชนิดและตรวจค่าให้


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """ค่าตั้งของ process (cache) — test ที่อยากเปลี่ยนค่าให้เรียก ``get_settings.cache_clear()``"""
    return build_settings()
