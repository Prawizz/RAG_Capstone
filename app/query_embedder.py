"""embedder ของคำถามตอน ``/v1/retrieve`` — BGE-M3 อีกตัวบน **CPU (fp32) เสมอ** แยกจาก embedder ของ ingest worker

เหตุผลที่แยก: worker ปล่อย/โหลด embedder บน GPU สลับกับ OCR (ห้ามอยู่บน GPU พร้อมกัน) ถ้าคำถามใช้ตัวเดียวกัน
retrieve จะต้องรอ OCR หรือดึงโมเดลกลับขึ้น GPU ระหว่าง OCR · บน CPU embed คำถามหนึ่งข้อ ~50 ms (วัดแล้วในขั้นก่อน)

- โหลดครั้งเดียว (``load``) ใน lifespan ของ API — app ที่ไม่ได้เปิด lifespan (test) โหลดตอนใช้ครั้งแรก
- โหลดไม่ได้ = ``down`` จนกว่าจะเริ่ม process ใหม่ (ไม่ลองโหลดซ้ำทุก request: โหลดโมเดล 2 GB ช้า) ``/v1/health/ready``
  รายงาน embedding เป็น down พร้อมเหตุผล และ retrieve ตอบ 503 unavailable
- inference อยู่ใน lock (route ของ FastAPI รันใน threadpool หลาย thread พร้อมกัน — โมเดลตัวเดียวห้ามเรียกซ้อน)
- ``EMBEDDING_TEST_MODE`` = ``HashEmbedder`` (เฉพาะ test) เหมือน ``app.embedding.get_embedder``
- ข้อความคำถามห้ามอยู่ใน log / ข้อความ error (``EmbeddingError`` ของ BGE-M3 แนบข้อความต้นฉบับ — เก็บแค่ชนิดของ error)

คำถามต้องผ่าน ``app.thai.normalizer.normalize`` ก่อนส่งเข้ามา (ผู้เรียกทำ — normalizer มีตัวเดียว กฎเหล็กข้อ 1)
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Literal, Optional

from app.config import Settings
from app.embedding import BGEM3Embedder, Embedder, HashEmbedder, bm25_sparse
from app.vectorstore import Query

log = logging.getLogger("app.query_embedder")

QUERY_DEVICE = "cpu"
QUERY_DTYPE = "fp32"

State = Literal["not_loaded", "ok", "down"]


class QueryEmbedderUnavailable(RuntimeError):
    """embed คำถามไม่ได้ (โหลดโมเดลไม่ได้ หรือ inference ล้ม) — API ตอบ 503 unavailable"""


def make_query_embedder(settings: Settings) -> Embedder:
    if settings.embedding_test_mode:
        return HashEmbedder(settings.vector_size)
    return BGEM3Embedder(settings.embedding_model, device=QUERY_DEVICE, dtype=QUERY_DTYPE,
                         max_length=settings.embedding_max_length, batch_size=1)


class QueryEmbedder:
    def __init__(self, settings: Settings, factory: Optional[Callable[[], Embedder]] = None) -> None:
        self.settings = settings
        self._factory = factory or (lambda: make_query_embedder(settings))
        self._embedder: Optional[Embedder] = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self.state: State = "not_loaded"
        self.reason: Optional[str] = None

    @property
    def description(self) -> str:
        if self.settings.embedding_test_mode:
            return "hash (EMBEDDING_TEST_MODE)"
        return f"{self.settings.embedding_model} บน {QUERY_DEVICE} ({QUERY_DTYPE})"

    def load(self) -> bool:
        """โหลดครั้งเดียว (เรียกซ้ำได้) คืน True ถ้าใช้ได้"""
        with self._load_lock:
            if self.state == "not_loaded":
                try:
                    embedder = self._factory()
                    if embedder.dense_size != self.settings.vector_size:
                        raise ValueError(f"มิติ dense {embedder.dense_size} ไม่ตรง VECTOR_SIZE={self.settings.vector_size}")
                    self._embedder = embedder
                    self.state = "ok"
                    log.info("query embedder พร้อม: %s", self.description)
                except Exception as e:  # ไม่ล้ม process — ready รายงาน down และ retrieve ตอบ 503
                    self.state = "down"
                    self.reason = f"{type(e).__name__}: {e}"
                    log.error("query embedder (%s) ใช้ไม่ได้: %s", self.description, self.reason)
            return self.state == "ok"

    def embed(self, normalized_query: str) -> Query:
        """dense + sparse_bge จากโมเดล (ใน lock) + sparse_bm25 ตัดคำฝั่ง client

        Raises:
            QueryEmbedderUnavailable: โหลดไม่ได้ หรือ inference ล้ม (ข้อความ error ไม่มีข้อความคำถาม)
        """
        if self.state == "not_loaded":
            self.load()
        if self.state != "ok" or self._embedder is None:
            raise QueryEmbedderUnavailable(f"query embedder ใช้ไม่ได้: {self.reason}")
        try:
            with self._infer_lock:
                e = self._embedder.embed_query(normalized_query)
        except Exception as exc:
            raise QueryEmbedderUnavailable(f"embed คำถามไม่สำเร็จ ({type(exc).__name__})") from None
        return Query(dense=e.dense, sparse_bge=e.sparse, sparse_bm25=bm25_sparse(normalized_query))

    def close(self) -> None:
        with self._load_lock:
            if self._embedder is not None:
                self._embedder.release()
                self._embedder = None
