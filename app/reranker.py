"""cross-encoder reranker (ทดลอง ปิดเป็นค่าเริ่มต้น) — ``RERANKER=bge-v2-m3`` = BAAI/bge-reranker-v2-m3

ใน ``/v1/retrieve``: ดึง ``RERANK_CANDIDATES`` (20) อันดับแรกจาก RRF -> ให้คะแนนคู่ (คำถาม, chunk) -> เรียงใหม่ -> คืน top_k
- ข้อความ chunk = ``context line + เนื้อ`` ชุดเดียวกับที่ส่งเข้า embedding (``Chunk.embed_text``) · คำถาม = หลัง normalize
- คะแนน = sigmoid(logit) ตามวิธีของ model card (FlagReranker ``normalize=True``) อยู่ในช่วง 0–1
  **ไม่ได้แทน ``score`` ใน API** (score ยังเป็น dense cosine) — คะแนน reranker ลง log (``rerank_top``) ไว้วัดผลเท่านั้น
- ``max_length`` 512 ตาม model card (คู่ที่ยาวเกินตัดฝั่ง chunk) · CPU fp32 / CUDA bf16
  (fp16 ล้นจนได้ NaN จริงในชุด dev 1 ข้อ — bf16 ช่วงตัวเลขเท่า fp32) · คะแนนไม่ใช่ตัวเลขจำกัด (NaN/inf) = error ไม่เรียงต่อเงียบๆ
- โหลดครั้งเดียว (lifespan) ไม่ได้ = ``down`` จนกว่าจะเริ่ม process ใหม่ -> retrieve ตอบ 503 และ ready รายงาน not_ready
  (ไม่ถอยไปไม่ rerank เงียบๆ) · inference อยู่ใน lock
- ห้ามรันบน GPU พร้อม OCR (GPU 8 GB) — ค่าเริ่มต้น ``RERANKER_DEVICE=cpu``
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Callable, Literal, Optional, Protocol, Sequence

from app.config import Settings

log = logging.getLogger("app.reranker")

RERANK_CANDIDATES = 20
MODELS = {"bge-v2-m3": "BAAI/bge-reranker-v2-m3"}
MAX_LENGTH = 512
BATCH_SIZE = 10

State = Literal["off", "not_loaded", "ok", "down"]


class RerankerUnavailable(RuntimeError):
    pass


class PairScorer(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...

    def release(self) -> None: ...


class CrossEncoder:
    """``AutoModelForSequenceClassification`` + tokenizer แบบคู่ (ตาม model card ของ bge-reranker-v2-m3)"""

    def __init__(self, model_id: str, device: str = "cpu", max_length: int = MAX_LENGTH,
                 batch_size: int = BATCH_SIZE) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            raise RerankerUnavailable("RERANKER_DEVICE=cuda แต่ไม่พบ GPU CUDA (ไม่ถอยไป CPU เอง)")
        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=dtype).to(device).eval()

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        out: list[float] = []
        torch = self.torch
        with torch.inference_mode():
            for start in range(0, len(passages), self.batch_size):
                pairs = [[query, p] for p in passages[start:start + self.batch_size]]
                enc = self.tokenizer(pairs, padding=True, truncation=True, max_length=self.max_length,
                                     return_tensors="pt").to(self.device)
                logits = self.model(**enc, return_dict=True).logits.view(-1).float()
                out.extend(torch.sigmoid(logits).tolist())
        return out

    def release(self) -> None:
        self.model = None
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


class Reranker:
    """ตัวห่อของ API: สถานะ + โหลดครั้งเดียว + lock (เหมือน ``QueryEmbedder``)"""

    def __init__(self, settings: Settings, factory: Optional[Callable[[], PairScorer]] = None) -> None:
        self.settings = settings
        self.name = settings.reranker
        self._factory = factory or (lambda: CrossEncoder(MODELS[self.name], device=settings.reranker_device))
        self._scorer: Optional[PairScorer] = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self.state: State = "off" if self.name == "off" else "not_loaded"
        self.reason: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self.name != "off"

    @property
    def description(self) -> str:
        return f"{MODELS.get(self.name, self.name)} บน {self.settings.reranker_device}"

    def load(self) -> bool:
        if not self.enabled:
            return False
        with self._load_lock:
            if self.state == "not_loaded":
                try:
                    self._scorer = self._factory()
                    self.state = "ok"
                    log.info("reranker พร้อม: %s", self.description)
                except Exception as e:
                    self.state = "down"
                    self.reason = f"{type(e).__name__}: {e}"
                    log.error("reranker (%s) ใช้ไม่ได้: %s", self.description, self.reason)
            return self.state == "ok"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """คะแนนต่อ passage (0–1) — ใช้ไม่ได้ -> ``RerankerUnavailable`` (ข้อความ error ไม่มีข้อความคำถาม/chunk)"""
        if self.state == "not_loaded":
            self.load()
        if self.state != "ok" or self._scorer is None:
            raise RerankerUnavailable(f"reranker ใช้ไม่ได้: {self.reason}")
        try:
            with self._infer_lock:
                scores = self._scorer.score(query, list(passages))
        except Exception as exc:
            raise RerankerUnavailable(f"rerank ไม่สำเร็จ ({type(exc).__name__})") from None
        if len(scores) != len(passages):
            raise RerankerUnavailable("จำนวนคะแนนไม่ตรงจำนวน passage")
        out = [float(s) for s in scores]
        if not all(math.isfinite(x) for x in out):
            raise RerankerUnavailable("คะแนน reranker ไม่ใช่ตัวเลข (NaN/inf)")
        return out

    def close(self) -> None:
        with self._load_lock:
            if self._scorer is not None:
                self._scorer.release()
                self._scorer = None


def rerank_order(scores: Sequence[float]) -> list[int]:
    """ลำดับใหม่ตามคะแนนมากไปน้อย — คะแนนเท่ากันคงลำดับ RRF เดิม (stable)"""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))
