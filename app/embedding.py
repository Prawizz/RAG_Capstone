"""Embedding: BGE-M3 (dense 1024 มิติ + sparse lexical weights) และ hash vector สำหรับ test เท่านั้น

**ห้าม fallback เงียบๆ** — ถ้าโหลด BGE-M3 ไม่ได้ ``get_embedder()`` โยน ``EmbeddingUnavailable`` (ingest จริงต้องล้มดังๆ
ไม่ใช่เขียนเวกเตอร์ปลอมลง index แล้วค้นได้ผลสุ่ม) hash vector (``HashEmbedder``) ใช้ได้ต่อเมื่อเปิดโหมด test **ระบุชัดเจน**เท่านั้น:
``get_embedder(test_mode=True)`` หรือ ``EMBEDDING_TEST_MODE=1`` (``Settings.embedding_test_mode``) หรือสร้าง ``HashEmbedder()`` ตรงๆ

**BGE-M3 จริง ไม่เพิ่ม dependency** — ใช้ ``transformers`` + ``torch`` (มีอยู่แล้วใน requirements-ocr.txt) ทำซ้ำสูตรของ FlagEmbedding ``BGEM3FlagModel``:
- dense  = ``last_hidden_state[:, 0]`` (CLS) แล้ว L2-normalize
- sparse = ``relu(sparse_linear(last_hidden_state))`` ต่อ token (ไฟล์ ``sparse_linear.pt`` ของ repo BAAI/bge-m3, Linear(1024→1))
  ตัด token พิเศษ (cls/eos/pad/unk) แล้วเก็บน้ำหนักสูงสุดต่อ token id ที่ weight > 0 (index = token id ของ tokenizer)
ไม่ใช้ colbert (multi-vector) — นอก scope ของงานนี้

**ชนิดตัวเลขบน GPU** — ค่าเริ่มต้น bf16 (``Settings.embedding_dtype``; CPU ใช้ fp32 เสมอ) เหตุผลที่ไม่ใช้ fp16: BGE-M3 ใน fp16 บน GPU ให้ผลไม่ finite เป็นบางครั้งโดยสุ่ม (วัดได้ ~1–3 chunk ต่อ 1,500 chunk ยาว ~1,000 ตัวอักษร
ไม่ซ้ำกันทุกรอบ แม้ตัดเป็น batch เดียวกัน; พบทั้ง sdpa และ eager) ทุกแถวที่ได้ NaN/inf (ไม่ว่า dtype ไหน) จึงถูกคำนวณใหม่ทีละแถวใน dtype เดิม แล้วถ้ายังไม่ได้ใน fp32
(ตัวเดียวกัน แค่ละเอียดกว่า) ถ้ายังไม่ finite = ``EmbeddingError`` — ไม่มีทางที่เวกเตอร์ NaN หลุดเข้า index (``retries`` / ``fp32_fallbacks`` นับไว้ให้ดู)

**VRAM** — bf16/fp16 ครึ่งหนึ่งของ fp32 บน GPU; ``Embedder.release()`` คืน VRAM (ingest ต้องทำ OCR ให้เสร็จก่อนแล้วค่อย embed
ห้ามให้ Typhoon กับ BGE-M3 อยู่บน GPU พร้อมกัน — เรียก ``release_gpu_memory()`` หลังปล่อยโมเดลตัวก่อนหน้า)
"""

from __future__ import annotations

import gc
import hashlib
import math
import re
import struct
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from app.config import DEFAULT_EMBEDDING_MODEL, Settings, get_settings
from app.thai.normalizer import normalize

BGE_M3_DIMENSION = 1024

_WORD_CHARS = re.compile(r"[\w]", re.UNICODE)


class EmbeddingError(RuntimeError):
    """embedding ทำงานไม่สำเร็จ (ห้ามส่งเวกเตอร์ที่ไม่ถูกต้องต่อไปเงียบๆ)"""


class EmbeddingUnavailable(EmbeddingError):
    """โหลด/ใช้โมเดล embedding ไม่ได้ (ไม่ fallback เป็น hash vector — ดู docstring ของโมดูล)"""


@dataclass(frozen=True)
class SparseVector:
    """sparse vector แบบ (index, value) — index เป็นจำนวนเต็ม 32 บิตไม่ติดลบ เรียงจากน้อยไปมากและไม่ซ้ำ"""

    indices: tuple[int, ...] = ()
    values: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError("indices กับ values ต้องยาวเท่ากัน")

    @classmethod
    def from_weights(cls, weights: dict[int, float]) -> "SparseVector":
        items = sorted((i, w) for i, w in weights.items() if w > 0)
        return cls(tuple(i for i, _ in items), tuple(w for _, w in items))

    def __bool__(self) -> bool:
        return bool(self.indices)


@dataclass(frozen=True)
class Embedded:
    """ผล embed ข้อความหนึ่งก้อน: dense + sparse lexical weights ของ BGE-M3"""

    dense: tuple[float, ...]
    sparse: SparseVector


class Embedder(Protocol):
    dense_size: int

    def embed_documents(self, texts: Sequence[str]) -> list[Embedded]: ...

    def embed_query(self, text: str) -> Embedded: ...

    def release(self) -> None: ...


def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Cosine similarity ระหว่างเวกเตอร์สองตัว"""
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimensions do not match: {len(v1)} vs {len(v2)}")
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


# ---------- GPU ----------


def release_gpu_memory() -> None:
    """เก็บ garbage แล้วคืน cache ของ CUDA (ไม่ทำอะไรถ้าไม่มี torch/GPU)"""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:  # ไม่มี torch = ไม่มี VRAM ให้คืน
        pass


def gpu_peak_memory_mb() -> float:
    """VRAM สูงสุดที่ process นี้จองไว้ (MB) นับจากครั้งล่าสุดที่ ``reset_gpu_peak()``; ไม่มี GPU = 0"""
    try:
        import torch

        return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    except Exception:
        return 0.0


def reset_gpu_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ---------- BGE-M3 ----------


_TORCH_DTYPES = {"bf16": lambda t: t.bfloat16, "fp16": lambda t: t.float16, "fp32": lambda t: t.float32}


class BGEM3Embedder:
    """BGE-M3 dense + sparse ผ่าน transformers (ดูสูตรใน docstring ของโมดูล)

    Raises:
        EmbeddingUnavailable: โหลดโมเดลหรือ ``sparse_linear.pt`` ไม่ได้
    """

    dense_size = BGE_M3_DIMENSION

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str = "auto",
        dtype: str = "bf16",
        max_length: int = 2048,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.retries = 0  # แถวที่ได้ NaN/inf แล้วคำนวณใหม่ทีละแถว
        self.fp32_fallbacks = 0  # แถวที่ต้องใช้ fp32
        self.max_length = max_length
        self.batch_size = batch_size
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            from app.runtime import resolve_embedding_device

            # auto|cpu|cuda -> อุปกรณ์จริง (log ทุกครั้ง; cuda แต่ไม่มี GPU = error ไม่ถอยไป CPU เอง) · CPU = fp32
            choice = resolve_embedding_device(device, dtype)
            device = choice.device
            self.device = device
            self.device_choice = choice
            self.dtype = _TORCH_DTYPES[choice.dtype](torch)
            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name, dtype=self.dtype).to(device).eval()
            hidden = self._model.config.hidden_size
            linear = torch.nn.Linear(hidden, 1)
            linear.load_state_dict(torch.load(self._sparse_linear_path(model_name), map_location="cpu"))
            self._sparse_linear = linear.to(device=device, dtype=self.dtype).eval()
            tok = self._tokenizer
            self._special_ids = {i for i in (tok.cls_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id) if i is not None}
        except Exception as e:  # ห้ามกลืน: ผู้เรียกต้องรู้ว่า embedding จริงใช้ไม่ได้
            self.release()
            raise EmbeddingUnavailable(f"โหลด {model_name} ไม่ได้: {type(e).__name__}: {e}") from e

    @staticmethod
    def _sparse_linear_path(model_name: str) -> str:
        from pathlib import Path

        local = Path(model_name) / "sparse_linear.pt"
        if local.is_file():
            return str(local)
        from huggingface_hub import hf_hub_download

        return hf_hub_download(model_name, "sparse_linear.pt")

    def _encode(self, texts: Sequence[str]) -> list[Optional[Embedded]]:
        """embed หนึ่ง batch; แถวที่ dense หรือ sparse ไม่ finite ได้ ``None`` (ผู้เรียกคำนวณใหม่)"""
        torch = self._torch
        enc = self._tokenizer(
            list(texts), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            hidden = self._model(**enc).last_hidden_state
            dense = torch.nn.functional.normalize(hidden[:, 0].float(), dim=-1).cpu()
            weights = torch.relu(self._sparse_linear(hidden)).squeeze(-1).float().cpu()
        ids, mask = enc["input_ids"].cpu(), enc["attention_mask"].cpu().bool()
        finite = torch.isfinite(dense).all(dim=-1) & (torch.isfinite(weights) | ~mask).all(dim=-1)
        out: list[Optional[Embedded]] = []
        for row in range(len(texts)):
            if not finite[row]:
                out.append(None)
                continue
            best: dict[int, float] = {}
            for tok_id, w in zip(ids[row].tolist(), weights[row].tolist()):
                if tok_id in self._special_ids or w <= 0:
                    continue
                if w > best.get(tok_id, 0.0):
                    best[tok_id] = w
            out.append(Embedded(tuple(dense[row].tolist()), SparseVector.from_weights(best)))
        return out

    @contextmanager
    def _fp32(self):
        """ยกโมเดลเป็น fp32 ชั่วคราว (น้ำหนัก bf16/fp16 -> fp32 -> กลับ ไม่สูญเสียค่า) ใช้เฉพาะแถวที่ fp16 ล้มเหลว"""
        if self.dtype == self._torch.float32:
            yield
            return
        self._model.float()
        self._sparse_linear.float()
        try:
            yield
        finally:
            self._model.to(self.dtype)
            self._sparse_linear.to(self.dtype)

    def _encode_checked(self, texts: Sequence[str]) -> list[Embedded]:
        """``_encode`` + แก้แถวที่ไม่ finite: ทีละแถวใน dtype เดิม -> fp32 -> ``EmbeddingError``"""
        result = self._encode(texts)
        for i, emb in enumerate(result):
            if emb is not None:
                continue
            self.retries += 1
            emb = self._encode([texts[i]])[0]
            if emb is None:
                self.fp32_fallbacks += 1
                with self._fp32():
                    emb = self._encode([texts[i]])[0]
            if emb is None:
                raise EmbeddingError(f"BGE-M3 ให้เวกเตอร์ที่ไม่ใช่ตัวเลขจำกัด แม้ใน fp32 (ข้อความ {len(texts[i])} ตัวอักษร: {texts[i][:60]!r})")
            result[i] = emb
        return result  # type: ignore[return-value]

    def embed_documents(self, texts: Sequence[str]) -> list[Embedded]:
        # เรียงตามความยาวเพื่อลด padding แล้วคืนลำดับเดิม
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        result: list[Optional[Embedded]] = [None] * len(texts)
        for start in range(0, len(order), self.batch_size):
            batch = order[start:start + self.batch_size]
            for i, emb in zip(batch, self._encode_checked([texts[i] for i in batch])):
                result[i] = emb
        return result  # type: ignore[return-value]

    def embed_query(self, text: str) -> Embedded:
        return self._encode_checked([text])[0]

    def release(self) -> None:
        """ปล่อยโมเดลออกจาก VRAM/RAM (เรียกซ้ำได้) — ใช้หลัง embed เสร็จก่อนโหลดโมเดลอื่นขึ้น GPU"""
        for name in ("_model", "_sparse_linear", "_tokenizer"):
            if hasattr(self, name):
                delattr(self, name)
        release_gpu_memory()


# ---------- hash vector (test เท่านั้น) ----------


def _hash_slots(token: str, dim: int) -> list[tuple[int, float]]:
    seed = token.encode("utf-8")
    slots = []
    for i in range(4):
        for val in struct.unpack("8I", hashlib.sha256(seed + bytes([i])).digest()):
            slots.append((val % dim, 1.0 if (val & 1) == 0 else -1.0))
    return slots


def lexical_tokens(text: str) -> list[str]:
    """ตัดคำไทย (pythainlp newmm) บนข้อความที่ผ่าน normalizer กลางแล้ว ตัดช่องว่าง/เครื่องหมายล้วนทิ้ง ตัวพิมพ์เล็ก

    ใช้ทั้งฝั่ง index และฝั่ง query ของ sparse_bm25 (ต้องเหมือนกันทุกประการ)
    """
    clean = normalize(text)
    if not clean.strip():
        return []
    from pythainlp.tokenize import word_tokenize

    return [t.lower() for t in word_tokenize(clean, engine="newmm", keep_whitespace=False) if _WORD_CHARS.search(t)]


def token_index(token: str) -> int:
    """token -> index 32 บิตคงที่ข้าม process (crc32; ``hash()`` ของ Python สุ่มทุกครั้งที่รัน)"""
    return zlib.crc32(token.encode("utf-8"))


def bm25_sparse(text: str) -> SparseVector:
    """sparse สำหรับ ``sparse_bm25``: ค่า = จำนวนครั้งที่คำโผล่ (tf) — IDF ให้ Qdrant คูณเองด้วย modifier IDF

    ฝั่ง query ใช้ฟังก์ชันเดียวกัน (คำซ้ำในคำถามได้น้ำหนักตามจำนวน)
    """
    counts: dict[int, float] = {}
    for tok in lexical_tokens(text):
        idx = token_index(tok)
        counts[idx] = counts.get(idx, 0.0) + 1.0
    return SparseVector.from_weights(counts)


class HashEmbedder:
    """เวกเตอร์ปลอมแบบกำหนดผลได้ (hash ของคำ + trigram) — **สำหรับ test เท่านั้น** ไม่มีความหมายเชิงความหมาย

    dense = hash เป็น ``dense_size`` มิติ L2-normalized; sparse = tf ของคำ (index เหมือน ``bm25_sparse``)
    """

    def __init__(self, dense_size: int = BGE_M3_DIMENSION) -> None:
        self.dense_size = dense_size

    def _dense(self, text: str) -> tuple[float, ...]:
        clean = normalize(text)
        vec = [0.0] * self.dense_size
        if not clean.strip():
            return tuple(vec)
        for token in lexical_tokens(clean):
            weight = 2.0 if any(c.isdigit() for c in token) else 1.0
            for idx, sign in _hash_slots(token, self.dense_size):
                vec[idx] += sign * weight
        for i in range(len(clean) - 2):
            for idx, sign in _hash_slots(clean[i:i + 3], self.dense_size):
                vec[idx] += sign * 0.3
        norm = math.sqrt(sum(x * x for x in vec))
        return tuple(x / norm for x in vec) if norm else tuple(vec)

    def embed_documents(self, texts: Sequence[str]) -> list[Embedded]:
        return [Embedded(self._dense(t), bm25_sparse(t)) for t in texts]

    def embed_query(self, text: str) -> Embedded:
        return self.embed_documents([text])[0]

    def release(self) -> None:
        pass


# ---------- ตัวเลือก embedder ----------


def get_embedder(settings: Optional[Settings] = None, *, test_mode: Optional[bool] = None) -> Embedder:
    """embedder ตามค่าตั้ง: BGE-M3 จริง หรือ ``HashEmbedder`` เมื่อ **เปิดโหมด test ชัดเจน**เท่านั้น

    ``test_mode`` ไม่ระบุ = ใช้ ``Settings.embedding_test_mode`` (EMBEDDING_TEST_MODE)

    Raises:
        EmbeddingUnavailable: โหลด BGE-M3 ไม่ได้ (ไม่มี fallback)
    """
    settings = settings or get_settings()
    if test_mode is None:
        test_mode = settings.embedding_test_mode
    if test_mode:
        return HashEmbedder(settings.vector_size)
    return BGEM3Embedder(
        settings.embedding_model,
        device=settings.embedding_device,
        dtype=settings.embedding_dtype,
        max_length=settings.embedding_max_length,
        batch_size=settings.embedding_batch_size,
    )


# ---------- ชั้นเข้ากันได้กับ prototype (dense อย่างเดียว) ----------


class EmbeddingModel:
    """ตัวห่อ dense-only สำหรับ prototype ใน ``app/retrieval.py`` — ใช้ embedder ผ่าน ``get_embedder`` (ไม่ fallback เงียบ)

    โหลดโมเดลตอนใช้ครั้งแรก ไม่ใช่ตอน import
    """

    def __init__(self, embedder: Optional[Embedder] = None, settings: Optional[Settings] = None) -> None:
        self._embedder = embedder
        self._settings = settings

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder(self._settings)
        return self._embedder

    @property
    def dimension(self) -> int:
        return self.embedder.dense_size

    def embed_text(self, text: str) -> list[float]:
        return list(self.embedder.embed_query(text).dense)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(e.dense) for e in self.embedder.embed_documents(texts)]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text(query)


_default_model: Optional[EmbeddingModel] = None


def _default() -> EmbeddingModel:
    global _default_model
    if _default_model is None:
        _default_model = EmbeddingModel()
    return _default_model


def embed_text(text: str) -> list[float]:
    return _default().embed_text(text)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    return _default().embed_texts(texts)


def embed_query(query: str) -> list[float]:
    return _default().embed_query(query)
