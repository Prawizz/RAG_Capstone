"""[M3] SOTA Multilingual Embedding Model (BGE-M3 / 1024 dimensions)

รองรับ:
- BGE-M3 (BAAI/bge-m3) เวกเตอร์หนาแน่น 1024 มิติ (Dense Embeddings)
- การสร้าง Embedding จากคำถาม (Query) และเอกสาร (Child Chunks)
- การคำนวณ Cosine Similarity: sim(q, d) = (q · d) / (||q|| * ||d||)
- Fallback ที่รวดเร็วและคงเส้นคงวา (Deterministic Semantic N-gram Hashing 1024 dim)
  หากยังไม่ได้ติดตั้ง PyTorch หรืออยู่ในสภาพแวดล้อมทดสอบที่ไม่มี GPU
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Sequence

from app.thai.normalizer import normalize

# ขนาดมิติเวกเตอร์ของ BGE-M3 (ตามสเปกสถาปัตยกรรม 1024 มิติ)
BGE_M3_DIMENSION = 1024


def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    """คำนวณ Cosine Similarity ระหว่างเวกเตอร์สองตัว"""
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimensions do not match: {len(v1)} vs {len(v2)}")

    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def _hash_token_to_embedding(token: str, dim: int = BGE_M3_DIMENSION) -> list[float]:
    """สร้าง Dense Feature Hash สำหรับ Token หรือ N-gram (Deterministic, L2-normalized)"""
    vec = [0.0] * dim
    # Hash หลายจุดเพื่อกระจายความหนาแน่นลงในเวกเตอร์ 1024 มิติ
    seed = token.encode("utf-8")
    for i in range(4):
        h = hashlib.sha256(seed + bytes([i])).digest()
        # แปลงเป็น int 8 ตัว
        slots = struct.unpack("8I", h)
        for val in slots:
            idx = val % dim
            sign = 1.0 if (val & 1) == 0 else -1.0
            vec[idx] += sign

    return vec


def _dense_fallback_embed(text: str, dim: int = BGE_M3_DIMENSION) -> list[float]:
    """คำนวณเวกเตอร์ Dense 1024 มิติจากข้อความไทย (Word + Subword + Char Trigrams)

    รักษา Semantic Overlap ระหว่างคำศัพท์ ตัวเลข และชื่อเฉพาะ
    และทำการ Normalize ให้ L2 Norm = 1.0
    """
    clean_text = normalize(text)
    if not clean_text.strip():
        return [0.0] * dim

    try:
        from pythainlp.tokenize import word_tokenize
        tokens = word_tokenize(clean_text, engine="newmm")
    except Exception:
        tokens = clean_text.split()

    vec = [0.0] * dim

    # 1. Word-level features
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        token_vec = _hash_token_to_embedding(token, dim)
        weight = 2.0 if any(c.isdigit() for c in token) else 1.0
        for i in range(dim):
            vec[i] += token_vec[i] * weight

    # 2. Subword / Character Trigram features (ช่วยเรื่อง Out-of-Vocabulary และคำสะกดติดกัน)
    for i in range(len(clean_text) - 2):
        trigram = clean_text[i : i + 3]
        t_vec = _hash_token_to_embedding(trigram, dim)
        for j in range(dim):
            vec[j] += t_vec[j] * 0.3

    # Normalize L2 norm = 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0.0:
        vec = [x / norm for x in vec]

    return vec


class EmbeddingModel:
    """Class จัดการ Embedding รองรับ BGE-M3 และ Fallback สำหรับ Enterprise RAG"""

    def __init__(self, model_name: str = "BAAI/bge-m3", dimension: int = BGE_M3_DIMENSION):
        self.model_name = model_name
        self.dimension = dimension
        self._st_model = None
        self._try_load_model()

    def _try_load_model(self) -> None:
        """พยายามโหลด sentence_transformers ถ้ามีติดตั้งในระบบ"""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
            self._st_model = SentenceTransformer(self.model_name)
        except Exception:
            self._st_model = None

    def embed_text(self, text: str) -> list[float]:
        """สร้าง Embedding เวกเตอร์สำหรับข้อความ 1 ก้อน"""
        if self._st_model is not None:
            try:
                emb = self._st_model.encode(text, normalize_embeddings=True)
                return emb.tolist()
            except Exception:
                pass
        return _dense_fallback_embed(text, self.dimension)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """สร้าง Embedding เวกเตอร์สำหรับข้อความหลายก้อนพร้อมกัน"""
        if self._st_model is not None:
            try:
                embs = self._st_model.encode(list(texts), normalize_embeddings=True)
                return [e.tolist() for e in embs]
            except Exception:
                pass
        return [_dense_fallback_embed(t, self.dimension) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        """สร้าง Embedding เวกเตอร์สำหรับ Query"""
        return self.embed_text(query)


# Default Singleton Instance สำหรับเรียกใช้งานทั่วไป
_default_embedder = EmbeddingModel()


def embed_text(text: str) -> list[float]:
    return _default_embedder.embed_text(text)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    return _default_embedder.embed_texts(texts)


def embed_query(query: str) -> list[float]:
    return _default_embedder.embed_query(query)
