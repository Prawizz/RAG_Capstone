"""embedding: ไม่ fallback เงียบ · hash vector เฉพาะโหมด test ที่ระบุชัด · sparse (bm25 / BGE) · BGE-M3 จริง (mark slow)"""

import math
import subprocess
import sys
from pathlib import Path

import pytest

from app.config import Settings, build_settings
from app.embedding import (
    BGE_M3_DIMENSION,
    BGEM3Embedder,
    EmbeddingError,
    EmbeddingModel,
    EmbeddingUnavailable,
    HashEmbedder,
    SparseVector,
    bm25_sparse,
    cosine_similarity,
    get_embedder,
    lexical_tokens,
    token_index,
)

ROOT = Path(__file__).parent.parent
MISSING_MODEL = str(ROOT / "ไม่มีโมเดลนี้")  # path ที่ไม่มีอยู่จริง -> โหลดไม่ได้เร็ว ไม่แตะเครือข่าย


# ---------- ไม่ fallback เงียบ ----------


def test_real_embedder_that_cannot_load_raises_instead_of_falling_back(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(EmbeddingUnavailable):
        get_embedder(Settings(embedding_model=MISSING_MODEL, embedding_test_mode=False))
    with pytest.raises(EmbeddingUnavailable):
        BGEM3Embedder(MISSING_MODEL)


def test_embedding_model_wrapper_also_raises_when_real_model_is_unavailable(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = EmbeddingModel(settings=Settings(embedding_model=MISSING_MODEL, embedding_test_mode=False))
    with pytest.raises(EmbeddingUnavailable):
        model.embed_text("ข้อความ")


def test_hash_embedder_only_when_test_mode_is_explicit():
    assert isinstance(get_embedder(Settings(embedding_test_mode=True)), HashEmbedder)
    assert isinstance(get_embedder(Settings(embedding_model=MISSING_MODEL), test_mode=True), HashEmbedder)
    assert Settings().embedding_test_mode is False  # ค่าเริ่มต้นปิด


def test_test_mode_is_read_from_the_environment_setting(tmp_path, monkeypatch):
    monkeypatch.delenv("EMBEDDING_TEST_MODE", raising=False)
    assert build_settings(tmp_path / "ไม่มี.env").embedding_test_mode is False
    monkeypatch.setenv("EMBEDDING_TEST_MODE", "1")
    assert build_settings(tmp_path / "ไม่มี.env").embedding_test_mode is True


def test_importing_the_module_loads_no_model():
    """prototype เดิมโหลดโมเดลตอน import -> ช้า/กิน VRAM โดยไม่รู้ตัว"""
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); import app.embedding; "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, env={"SYSTEMROOT": "C:\\Windows"})


# ---------- sparse ----------


def test_sparse_vector_is_sorted_unique_and_positive():
    sv = SparseVector.from_weights({5: 0.5, 2: 1.0, 9: 0.0, 7: -1.0})
    assert sv.indices == (2, 5) and sv.values == (1.0, 0.5)
    with pytest.raises(ValueError):
        SparseVector((1, 2), (1.0,))
    assert not SparseVector() and sv


def test_lexical_tokens_use_the_normalizer_and_drop_punctuation_and_case():
    assert lexical_tokens("ราคา  100.5 บาท !!") == lexical_tokens("ราคา 100.5 บาท")
    assert "abc" in lexical_tokens("ABC กำไร")
    assert lexical_tokens("   ") == [] and lexical_tokens("--- ***") == []
    assert lexical_tokens("กํา") == lexical_tokens("กำ")  # normalizer กลาง: ํา -> ำ


def test_bm25_sparse_counts_terms_with_stable_indices():
    sv = bm25_sparse("กำไร กำไร รายได้")
    weights = dict(zip(sv.indices, sv.values))
    assert weights[token_index("กำไร")] == 2.0 and weights[token_index("รายได้")] == 1.0
    assert all(0 <= i < 2**32 for i in sv.indices)
    assert bm25_sparse("กำไร รายได้ กำไร") == sv  # ลำดับคำไม่มีผล


def test_token_index_is_stable_across_processes():
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
            "from app.embedding import token_index; print(token_index('\\u0e01\\u0e33\\u0e44\\u0e23'))")  # กำไร
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True,
                         env={"PYTHONHASHSEED": "random", "PYTHONIOENCODING": "utf-8", "SYSTEMROOT": "C:\\Windows"})
    assert int(out.stdout) == token_index("กำไร")


# ---------- hash embedder (test เท่านั้น) ----------


def test_hash_embedder_is_deterministic_normalized_and_has_both_parts():
    e = HashEmbedder(64)
    a, b = e.embed_query("รายได้รวม 1,022 ล้านบาท"), e.embed_query("รายได้รวม 1,022 ล้านบาท")
    assert a == b and len(a.dense) == 64 and e.dense_size == 64
    assert math.isclose(sum(x * x for x in a.dense), 1.0, rel_tol=1e-9)
    assert a.sparse == bm25_sparse("รายได้รวม 1,022 ล้านบาท")
    assert e.embed_documents(["ก", "ข"]) == [e.embed_query("ก"), e.embed_query("ข")]


def test_hash_embedder_default_size_matches_bge_m3():
    assert len(HashEmbedder().embed_query("ทดสอบ").dense) == BGE_M3_DIMENSION


def test_empty_text_gives_zero_dense_and_empty_sparse():
    e = HashEmbedder(16).embed_query("  ")
    assert set(e.dense) == {0.0} and not e.sparse


# ---------- BGE-M3 จริง (ต้องมีโมเดลในเครื่อง + GPU/CPU ช้า) ----------


@pytest.mark.slow
def test_bge_m3_returns_dense_and_sparse_and_releases_vram():
    import torch

    e = BGEM3Embedder()
    try:
        if torch.cuda.is_available():
            assert e.dtype == torch.float16  # fp16 บน GPU
        q = e.embed_query("รายได้รวมของ ซีพี ออลล์ ปี 2568")
        docs = e.embed_documents(["บริษัทฯ มีรายได้รวม 1,022,143 ล้านบาท", "นโยบายวันลาพักร้อนของพนักงานประจำ"])
        assert len(q.dense) == BGE_M3_DIMENSION and math.isclose(sum(x * x for x in q.dense), 1.0, rel_tol=1e-2)
        assert q.sparse and all(v > 0 for v in q.sparse.values) and list(q.sparse.indices) == sorted(set(q.sparse.indices))
        assert cosine_similarity(q.dense, docs[0].dense) > cosine_similarity(q.dense, docs[1].dense)
        # ผลคืนตามลำดับอินพุตแม้ข้างในเรียงตามความยาว และ padding ไม่เปลี่ยนค่า dense (fp16 จึงเผื่อคลาดเล็กน้อย)
        assert e.embed_documents(["สั้น", "ยาวกว่านั้นมาก" * 20])[0].dense == pytest.approx(e.embed_query("สั้น").dense, abs=2e-3)
    finally:
        e.release()
    if torch.cuda.is_available():
        assert torch.cuda.memory_allocated() < 100 * 2**20  # ปล่อยโมเดล (~1 GB) แล้ว


# ---------- fp16 ได้ NaN: แถวที่ไม่ finite ต้องถูกคำนวณใหม่ ไม่หลุดเข้า index ----------


def _vec(tag: float) -> "Embedded":
    from app.embedding import Embedded

    return Embedded((tag, 0.0), SparseVector((1,), (tag,)))


class _ScriptedBGE(BGEM3Embedder):
    """BGEM3Embedder ที่ไม่โหลดโมเดล: ``_encode`` ตอบตามสคริปต์ว่าแถวไหนล้มใน fp16 / fp32"""

    def __init__(self, bad_in_batch=(), bad_single_fp16=(), bad_fp32=()):
        self.fp16_retries = self.fp32_fallbacks = 0
        self.batch_size = 16
        self.bad_in_batch, self.bad_single_fp16, self.bad_fp32 = set(bad_in_batch), set(bad_single_fp16), set(bad_fp32)
        self.in_fp32 = False
        self.batches = []

    from contextlib import contextmanager as _cm

    @_cm
    def _fp32(self):
        self.in_fp32 = True
        try:
            yield
        finally:
            self.in_fp32 = False

    def _encode(self, texts):
        self.batches.append(list(texts))
        out = []
        for t in texts:
            if self.in_fp32:
                bad = t in self.bad_fp32
            elif len(texts) == 1:
                bad = t in self.bad_single_fp16
            else:
                bad = t in self.bad_in_batch
            out.append(None if bad else _vec(float(len(t))))
        return out


def test_non_finite_row_is_recomputed_alone_in_fp16():
    e = _ScriptedBGE(bad_in_batch={"bb"})
    out = e.embed_documents(["a", "bb", "ccc"])
    assert [o.dense[0] for o in out] == [1.0, 2.0, 3.0]  # ผลคืนตามลำดับอินพุต
    assert (e.fp16_retries, e.fp32_fallbacks) == (1, 0)
    assert ["bb"] in e.batches


def test_row_that_fails_alone_falls_back_to_fp32():
    e = _ScriptedBGE(bad_in_batch={"bb"}, bad_single_fp16={"bb"})
    out = e.embed_documents(["a", "bb"])
    assert out[1].dense[0] == 2.0 and (e.fp16_retries, e.fp32_fallbacks) == (1, 1)
    assert not e.in_fp32  # คืนสถานะ fp16 แล้ว


def test_row_that_fails_everywhere_raises_instead_of_returning_nan():
    e = _ScriptedBGE(bad_in_batch={"bb"}, bad_single_fp16={"bb"}, bad_fp32={"bb"})
    with pytest.raises(EmbeddingError):
        e.embed_documents(["a", "bb"])
    with pytest.raises(EmbeddingError):
        _ScriptedBGE(bad_single_fp16={"q"}, bad_fp32={"q"}).embed_query("q")


def test_clean_batches_do_not_retry():
    e = _ScriptedBGE()
    e.embed_documents(["a", "bb", "ccc"])
    assert (e.fp16_retries, e.fp32_fallbacks) == (0, 0) and len(e.batches) == 1
