"""[M4-M6] Comprehensive Tests for Thai Hybrid Search (BM25 + BGE-M3 + RRF) and Abstention Gate
"""

from __future__ import annotations

import pytest

from app.chunking import ChildChunk, ParentChunk
from app.embedding import (
    BGE_M3_DIMENSION,
    EmbeddingModel,
    cosine_similarity,
    embed_query,
    embed_text,
)
from app.retrieval import (
    FALLBACK_ABSTAIN_RESPONSE,
    RRF_DEFAULT_K,
    TIER1_CONFORMAL_THRESHOLD,
    DenseVectorIndex,
    HybridRetriever,
    MultiTieredAbstentionGate,
    ThaiBM25Index,
    reciprocal_rank_fusion,
    relative_score_fusion,
)


def _make_dummy_child(
    chunk_id: str,
    content: str,
    parent_id: str = "p-1",
    unit_index: int = 0,
    sentence_index: int = 0,
) -> ChildChunk:
    return ChildChunk(
        chunk_id=chunk_id,
        parent_id=parent_id,
        document_id="doc-test.pdf",
        unit_index=unit_index,
        sentence_index=sentence_index,
        content=content,
        char_start=0,
        char_end=len(content),
        position={"page": unit_index + 1},
        label=f"หน้า {unit_index + 1}",
    )


# -------------------------------------------------------------
# 1. Embedding Tests (BGE-M3 1024 dim)
# -------------------------------------------------------------
class TestEmbeddingModel:
    def test_dimension_is_1024(self):
        vec = embed_text("บริษัท ซีพี ออลล์ จำกัด (มหาชน)")
        assert len(vec) == BGE_M3_DIMENSION

    def test_cosine_similarity_identical_text(self):
        v1 = embed_text("รายได้รวม 1,022,143 ล้านบาท")
        v2 = embed_text("รายได้รวม 1,022,143 ล้านบาท")
        sim = cosine_similarity(v1, v2)
        assert pytest.approx(sim, abs=1e-4) == 1.0

    def test_cosine_similarity_different_text(self):
        v1 = embed_text("รายได้รวมและผลประกอบการทางการเงิน")
        v2 = embed_text("นโยบายวันลาพักร้อนของพนักงานประจำ")
        sim = cosine_similarity(v1, v2)
        assert sim < 0.95

    def test_embed_empty_text(self):
        vec = embed_text("")
        assert len(vec) == BGE_M3_DIMENSION


# -------------------------------------------------------------
# 2. Thai BM25 Index Tests
# -------------------------------------------------------------
class TestThaiBM25Index:
    def test_indexing_and_search(self):
        chunks = [
            _make_dummy_child("c1", "บริษัท ซีพี ออลล์ มีรายได้รวม 1,022,143 ล้านบาท"),
            _make_dummy_child("c2", "การขยายสาขาใหม่ในไทยเพิ่ม 700 สาขา"),
            _make_dummy_child("c3", "พนักงานมีสิทธิลาพักร้อนได้ 10 วันทำการ"),
        ]
        index = ThaiBM25Index()
        index.index(chunks)

        results = index.search("รายได้รวม ซีพี ออลล์", top_k=2)
        assert len(results) > 0
        best_chunk, score, rank = results[0]
        assert best_chunk.chunk_id == "c1"
        assert rank == 1
        assert score > 0.0

    def test_search_no_match(self):
        chunks = [_make_dummy_child("c1", "ข้อมูลทางธุรกิจ")]
        index = ThaiBM25Index()
        index.index(chunks)
        results = index.search("คำที่ไม่มีในระบบอย่างแน่นอนxyz", top_k=5)
        assert len(results) == 0


# -------------------------------------------------------------
# 3. Dense Vector Index Tests
# -------------------------------------------------------------
class TestDenseVectorIndex:
    def test_dense_search(self):
        chunks = [
            _make_dummy_child("c1", "ผลการดำเนินงานทางการเงินและกำไรสุทธิ"),
            _make_dummy_child("c2", "นโยบายการลาป่วยและสิทธิประโยชน์"),
        ]
        index = DenseVectorIndex()
        index.index(chunks)

        results = index.search("กำไรสุทธิทางการเงิน", top_k=2)
        assert len(results) == 2
        best_chunk, sim, rank = results[0]
        assert best_chunk.chunk_id == "c1"
        assert rank == 1
        assert sim > 0.0


# -------------------------------------------------------------
# 4. Fusion Tests (RRF vs RSF)
# -------------------------------------------------------------
class TestFusionTechniques:
    def test_reciprocal_rank_fusion(self):
        c1 = _make_dummy_child("c1", "chunk 1")
        c2 = _make_dummy_child("c2", "chunk 2")

        # Dense rank: c1=1, c2=2
        dense_res = [(c1, 0.9, 1), (c2, 0.7, 2)]
        # BM25 rank: c2=1, c1=2
        bm25_res = [(c2, 12.0, 1), (c1, 8.0, 2)]

        rrf_scores = reciprocal_rank_fusion(dense_res, bm25_res, k=60)
        # S_RRF(c1) = 1/(60+1) + 1/(60+2) = 1/61 + 1/62
        expected_c1 = (1.0 / 61.0) + (1.0 / 62.0)
        assert pytest.approx(rrf_scores["c1"], rel=1e-4) == expected_c1
        assert pytest.approx(rrf_scores["c2"], rel=1e-4) == expected_c1

    def test_relative_score_fusion_rsf(self):
        c1 = _make_dummy_child("c1", "chunk 1")
        c2 = _make_dummy_child("c2", "chunk 2")

        dense_res = [(c1, 1.0, 1), (c2, 0.0, 2)]
        bm25_res = [(c1, 10.0, 1), (c2, 0.0, 2)]

        rsf_scores = relative_score_fusion(dense_res, bm25_res, alpha=0.65)
        # c1 มี score สูงสุดทั้งสองฝั่ง -> 0.65*1.0 + 0.35*1.0 = 1.0
        assert pytest.approx(rsf_scores["c1"], abs=1e-3) == 1.0
        assert pytest.approx(rsf_scores["c2"], abs=1e-3) == 0.0


# -------------------------------------------------------------
# 5. Multi-Tiered Abstention Gate Tests
# -------------------------------------------------------------
class TestAbstentionGate:
    def test_tier1_fast_abstain_on_low_score(self):
        gate = MultiTieredAbstentionGate(conformal_threshold=0.05)
        decision = gate.check_tier1_retrieval(top_score=0.01)
        assert decision.should_abstain is True
        assert decision.tier_failed == "Tier1_Retrieval"
        assert decision.message == FALLBACK_ABSTAIN_RESPONSE

    def test_tier1_passes_when_score_sufficient(self):
        gate = MultiTieredAbstentionGate(conformal_threshold=0.02)
        decision = gate.check_tier1_retrieval(top_score=0.08)
        assert decision.should_abstain is False

    def test_tier2_rejects_high_entropy(self):
        gate = MultiTieredAbstentionGate(entropy_threshold=0.50)
        decision = gate.check_tier2_uncertainty(semantic_entropy=0.75)
        assert decision.should_abstain is True
        assert decision.tier_failed == "Tier2_Uncertainty"

    def test_tier3_rejects_hallucination_grounding(self):
        gate = MultiTieredAbstentionGate(grounding_threshold=1.0)
        decision = gate.check_tier3_grounding(grounding_ratio=0.66)
        assert decision.should_abstain is True
        assert decision.tier_failed == "Tier3_Entailment"


# -------------------------------------------------------------
# 6. Hybrid Retriever Integration Tests
# -------------------------------------------------------------
class TestHybridRetrieverIntegration:
    def test_full_pipeline_rrf_and_parent_resolution(self):
        children = [
            _make_dummy_child("c1", "บริษัทมีรายได้รวม 1,022,143 ล้านบาท", parent_id="p1"),
            _make_dummy_child("c2", "กำไรสุทธิประจำปี 28,206 ล้านบาท", parent_id="p1"),
            _make_dummy_child("c3", "ขยายสาขาใหม่เพิ่มอีก 700 สาขา", parent_id="p2"),
        ]
        parents = [
            ParentChunk(
                chunk_id="p1",
                document_id="doc-test.pdf",
                unit_index=0,
                content="บริษัทมีรายได้รวม 1,022,143 ล้านบาท กำไรสุทธิประจำปี 28,206 ล้านบาท",
                char_start=0,
                char_end=80,
                child_ids=["c1", "c2"],
                sentence_count=2,
                position={"page": 1},
                label="หน้า 1",
            ),
            ParentChunk(
                chunk_id="p2",
                document_id="doc-test.pdf",
                unit_index=0,
                content="ขยายสาขาใหม่เพิ่มอีก 700 สาขา ทั่วประเทศ",
                char_start=81,
                char_end=130,
                child_ids=["c3"],
                sentence_count=1,
                position={"page": 1},
                label="หน้า 1",
            ),
        ]

        retriever = HybridRetriever(rrf_k=60)
        retriever.index(children, parents)

        results, decision = retriever.retrieve("รายได้รวมและกำไรสุทธิ", top_k=2)
        assert decision.should_abstain is False
        assert len(results) > 0

        # ตรวจสอบว่า c1 หรือ c2 ติดอันดับแรก และ resolve parent_id = p1 ได้ถูกต้อง
        top_res = results[0]
        assert top_res.child.chunk_id in ["c1", "c2"]
        assert top_res.parent is not None
        assert top_res.parent.chunk_id == "p1"
        assert "1,022,143" in top_res.parent.content

        # ตรวจสอบการเตรียม Payload สำหรับ LLM
        payload = retriever.prepare_llm_payload(results)
        assert "citation_context" in payload
        assert "parent_contexts" in payload
        assert len(payload["parent_contexts"]) >= 1
        assert payload["citation_context"][0]["sentence_id"] == "S1"

    def test_abstains_on_irrelevant_query(self):
        children = [
            _make_dummy_child("c1", "การเงินและการบัญชีบริษัท"),
        ]
        retriever = HybridRetriever(
            abstention_gate=MultiTieredAbstentionGate(conformal_threshold=0.05)
        )
        retriever.index(children)

        results, decision = retriever.retrieve("อัตราดอกเบี้ยสินเชื่อส่วนบุคคลกู้ซื้อบ้าน")
        assert decision.should_abstain is True
        assert decision.message == FALLBACK_ABSTAIN_RESPONSE
