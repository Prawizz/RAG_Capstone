"""[M5-M6] Unit Tests for Reranker and Conformal Abstention Gate"""

from __future__ import annotations

import pytest

from app.rerank import (
    ConformalAbstentionGate,
    FALLBACK_ABSTAIN_MESSAGE,
    compute_lexical_alignment,
    rerank_hits,
)
from app.vectorstore import Hit


def _make_dummy_hit(
    text: str,
    score: float = 0.5,
    context: str = "",
    file_id: str = "doc1.pdf",
) -> Hit:
    return Hit(
        score=score,
        file_id=file_id,
        filename=file_id,
        unit_index=0,
        unit_type="page",
        label="หน้า 1",
        position={"page": 1},
        status="ok",
        method="direct",
        text=text,
        char_start=0,
        char_end=len(text),
        sentences=((0, len(text)),),
        context=context,
    )


class TestReranker:
    def test_compute_lexical_alignment_exact_match(self):
        query = "รายได้รวม 2568 CP all"
        doc = "ในปี 2568 บริษัท CP all มีรายได้รวม 1,022,143 ล้านบาท"
        score, matched = compute_lexical_alignment(query, doc)
        assert score >= 0.70
        assert "2568" in matched

    def test_compute_lexical_alignment_no_overlap(self):
        query = "สภาพอากาศ ประเทศญี่ปุ่น"
        doc = "ผลการดำเนินงานทางการเงินและงบกำไรขาดทุนของบริษัท"
        score, matched = compute_lexical_alignment(query, doc)
        assert score == 0.0
        assert len(matched) == 0

    def test_rerank_hits_reorders_by_relevance(self):
        query = "สาขาใหม่ เซเว่น อีเลฟเว่น"
        hit_irrelevant = _make_dummy_hit(
            "การกำกับดูแลกิจการที่ดีและความรับผิดชอบต่อสังคม",
            score=0.80,  # dense สูงหลอกๆ
        )
        hit_relevant = _make_dummy_hit(
            "บริษัทเปิดร้าน เซเว่น อีเลฟเว่น สาขาใหม่ เพิ่มขึ้น 700 สาขา",
            score=0.75,
        )

        reranked = rerank_hits(query, [hit_irrelevant, hit_relevant])
        assert len(reranked) == 2
        # hit_relevant ควรขึ้นเป็นอันดับ 1 หลัง rerank เพราะ lexical match สูงกว่า
        assert reranked[0].hit.text == hit_relevant.text
        assert reranked[0].rerank_score > reranked[1].rerank_score


class TestConformalAbstentionGate:
    def test_abstain_on_empty_query(self):
        gate = ConformalAbstentionGate()
        hits = [_make_dummy_hit("เนื้อหาปกติ")]
        reranked = rerank_hits("  ", hits)
        decision = gate.evaluate("  ", reranked)
        assert decision.should_abstain is True
        assert decision.reason == "EMPTY_QUERY"
        assert decision.fallback_message == FALLBACK_ABSTAIN_MESSAGE

    def test_abstain_on_empty_hits(self):
        gate = ConformalAbstentionGate()
        decision = gate.evaluate("คำถามอะไรสักอย่าง", [])
        assert decision.should_abstain is True
        assert decision.reason == "NO_EVIDENCE_FOUND"
        assert decision.fallback_message == FALLBACK_ABSTAIN_MESSAGE

    def test_abstain_on_low_confidence_unrelated_evidence(self):
        gate = ConformalAbstentionGate(min_confidence_threshold=0.30, min_lexical_overlap=0.10)
        query = "สูตรการคำนวณฟิสิกส์ควอนตัม"
        hits = [_make_dummy_hit("การจัดส่งสินค้าและคลังสินค้าในประเทศ", score=0.01)]
        reranked = rerank_hits(query, hits)
        decision = gate.evaluate(query, reranked)
        assert decision.should_abstain is True
        assert decision.reason == "INSUFFICIENT_EVIDENCE"

    def test_allow_answer_when_evidence_strong(self):
        gate = ConformalAbstentionGate(min_confidence_threshold=0.20, min_lexical_overlap=0.10)
        query = "รายได้รวมปี 2568"
        hits = [_make_dummy_hit("ในปี 2568 บริษัทมีรายได้รวมเพิ่มขึ้น", score=0.85)]
        reranked = rerank_hits(query, hits)
        decision = gate.evaluate(query, reranked)
        assert decision.should_abstain is False
        assert decision.reason is None

