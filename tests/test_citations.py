"""[M6] Tests for Sentence-level Citations & ALCE Verification
"""

from __future__ import annotations

import pytest

from app.citations import (
    build_citations_from_chunks,
    compute_nli_entailment_score,
    evaluate_citation_quality,
    parse_inline_citations,
)
from tests.test_retrieval import _make_dummy_child


class TestCitations:
    def test_build_citations_from_chunks(self):
        chunks = [
            _make_dummy_child("c1", "ประโยคแรกของการรายงาน", unit_index=0),
            _make_dummy_child("c2", "ประโยคที่สองของการรายงาน", unit_index=0),
        ]
        citation_map = build_citations_from_chunks(chunks)
        assert "S1" in citation_map
        assert "S2" in citation_map
        assert citation_map["S1"].snippet == "ประโยคแรกของการรายงาน"
        assert citation_map["S1"].label == "หน้า 1"

    def test_parse_inline_citations(self):
        text = "บริษัทมีรายได้เติบโตร้อยละ 11 [S1] และขยายสาขา 700 สาขา [S2][S3]"
        tags = parse_inline_citations(text)
        assert tags == ["S1", "S2", "S3"]

    def test_evaluate_citation_quality_full_recall(self):
        chunks = [
            _make_dummy_child("c1", "บริษัทมีรายได้รวม 1,022,143 ล้านบาท"),
        ]
        citation_map = build_citations_from_chunks(chunks)

        answer = "ในปี 2568 บริษัทมีรายได้รวม 1,022,143 ล้านบาท [S1]"
        result = evaluate_citation_quality(answer, citation_map)
        assert result.citation_recall == 1.0
        assert result.citations_used == ["S1"]
        assert len(result.unsupported_sentences) == 0

    def test_evaluate_citation_quality_missing_citation(self):
        chunks = [_make_dummy_child("c1", "ข้อมูล ก")]
        citation_map = build_citations_from_chunks(chunks)

        answer = "ประโยคที่หนึ่งที่มีการอ้างอิง [S1]\nประโยคที่สองที่ไม่มีการอ้างอิงใดๆ"
        result = evaluate_citation_quality(answer, citation_map)
        assert result.citation_recall == 0.5
        assert len(result.unsupported_sentences) == 1

    def test_nli_entailment_score_matching_facts(self):
        claim = "บริษัทมีรายได้รวม 1,022,143 ล้านบาท"
        evidence = "ในปีนี้ บริษัทฯ มีรายได้รวม 1,022,143 ล้านบาท และกำไรสุทธิ 28,206 ล้านบาท"
        score = compute_nli_entailment_score(claim, evidence)
        assert score >= 0.70

    def test_nli_entailment_score_contradiction_number(self):
        # ตัวเลขใน claim ไม่ตรงกับ evidence
        claim = "บริษัทมีรายได้รวม 999,999 ล้านบาท"
        evidence = "ในปีนี้ บริษัทฯ มีรายได้รวม 1,022,143 ล้านบาท"
        score = compute_nli_entailment_score(claim, evidence)
        assert score <= 0.20
