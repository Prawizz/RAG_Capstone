"""[M5-M6] Tests for Enterprise Thai Query Expansion and Salience Decomposition"""

from __future__ import annotations

import pytest

from app.query_expansion import (
    convert_buddhist_to_christian_year,
    expand_query,
)


class TestQueryExpansion:
    def test_buddhist_to_christian_year_conversion(self):
        assert convert_buddhist_to_christian_year("2568") == "2025"
        assert convert_buddhist_to_christian_year("2567") == "2024"
        assert convert_buddhist_to_christian_year("2025") == "2568"
        assert convert_buddhist_to_christian_year("9999") is None

    def test_expand_query_with_year_and_acronym(self):
        query = "ปี 2568 EBITDA ของ CP ALL และ 7-11 เป็นเท่าไหร่"
        expanded = expand_query(query)

        # ต้องขยายปี 2568 -> 2025
        assert "2568" in expanded.year_mappings
        assert expanded.year_mappings["2568"] == "2025"
        assert "2025" in expanded.expanded_text

        # ต้องขยายคำย่อ EBITDA
        assert "กำไรก่อนดอกเบี้ย" in expanded.expanded_text

        # ต้องแยก Core Intent ออกจาก Boilerplate
        assert "ebitda" in [t.lower() for t in expanded.salient_terms]
        # คำว่า 'ปี' ควรถูกจัดเป็น boilerplate
        assert "ปี" in expanded.boilerplate_terms

    def test_expand_query_empty_or_whitespace(self):
        expanded = expand_query("   ")
        assert expanded.original_query == ""
        assert len(expanded.canonical_terms) == 0

