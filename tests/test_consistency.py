"""[M6] Tests for Multi-Evidence Consistency & Conflict Detector"""

from __future__ import annotations

import pytest

from app.consistency import detect_evidence_conflicts
from app.vectorstore import Hit


def _make_hit(file_id: str, label: str, text: str) -> Hit:
    return Hit(
        score=0.9,
        file_id=file_id,
        filename=file_id,
        unit_index=0,
        unit_type="page",
        label=label,
        position={"page": 1},
        status="ok",
        method="text_layer",
        text=text,
        char_start=0,
        char_end=len(text),
        sentences=((0, len(text)),),
    )


class TestEvidenceConsistency:
    def test_no_conflict_when_facts_align(self):
        h1 = _make_hit("doc.pdf", "หน้า 5", "ในปี 2568 บริษัทมีกำไรสุทธิ 28,206 ล้านบาท")
        h2 = _make_hit("doc.pdf", "หน้า 145", "สำหรับปี 2568 บริษัทมีกำไรสุทธิ 28,206 ล้านบาท")

        report = detect_evidence_conflicts([h1, h2])
        assert report.has_conflict is False
        assert len(report.conflicts) == 0
        assert report.guidance_text == ""

    def test_detect_conflict_when_numbers_differ(self):
        # หน้า 5 ระบุงบรวม 28,206 ล้านบาท แต่หน้า 198 ระบุงบเฉพาะกิจการ 27,850 ล้านบาท
        h1 = _make_hit("doc.pdf", "หน้า 5", "ในปี 2568 บริษัทมีกำไรสุทธิ 28,206 ล้านบาท")
        h2 = _make_hit("doc.pdf", "หน้า 198", "ในปี 2568 บริษัทมีกำไรสุทธิ 27,850 ล้านบาท")

        report = detect_evidence_conflicts([h1, h2])
        assert report.has_conflict is True
        assert len(report.conflicts) == 1
        conflict = report.conflicts[0]
        assert conflict.metric == "กำไรสุทธิ"
        assert conflict.year == "2568"
        assert "28,206" in conflict.guidance
        assert "27,850" in conflict.guidance
        assert "[ข้อสังเกตความสอดคล้องของหลักฐาน" in report.guidance_text

