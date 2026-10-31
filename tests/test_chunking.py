"""[M3] Tests สำหรับ Thai-Aware Chunking

ครอบคลุม:
- segment_sentences (SBD ทั้งสองเส้นทาง)
- chunk_unit (child + parent สร้างถูก)
- offset integrity (char_start/char_end ชี้ถูกต้อง)
- parent-child relationship (FK ถูกต้อง, sliding window)
- unit ที่ status != ok ถูกข้าม
- ไม่ข้ามขอบเขต unit
- format_citation_context (S1/S2 tag)
- dataset smoke test: ลองกับ expected_citation จาก rag_evaluation_1000.jsonl
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

import pytest

from app.chunking import (
    ChildChunk,
    ParentChunk,
    ChunkResult,
    PARENT_WINDOW_MAX,
    PARENT_OVERLAP,
    MIN_SENT_CHARS,
    MAX_CHILD_CHARS,
    _merge_short_sentences,
    _split_by_newlines_and_punct,
    _truncate_at_grapheme,
    _find_offsets_in_original,
    chunk_unit,
    chunk_document,
    format_citation_context,
    segment_sentences,
    get_parent_for_child,
    build_system_prompt,
)
from app.models import (
    Block,
    BlockType,
    ExtractionMethod,
    QualityReport,
    Unit,
    UnitStatus,
    UnitType,
)
from app.thai.normalizer import normalize

# ---------- helpers ----------

EVAL_PATH = Path(__file__).parent.parent / "eval" / "datasets" / "rag_evaluation_1000.jsonl"


def _make_unit(
    text: str,
    index: int = 0,
    status: UnitStatus = UnitStatus.OK,
    unit_type: UnitType = UnitType.PAGE,
) -> Unit:
    """สร้าง Unit ตัวอย่างสำหรับ test"""
    import hashlib

    text_hash = hashlib.sha256(text.encode()).hexdigest()
    flags = []
    if not text.strip() and status == UnitStatus.OK:
        from app.models import QualityFlag
        flags = [QualityFlag.EMPTY]

    if unit_type == UnitType.PAGE:
        position: dict[str, Any] = {"page": index + 1}
        label = f"หน้า {index + 1}"
    elif unit_type == UnitType.SLIDE:
        position = {"slide": index + 1}
        label = f"สไลด์ {index + 1}"
    elif unit_type == UnitType.SECTION:
        position = {"heading_path": ["หัวข้อ"], "paragraphs": [1, 2]}
        label = "หัวข้อ"
    else:
        position = {"sheet": "Sheet1", "range": "A1:Z10"}
        label = "Sheet1"

    reason = None if status == UnitStatus.OK else "test_skip"

    return Unit(
        index=index,
        type=unit_type,
        position=position,
        label=label,
        status=status,
        reason=reason,
        method=ExtractionMethod.TEXT_LAYER,
        blocks=[
            Block(type=BlockType.PARAGRAPH, text=text, order=0)
        ] if text.strip() else [],
        text=text,
        text_hash=text_hash,
        quality=QualityReport(flags=flags),
        signals={"char_count": float(len(text))},
    )


# ---------- segment_sentences ----------

class TestSegmentSentences:
    def test_empty_string_returns_empty(self):
        assert segment_sentences("") == []

    def test_whitespace_only_returns_empty(self):
        assert segment_sentences("   \n  ") == []

    def test_single_sentence_thai(self):
        text = "บริษัทมีรายได้รวม 1,022,143 ล้านบาท"
        result = segment_sentences(text)
        assert len(result) >= 1
        # ข้อความทั้งหมดต้องถูกเก็บไว้
        combined = " ".join(result)
        assert "1,022,143" in combined

    def test_newline_splits_sentences(self):
        text = "ประโยคแรกมีความยาวพอสมควร\nประโยคที่สองมีความยาวพอสมควรเช่นกัน"
        result = segment_sentences(text)
        assert len(result) >= 2

    def test_multiple_newlines_split(self):
        text = "หัวเรื่อง\n\nเนื้อหาย่อหน้าแรกที่มีความยาวพอสมควร\n\nเนื้อหาย่อหน้าที่สองที่มีความยาวพอสมควร"
        result = segment_sentences(text)
        assert len(result) >= 2

    def test_merge_short_below_min(self):
        # ชิ้นที่สั้นกว่า MIN_SENT_CHARS ต้อง merge
        short = "สั้น"
        long_sent = "ประโยคยาวๆ ที่มีความหมายชัดเจนและยาวพอสมควร"
        result = _merge_short_sentences([short, long_sent])
        # short ต้อง merge เข้า long
        assert len(result) >= 1
        combined = " ".join(result)
        assert "สั้น" in combined

    def test_long_text_from_dataset_splits(self):
        """ทดสอบด้วยข้อความจาก expected_citation ใน dataset"""
        text = (
            "โดยในปี 2568 บริษัทฯ ขยายสาขาใหม่ในไทยเพิ่มอีก 700 สาขา "
            "ทำให้สิ้นปีมีสาขารวมทั้งสิ้น 15,945 สาขา "
            "และร้อยละ 90 ของร้านเปิด"
        )
        result = segment_sentences(text)
        assert len(result) >= 1
        # ข้อมูลสำคัญต้องไม่หาย
        combined = " ".join(result)
        assert "700" in combined
        assert "15,945" in combined


class TestMergeShortSentences:
    def test_all_long_sentences_unchanged(self):
        sents = [
            "ประโยคยาวพอสมควรที่ไม่ควร merge",
            "ประโยคยาวพอสมควรที่ไม่ควร merge เช่นกัน",
        ]
        result = _merge_short_sentences(sents)
        assert len(result) == 2

    def test_single_short_sentence(self):
        result = _merge_short_sentences(["สั้น"])
        assert len(result) == 1
        assert result[0] == "สั้น"

    def test_empty_list(self):
        assert _merge_short_sentences([]) == []

    def test_short_merges_with_next(self):
        result = _merge_short_sentences(["abc", "ประโยคยาวพอสมควร"])
        assert len(result) == 1
        assert "abc" in result[0]


class TestTruncateAtGrapheme:
    def test_ascii_truncation(self):
        head, tail = _truncate_at_grapheme("hello world", 5)
        assert head == "hello"
        assert tail == " world"

    def test_thai_truncation_does_not_split_cluster(self):
        # "ก้" = ก + ้ เป็น 1 grapheme cluster
        text = "กา" * 50  # 100 chars
        head, tail = _truncate_at_grapheme(text, 10)
        assert len(head) <= 10
        assert head + tail == text

    def test_empty_string(self):
        head, tail = _truncate_at_grapheme("", 10)
        assert head == ""
        assert tail == ""


class TestFindOffsetsInOriginal:
    def test_simple_two_sentences(self):
        original = "ประโยคแรก ประโยคที่สอง"
        sentences = ["ประโยคแรก", "ประโยคที่สอง"]
        offsets = _find_offsets_in_original(sentences, original)
        assert len(offsets) == 2
        assert offsets[0][0] == 0
        assert original[offsets[0][0]:offsets[0][1]] == "ประโยคแรก"
        assert original[offsets[1][0]:offsets[1][1]] == "ประโยคที่สอง"

    def test_not_found_returns_approximate(self):
        original = "ข้อความต้นฉบับ"
        sentences = ["ข้อความที่ normalize แล้วต่างออกไป"]
        offsets = _find_offsets_in_original(sentences, original)
        assert len(offsets) == 1
        start, end = offsets[0]
        assert 0 <= start <= end <= len(original)


# ---------- chunk_unit ----------

class TestChunkUnit:
    def test_skips_failed_unit(self):
        unit = _make_unit("ข้อความบางอย่าง", status=UnitStatus.FAILED)
        result = chunk_unit(unit, "doc-1")
        assert result.children == []
        assert result.parents == []
        assert result.skipped_reason is not None

    def test_skips_needs_ocr_unit(self):
        unit = _make_unit("ข้อความ", status=UnitStatus.NEEDS_OCR)
        result = chunk_unit(unit, "doc-1")
        assert result.children == []
        assert result.skipped_reason is not None

    def test_basic_thai_paragraph(self):
        """ข้อความไทยปกติต้องได้ child อย่างน้อย 1 ตัว"""
        text = (
            "บริษัท ซีพี ออลล์ จำกัด (มหาชน) มีรายได้รวม 1,022,143 ล้านบาท\n"
            "และกำไรสุทธิ 28,206 ล้านบาท เพิ่มขึ้นจากปีก่อนหน้าร้อยละ 11\n"
            "โดยในปี 2568 บริษัทฯ ขยายสาขาใหม่ในไทยเพิ่มอีก 700 สาขา"
        )
        unit = _make_unit(text)
        result = chunk_unit(unit, "cpall-doc")
        assert len(result.children) >= 1
        assert len(result.parents) >= 1

    def test_child_parent_fk_consistency(self):
        """child.parent_id ต้องชี้ไปยัง parent ที่มีจริง"""
        text = "\n".join(
            [f"ประโยคที่ {i+1} มีความยาวพอสมควรสำหรับการทดสอบระบบ chunk" for i in range(10)]
        )
        unit = _make_unit(text)
        result = chunk_unit(unit, "doc-test")

        parent_ids = {p.chunk_id for p in result.parents}
        for child in result.children:
            assert child.parent_id in parent_ids, (
                f"child {child.chunk_id} มี parent_id={child.parent_id} ที่ไม่มีใน parents"
            )

    def test_child_ids_in_parent_match_children(self):
        """child_ids ใน parent ต้องตรงกับ chunk_id ของ children"""
        text = "\n".join(
            [f"ประโยคที่ {i+1} มีความยาวพอสมควรสำหรับการทดสอบระบบ chunk" for i in range(8)]
        )
        unit = _make_unit(text)
        result = chunk_unit(unit, "doc-test")

        child_ids = {c.chunk_id for c in result.children}
        for parent in result.parents:
            for cid in parent.child_ids:
                assert cid in child_ids, (
                    f"parent {parent.chunk_id} อ้าง child_id={cid} ที่ไม่มี"
                )

    def test_char_offsets_in_bounds(self):
        """char_start / char_end ต้องอยู่ในขอบเขต Unit.text"""
        text = (
            "วิสัยทัศน์ของซีพี ออลล์คือ เราให้บริการความสะดวกกับทุกชุมชน\n"
            "บริษัทฯ ได้ขยายการให้บริการด้านสุขภาพผ่านการเปิดร้านจำหน่ายยา\n"
            "ภายใต้ชื่อ All Care Pharmacy และให้บริการ Mor Dee"
        )
        unit = _make_unit(text)
        result = chunk_unit(unit, "doc-test")

        for child in result.children:
            assert 0 <= child.char_start < child.char_end <= len(text), (
                f"offset ของ child {child.chunk_id} อยู่นอกขอบ: "
                f"[{child.char_start}, {child.char_end}] vs len={len(text)}"
            )

    def test_parent_char_start_le_child_char_start(self):
        """parent.char_start <= char_start ของ child แรก"""
        text = "\n".join(
            [f"ประโยคที่ {i+1} ที่มีเนื้อหาพอสมควร" for i in range(6)]
        )
        unit = _make_unit(text)
        result = chunk_unit(unit, "doc-test")

        child_by_id = {c.chunk_id: c for c in result.children}
        for parent in result.parents:
            first_child = child_by_id[parent.child_ids[0]]
            assert parent.char_start <= first_child.char_start

    def test_sentence_index_sequential(self):
        """sentence_index ของ children ใน unit เดียวกันต้องเรียงจาก 0"""
        text = "\n".join(
            [f"ประโยคที่ {i+1} ที่มีความยาวพอสมควร" for i in range(5)]
        )
        unit = _make_unit(text)
        result = chunk_unit(unit, "doc-test")

        indices = [c.sentence_index for c in result.children]
        assert indices == list(range(len(indices)))

    def test_child_content_not_exceed_max_chars(self):
        """ทุก child ต้องมีความยาวไม่เกิน MAX_CHILD_CHARS"""
        # สร้างประโยคยาวมากๆ
        long_text = "ก" * 2000
        unit = _make_unit(long_text)
        result = chunk_unit(unit, "doc-test")
        for child in result.children:
            assert len(child.content) <= MAX_CHILD_CHARS, (
                f"child content ยาวเกิน: {len(child.content)} > {MAX_CHILD_CHARS}"
            )

    def test_parent_window_size(self):
        """parent แต่ละตัวต้องมี sentence ไม่เกิน PARENT_WINDOW_MAX"""
        text = "\n".join(
            [f"ประโยคที่ {i+1} ที่มีความยาวพอสมควรสำหรับ test" for i in range(20)]
        )
        unit = _make_unit(text)
        result = chunk_unit(unit, "doc-test", parent_window=PARENT_WINDOW_MAX)
        for parent in result.parents:
            assert parent.sentence_count <= PARENT_WINDOW_MAX

    def test_document_id_propagated(self):
        """document_id ต้องถูก propagate ไปยังทุก child และ parent"""
        text = "ข้อความทดสอบที่มีความยาวพอสมควร\nบรรทัดที่สอง"
        unit = _make_unit(text)
        doc_id = "my-unique-doc-id"
        result = chunk_unit(unit, doc_id)
        for child in result.children:
            assert child.document_id == doc_id
        for parent in result.parents:
            assert parent.document_id == doc_id

    def test_position_propagated(self):
        """position และ label ต้องถูก propagate จาก unit"""
        text = "ข้อความทดสอบ\nที่มีหลายบรรทัด"
        unit = _make_unit(text, index=4)  # หน้า 5
        result = chunk_unit(unit, "doc-test")
        for child in result.children:
            assert child.position == {"page": 5}
            assert child.label == "หน้า 5"

    def test_noisy_whitespace_tolerance(self):
        """ข้อความที่มี whitespace แปลกๆ (จาก PDF) ต้องยังตัดได้"""
        # Noisy whitespace: ช่องว่างหลายตัวติดกัน, NBSP
        text = "บริษัท  ซีพี  ออลล์\u00a0จำกัด  มีรายได้\nกำไรสุทธิ  28,206  ล้านบาท"
        unit = _make_unit(text)
        result = chunk_unit(unit, "doc-noisy")
        assert len(result.children) >= 1
        # ข้อมูลสำคัญต้องไม่หาย
        combined = " ".join(c.content for c in result.children)
        assert "28,206" in combined

    def test_single_short_text(self):
        """ข้อความสั้นๆ ต้องได้ child อย่างน้อย 1 ตัว"""
        text = "ราคา 100 บาท"
        unit = _make_unit(text)
        result = chunk_unit(unit, "doc-short")
        assert len(result.children) >= 1

    def test_unit_index_propagated(self):
        """unit_index ต้องตรงกับ Unit.index"""
        text = "ข้อความทดสอบ\nบรรทัดสอง"
        unit = _make_unit(text, index=7)
        result = chunk_unit(unit, "doc-test")
        for child in result.children:
            assert child.unit_index == 7
        for parent in result.parents:
            assert parent.unit_index == 7


# ---------- chunk_document ----------

class TestChunkDocument:
    def test_skips_non_ok_units(self):
        units = [
            _make_unit("ข้อความหน้า 1\nบรรทัดสอง", index=0),
            _make_unit("ข้อความหน้า 2", index=1, status=UnitStatus.FAILED),
            _make_unit("ข้อความหน้า 3\nบรรทัดสอง", index=2),
        ]
        children, parents = chunk_document(units, "doc-multi")
        # unit 1 (FAILED) ต้องไม่มี children
        unit_indices = {c.unit_index for c in children}
        assert 1 not in unit_indices
        assert 0 in unit_indices
        assert 2 in unit_indices

    def test_no_cross_unit_chunks(self):
        """children จาก unit ต่างๆ ต้องไม่ปะปนกัน"""
        units = [
            _make_unit("ข้อความหน้า 1\nบรรทัดที่สองของหน้าหนึ่ง", index=0),
            _make_unit("ข้อความหน้า 2\nบรรทัดที่สองของหน้าสอง", index=1),
        ]
        children, parents = chunk_document(units, "doc-multi")

        # parent ต้องอยู่ใน unit เดียวกับ children ทั้งหมด
        child_by_id = {c.chunk_id: c for c in children}
        for parent in parents:
            parent_unit = parent.unit_index
            for cid in parent.child_ids:
                child = child_by_id[cid]
                assert child.unit_index == parent_unit, (
                    "parent ข้ามขอบเขต unit!"
                )

    def test_empty_unit_list(self):
        children, parents = chunk_document([], "doc-empty")
        assert children == []
        assert parents == []

    def test_all_failed_units(self):
        units = [
            _make_unit("ข้อความ", index=i, status=UnitStatus.FAILED)
            for i in range(3)
        ]
        children, parents = chunk_document(units, "doc-all-fail")
        assert children == []
        assert parents == []


# ---------- format_citation_context ----------

class TestFormatCitationContext:
    def test_sentence_ids_sequential(self):
        text = "ประโยคแรก\nประโยคที่สองที่ยาวพอสมควร\nประโยคที่สาม"
        unit = _make_unit(text)
        result = chunk_unit(unit, "doc-1")
        context = format_citation_context(result.children, start_index=1)
        ids = [c["sentence_id"] for c in context]
        expected = [f"S{i+1}" for i in range(len(context))]
        assert ids == expected

    def test_custom_start_index(self):
        text = "ประโยคทดสอบที่มีความยาวพอสมควร"
        unit = _make_unit(text)
        result = chunk_unit(unit, "doc-1")
        context = format_citation_context(result.children, start_index=5)
        assert context[0]["sentence_id"] == "S5"

    def test_metadata_fields_present(self):
        text = "ประโยคที่มีเนื้อหา\nประโยคที่สองมีเนื้อหา"
        unit = _make_unit(text, index=2)
        result = chunk_unit(unit, "doc-abc")
        context = format_citation_context(result.children)
        for item in context:
            assert "sentence_id" in item
            assert "text" in item
            assert "metadata" in item
            meta = item["metadata"]
            assert "doc_id" in meta
            assert "char_start" in meta
            assert "char_end" in meta
            assert "position" in meta
            assert "label" in meta

    def test_text_content_not_empty(self):
        text = "บริษัทมีรายได้ 1,022,143 ล้านบาท\nและกำไรสุทธิ 28,206 ล้านบาท"
        unit = _make_unit(text)
        result = chunk_unit(unit, "doc-1")
        context = format_citation_context(result.children)
        for item in context:
            assert item["text"].strip() != ""


# ---------- get_parent_for_child ----------

class TestGetParentForChild:
    def test_finds_correct_parent(self):
        text = "\n".join([f"ประโยคที่ {i+1} ที่มีความยาวพอสมควร" for i in range(5)])
        unit = _make_unit(text)
        result = chunk_unit(unit, "doc-1")

        for child in result.children:
            parent = get_parent_for_child(child, result.parents)
            assert parent is not None
            assert parent.chunk_id == child.parent_id

    def test_returns_none_for_unknown_child(self):
        fake_child = ChildChunk(
            chunk_id=str(uuid.uuid4()),
            parent_id="nonexistent-id",
            document_id="doc",
            unit_index=0,
            sentence_index=0,
            content="test",
            char_start=0,
            char_end=4,
            position={"page": 1},
            label="หน้า 1",
        )
        result = get_parent_for_child(fake_child, [])
        assert result is None


# ---------- build_system_prompt ----------

class TestBuildSystemPrompt:
    def test_prompt_contains_key_instructions(self):
        prompt = build_system_prompt()
        assert "[S1]" in prompt
        assert "อ้างอิง" in prompt
        assert "ห้ามใช้ความรู้ภายนอก" in prompt

    def test_prompt_is_non_empty_string(self):
        prompt = build_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50


# ---------- Dataset Smoke Test ----------

class TestDatasetSmokeTest:
    """ทดสอบกับ expected_citation จาก rag_evaluation_1000.jsonl"""

    @pytest.fixture(scope="class")
    def eval_samples(self):
        if not EVAL_PATH.exists():
            pytest.skip(f"ไม่พบ eval dataset ที่ {EVAL_PATH}")
        samples = []
        with EVAL_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def test_citation_texts_are_chunkable(self, eval_samples):
        """expected_citation จาก dataset ต้องตัดเป็น chunk ได้ (ไม่ crash)"""
        errors = []
        for sample in eval_samples[:50]:  # ทดสอบ 50 ตัวอย่างแรก
            citation_text = sample.get("expected_citation", "")
            if not citation_text or not citation_text.strip():
                continue
            try:
                unit = _make_unit(citation_text, index=0)
                result = chunk_unit(unit, f"doc-{sample['question_id']}")
                # ถ้า text ไม่ว่าง ต้องได้ children
                if citation_text.strip():
                    assert len(result.children) >= 1 or result.skipped_reason is not None
            except Exception as e:
                errors.append(f"question_id={sample['question_id']}: {e}")

        assert not errors, "พบข้อผิดพลาด:\n" + "\n".join(errors)

    def test_citation_text_appears_in_child_content(self, eval_samples):
        """เนื้อหาสำคัญ (expected_answer) ต้องอยู่ใน child ที่ตัดออกมา"""
        missing = []
        for sample in eval_samples[:30]:
            citation_text = sample.get("expected_citation", "")
            expected_answer = sample.get("expected_answer", "")
            if not citation_text.strip() or not expected_answer.strip():
                continue

            unit = _make_unit(citation_text, index=0)
            result = chunk_unit(unit, f"doc-{sample['question_id']}")

            if not result.children:
                continue

            all_content = " ".join(c.content for c in result.children)
            # expected_answer ควรปรากฏใน content (หลัง normalize)
            norm_answer = normalize(expected_answer)
            norm_content = normalize(all_content)
            # ตรวจแบบ substring อย่างหยาบๆ (ตัวเลข/คำสำคัญ)
            # ใช้คำแรกของ expected_answer เป็น probe
            first_token = norm_answer.split()[0] if norm_answer.split() else norm_answer[:5]
            if first_token and first_token not in norm_content:
                missing.append(
                    f"Q{sample['question_id']}: '{first_token}' ไม่อยู่ใน chunks "
                    f"(citation='{citation_text[:50]}...')"
                )

        # ยอมรับ miss ได้ไม่เกิน 20% (normalize อาจเปลี่ยนรูปบ้าง)
        if missing:
            miss_rate = len(missing) / 30
            assert miss_rate <= 0.20, (
                f"miss rate {miss_rate:.1%} > 20%:\n" + "\n".join(missing[:5])
            )

    def test_offsets_consistent_with_citation_text(self, eval_samples):
        """char_start / char_end ต้องอยู่ในขอบเขต citation_text"""
        for sample in eval_samples[:20]:
            citation_text = sample.get("expected_citation", "")
            if not citation_text.strip():
                continue

            unit = _make_unit(citation_text, index=0)
            result = chunk_unit(unit, f"doc-{sample['question_id']}")

            for child in result.children:
                assert 0 <= child.char_start <= len(citation_text), (
                    f"Q{sample['question_id']}: char_start={child.char_start} "
                    f"นอกขอบ len={len(citation_text)}"
                )
                assert 0 <= child.char_end <= len(citation_text), (
                    f"Q{sample['question_id']}: char_end={child.char_end} "
                    f"นอกขอบ len={len(citation_text)}"
                )

    def test_parent_context_wider_than_child(self, eval_samples):
        """parent ที่มี >= 2 children ต้องมี content ยาวกว่า child เดี่ยว"""
        for sample in eval_samples[:20]:
            citation_text = sample.get("expected_citation", "")
            if not citation_text.strip():
                continue

            unit = _make_unit(citation_text, index=0)
            result = chunk_unit(unit, f"doc-{sample['question_id']}")

            multi_child_parents = [p for p in result.parents if p.sentence_count >= 2]
            for parent in multi_child_parents:
                # content ของ parent ควรยาวกว่า child ที่สั้นที่สุดใน parent
                child_map = {c.chunk_id: c for c in result.children}
                shortest_child = min(
                    (child_map[cid] for cid in parent.child_ids if cid in child_map),
                    key=lambda c: len(c.content),
                    default=None,
                )
                if shortest_child:
                    assert len(parent.content) >= len(shortest_child.content)
