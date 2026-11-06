"""chunk ตามขนาดเป้าหมายสำหรับ index ลง Qdrant (``make_chunks``): ขอบประโยค · ไม่ข้าม unit · ซ้อน 1 ประโยค · offset ตรง · context line"""

import hashlib

import pytest

from app.chunking import Chunk, make_chunks, make_document_chunks, split_sentence_spans, unit_context_lines
from app.models import ExtractionMethod, QualityFlag, QualityReport, Unit, UnitStatus, UnitType


def unit(text: str, index: int = 0, status=UnitStatus.OK, reason=None, flags=()) -> Unit:
    return Unit(
        index=index, type=UnitType.PAGE, position={"page": index + 1}, label=f"หน้า {index + 1}", status=status,
        reason=reason, method=ExtractionMethod.TEXT_LAYER, text=text, text_hash=hashlib.sha256(text.encode()).hexdigest(),
        quality=QualityReport(flags=list(flags)),
    )


def lines(n: int, width: int = 40) -> str:
    return "\n".join(f"บรรทัด{i:03d} " + "ก" * width for i in range(n))


def test_unit_shorter_than_target_is_one_chunk():
    u = unit("ประโยคแรกของหน้านี้\nประโยคที่สองของหน้านี้")
    chunks = make_chunks(u, 1000)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.text == u.text and (c.char_start, c.char_end) == (0, len(u.text)) and len(c.sentences) == 2


def test_chunks_respect_the_target_size_and_cover_the_whole_text():
    u = unit(lines(40))
    chunks = make_chunks(u, 300)
    assert len(chunks) > 3 and all(len(c.text) <= 300 for c in chunks)
    assert chunks[0].char_start == 0 and chunks[-1].char_end == len(u.text.rstrip())
    covered = set()
    for c in chunks:
        covered.update(range(c.char_start, c.char_end))
    assert covered >= {i for i, ch in enumerate(u.text) if not ch.isspace()}  # ไม่มีอักขระตกหล่น


def test_offsets_point_into_the_original_unit_text():
    u = unit(lines(25))
    for c in make_chunks(u, 250):
        assert u.text[c.char_start:c.char_end] == c.text
        for s in c.sentences:
            assert c.char_start <= s.char_start < s.char_end <= c.char_end
            assert u.text[s.char_start:s.char_end].strip() == u.text[s.char_start:s.char_end]


def test_consecutive_chunks_overlap_by_exactly_one_sentence():
    chunks = make_chunks(unit(lines(30)), 250)
    assert len(chunks) > 2
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.sentences[0] == prev.sentences[-1]
        assert nxt.sentences[1:] and not set(nxt.sentences[1:]) & set(prev.sentences)


def test_chunk_boundaries_are_sentence_boundaries():
    u = unit(lines(30))
    sentence_starts = {s for s, _ in split_sentence_spans(u.text)}
    sentence_ends = {e for _, e in split_sentence_spans(u.text)}
    for c in make_chunks(u, 250):
        assert c.char_start in sentence_starts and c.char_end in sentence_ends


def test_always_advances_even_when_one_sentence_nearly_fills_a_chunk():
    u = unit("\n".join("ก" * 90 + f" {i}" for i in range(6)))
    chunks = make_chunks(u, 100)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len(chunks) == 6 and all(len(c.sentences) == 1 for c in chunks)  # ซ้อนไม่ได้เพราะสองประโยคเกินเป้า


def test_sentence_longer_than_target_is_cut_at_grapheme_boundaries():
    u = unit("ก็" * 120)  # ก + ็ เป็น grapheme เดียว: ห้ามตัดคั่นกลาง
    chunks = make_chunks(u, 50)
    assert len(chunks) >= 5 and all(len(c.text) <= 50 for c in chunks)
    assert all(not c.text.startswith("็") for c in chunks)
    assert "".join(c.text for c in chunks if True)  # ไม่ว่าง


def test_chunks_never_cross_units():
    a, b = unit(lines(8), 0), unit(lines(8), 1)
    chunks = make_document_chunks([a, b], 300)
    assert {c.unit_index for c in chunks} == {0, 1}
    for c in chunks:
        src = a if c.unit_index == 0 else b
        assert src.text[c.char_start:c.char_end] == c.text
        assert c.position == {"page": c.unit_index + 1} and c.label == f"หน้า {c.unit_index + 1}"
    assert [c.chunk_index for c in chunks if c.unit_index == 1][0] == 0  # chunk_index นับใหม่ต่อ unit


def test_non_ok_and_empty_units_make_no_chunks():
    assert make_chunks(unit("", status=UnitStatus.NEEDS_OCR, reason="หน้าเป็นภาพ"), 1000) == []
    assert make_chunks(unit("พัง", status=UnitStatus.FAILED, reason="อ่านไม่ได้"), 1000) == []
    assert make_chunks(unit("   \n  ", flags=[QualityFlag.EMPTY]), 1000) == []


def test_chunking_is_deterministic():
    u = unit(lines(20))
    assert make_chunks(u, 300) == make_chunks(u, 300)


def test_target_must_be_positive():
    with pytest.raises(ValueError):
        make_chunks(unit("ข้อความ"), 0)


# ---------- context line ----------


TABLE = "[แถวแรกของตาราง: ชื่อ | ยอด]\n" + "\n".join(f"รายการที่ {i} | {i * 100}" for i in range(1, 12))


def test_context_lines_are_found_with_offsets():
    u = unit("หัวข้อ\n" + TABLE + "\n[ป้ายก่อนหน้า: ก → ข]\nแถว")
    found = unit_context_lines(u.text)
    assert [line for _, _, line in found] == ["[แถวแรกของตาราง: ชื่อ | ยอด]", "[ป้ายก่อนหน้า: ก → ข]"]
    assert all(u.text[s:e] == line for s, e, line in found)


def test_context_line_is_prepended_for_embedding_but_not_part_of_text_or_offsets():
    u = unit(TABLE)
    chunks = make_chunks(u, 120)
    assert len(chunks) > 2
    first, later = chunks[0], chunks[1:]
    assert first.context == "" and first.text.startswith("[แถวแรกของตาราง:")  # chunk ที่มีบรรทัดนั้นอยู่แล้วไม่แปะซ้ำ
    for c in later:
        assert c.context == "[แถวแรกของตาราง: ชื่อ | ยอด]"
        assert c.embed_text == f"{c.context}\n{c.text}"
        assert "[แถวแรกของตาราง" not in c.text
        assert u.text[c.char_start:c.char_end] == c.text  # offset ไม่รวม context


def test_context_follows_the_latest_context_line_before_the_chunk():
    u = unit("[ป้ายก่อนหน้า: รายได้]\n" + lines(5, 30) + "\n[ป้ายก่อนหน้า: ค่าใช้จ่าย]\n" + lines(5, 30))
    chunks = make_chunks(u, 150)
    by_start = [(c.char_start, c.context) for c in chunks]
    second_label = u.text.index("[ป้ายก่อนหน้า: ค่าใช้จ่าย]")
    assert all(ctx == "[ป้ายก่อนหน้า: รายได้]" for start, ctx in by_start if 0 < start < second_label)
    assert all(ctx == "[ป้ายก่อนหน้า: ค่าใช้จ่าย]" for start, ctx in by_start if start > second_label)


def test_unit_without_context_line_has_empty_context():
    assert all(c.context == "" and c.embed_text == c.text for c in make_chunks(unit(lines(10)), 200))


def test_chunk_is_immutable():
    c = make_chunks(unit("ข้อความหนึ่ง"), 100)[0]
    assert isinstance(c, Chunk)
    with pytest.raises(Exception):
        c.text = "แก้"  # type: ignore[misc]
