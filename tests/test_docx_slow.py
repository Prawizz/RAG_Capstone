"""Test ช้า: ไฟล์ DOCX จริง (eval/datasets/files/ ไม่อยู่ใน git — ไม่มีไฟล์ก็ข้าม) รันด้วย ``pytest -m slow``"""

import re
from pathlib import Path

import pytest
from docx import Document as WordDocument
from docx.table import Table as WordTable

from app.extraction.docx import MAX_UNIT_CHARS
from app.extraction.pipeline import extract_file
from app.models import BlockType, DocumentStatus
from app.thai.normalizer import normalize

THESIS = Path(__file__).parent.parent / "eval" / "datasets" / "files" / "thesis_manual_th.docx"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not THESIS.exists(), reason="ไม่มี eval/datasets/files/thesis_manual_th.docx"),
]

LOOKAHEAD = 8
_CONTINUATION = re.compile(r"\(ส่วน (\d+)/\d+\)$")


@pytest.fixture(scope="module")
def doc():
    d = extract_file(THESIS)
    assert d.error is None
    return d


def all_text(p_element):
    """ข้อความทุก w:t ในย่อหน้า (รวมที่อยู่ใน smartTag/ins ซึ่ง python-docx ``Paragraph.text`` ไม่อ่าน)"""
    out = []
    for el in p_element.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "t":
            out.append(el.text or "")
        elif tag == "tab":
            out.append("\t")
        elif tag in ("br", "cr"):
            out.append("\n")
    return "".join(out)


def source_elements():
    """ย่อหน้า/ตารางระดับ body จาก python-docx (เป็นเครื่องมืออ่านคนละตัวกับ extractor ของเรา)"""
    word = WordDocument(str(THESIS))
    paragraphs, tables = [], []
    for child in word.element.body.iterchildren():
        if child.tag.endswith("}p"):
            text = normalize(all_text(child))
            if text:
                paragraphs.append(text)
        elif child.tag.endswith("}tbl"):
            tables.append(WordTable(child, word))
    return paragraphs, tables


def output_blocks(doc):
    """block ทั้งหมดตามลำดับเอกสาร; หัวข้อที่ไม่มีเนื้อหาของตัวเอง (ไม่มี unit) นับว่าปรากฏหนึ่งครั้งใน heading_path
    ของ unit ลูกตัวแรก จึงใส่เป็น block เสมือน (SimpleNamespace) ก่อน unit นั้น"""
    from types import SimpleNamespace

    blocks, emitted = [], set()
    for u in doc.units:
        path = u.position["heading_path"]
        m = _CONTINUATION.search(u.label)
        repeated_heading = bool(m and int(m.group(1)) > 1)
        for i in range(len(path) - 1):
            if tuple(path[: i + 1]) not in emitted:
                emitted.add(tuple(path[: i + 1]))
                blocks.append(SimpleNamespace(type=BlockType.HEADING, text=path[i]))
        if path:
            emitted.add(tuple(path))
        for b in u.blocks:
            if b.type == BlockType.HEADING and repeated_heading:
                continue  # หัวข้อที่ซ้ำขึ้นหัวส่วนที่ 2 เป็นต้นไป
            blocks.append(b)
    return blocks


def test_status_and_basic_shape(doc):
    assert doc.status == DocumentStatus.DONE and doc.has_outline
    assert all(u.type.value == "section" and u.method.value == "office_xml" for u in doc.units)


def test_every_nonempty_paragraph_appears_exactly_once_in_order(doc):
    """ลำดับเดียวกับเอกสาร ยกเว้นหัวข้อเปล่าที่ตกหล่น (ไม่มีเนื้อหา ไม่อยู่ใน heading_path ของ unit ใด) ซึ่ง extractor
    ยกไปไว้ต้นเนื้อหาของ unit ถัดไป จึงยอมให้เลื่อนตำแหน่งได้ไม่เกิน LOOKAHEAD block"""
    paragraphs, _ = source_elements()
    blocks = [b for b in output_blocks(doc) if b.type != BlockType.TABLE]
    used = [False] * len(blocks)
    ptr = 0
    merged_block = None  # block ที่เป็น "บทที่ N" + ชื่อบท รวมกัน
    for s in paragraphs:
        if merged_block is not None and blocks[merged_block].text.endswith(" " + s):
            merged_block = None
            continue
        merged_block = None
        for k in range(ptr, min(ptr + LOOKAHEAD, len(blocks))):
            b = blocks[k].text
            if not used[k] and (b == s or b.endswith(" " + s) or b.startswith(s + " ")):
                used[k] = True
                if b != s and b.startswith(s + " ") and not b.endswith(" " + s):
                    merged_block = k
                break
        else:
            pytest.fail(f"ย่อหน้าหายหรือลำดับเพี้ยน (block {ptr}): {s!r} vs {blocks[ptr].text if ptr < len(blocks) else None!r}")
        while ptr < len(blocks) and used[ptr]:
            ptr += 1
    assert not any(not u for u in used), [blocks[i].text for i, u in enumerate(used) if not u][:3]  # ไม่มี block เกินที่ไม่มีที่มา


def test_every_table_is_one_table_block_with_all_cell_text(doc):
    _, tables = source_elements()
    table_blocks = [b for b in output_blocks(doc) if b.type == BlockType.TABLE]
    assert len(table_blocks) == len(tables)
    for b, t in zip(table_blocks, tables):
        for row in t.rows:
            for cell in row.cells:
                text = normalize(cell.text)
                for piece in filter(None, (normalize(x) for x in cell.text.split("\n"))):
                    assert piece in b.text, piece
                assert text is not None


def test_unit_size_limit_except_single_long_paragraph(doc):
    for u in doc.units:
        if len(u.text) > MAX_UNIT_CHARS:
            content = [b for b in u.blocks if b.type != BlockType.HEADING]
            assert len(content) == 1, (u.label, len(u.text))


def test_heading_path_of_nested_heading(doc):
    paths = [u.position["heading_path"] for u in doc.units]
    assert [
        "บทที่ 1 บทนำ",
        "1.2 ความสำคัญของขั้นตอนการนำเสนอวิทยานิพนธ์",
        "1.2.1 ส่วนบทนำ",
    ] in paths


def test_appendix_chapter_is_under_appendix(doc):
    appendix_chapters = [
        u for u in doc.units
        if u.position["heading_path"][:2] == ["ภาคผนวก ก.", "บทที่ 1 บทนำ"]
    ]
    assert appendix_chapters
    assert not any(
        u.position["heading_path"][0] == "บทที่ 1 บทนำ" and "ภาคผนวก" in " ".join(u.position["heading_path"][1:])
        for u in doc.units
    )


def test_section_5_3_is_split(doc):
    parts = [u for u in doc.units if u.position["heading_path"][-1] == "5.3 วิธีลงรายการของบรรณานุกรม"]
    assert len(parts) >= 2
    assert [p.label.rsplit(" (ส่วน", 1)[-1] for p in parts] == [f" {k}/{len(parts)})" for k in range(1, len(parts) + 1)]
    assert all(p.text.startswith("5.3 วิธีลงรายการของบรรณานุกรม\n") for p in parts)


# เลขอัตโนมัติ 6 จุดนี้ คนเปิด Word ยืนยันแล้วว่าตรงกับที่ Word แสดง (ไม่ได้มาจากโค้ดของเรา)
WORD_CONFIRMED_NUMBERING = [
    "2.2 ส่วนที่ 2 ส่วนกลาง",
    "3.1 กระดาษที่ใช้พิมพ์",
    "3.2 ตัวพิมพ์",
    "3.3 การทำสำเนา",
    "3.4 การเว้นระยะการพิมพ์",
    "3.5 การเว้นระยะห่างจากริมกระดาษ",
]


@pytest.mark.parametrize("heading", WORD_CONFIRMED_NUMBERING)
def test_auto_numbering_confirmed_in_word(doc, heading):
    headings = {b.text for u in doc.units for b in u.blocks if b.type == BlockType.HEADING}
    assert heading in headings


def test_auto_numbering_unconfirmed_but_computed_from_numbering_xml(doc):
    # ค่านี้ผมคำนวณเองจาก numbering.xml (คนยังไม่ได้เปิด Word ยืนยัน)
    headings = {b.text for u in doc.units for b in u.blocks if b.type == BlockType.HEADING}
    assert "1.3 ภาษาที่ใช้ในการเขียนวิทยานิพนธ์" in headings
