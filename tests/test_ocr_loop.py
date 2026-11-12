"""ตัวตรวจลูปท้ายผล OCR (app/extraction/ocr_loop.py) — ข้อความทดสอบเขียนเองทั้งหมด"""

from app.extraction import ocr_loop
from app.extraction.ocr_loop import LOOP_MIN_CHARS, LOOP_MIN_REPEATS, find_tail_loop, trim_tail_loop

SENTENCE = "ครอบคลุมทั้งการบริหารความเสี่ยงของกลุ่มธุรกิจ "  # ท่อนที่โมเดลพิมพ์วน (จบด้วยช่องว่าง)
PREFIX = "บรรทัดแรกของหน้า\nย่อหน้าที่สอง\n"  # จบด้วย \n ซึ่งไม่ใช่ตัวท้ายของท่อนซ้ำ -> ช่วงซ้ำเริ่มตรงท่อนแรกพอดี


def test_clean_text_is_untouched():
    text = PREFIX + "ข้อความปกติที่จบประโยคแล้ว <page_number>12</page_number>"
    r = trim_tail_loop(text)
    assert r.text == text and r.removed_chars == 0 and r.periods == []


def test_sentence_loop_cut_mid_copy_keeps_prefix_and_one_copy():
    text = PREFIX + SENTENCE * 30 + SENTENCE[:17]  # ถูกตัดที่เพดานกลางท่อน
    r = trim_tail_loop(text)
    assert r.text == PREFIX + SENTENCE
    assert r.removed_chars == len(text) - len(PREFIX + SENTENCE)
    assert r.periods == [len(SENTENCE)]


def test_empty_cell_loop():
    head = "<table><tr><th>ชื่อ</th></tr>\n"
    text = head + "<td></td>" * 200 + "<td"
    r = trim_tail_loop(text)
    assert r.text == head + "<td></td>"


def test_row_loop_beats_cell_loop_inside_it():
    """แถวที่วนมี <td></td> ซ้ำข้างใน: ต้องตัดที่ระดับแถว (ช่วงซ้ำยาวสุด) ไม่ใช่เหลือแถวที่ถูกตัดช่อง"""
    head = "<table>\n"
    row = "<tr><td>บจก. ทดสอบ</td>" + "<td></td>" * 12 + "</tr>"
    text = head + row * 40 + row[:50]
    r = trim_tail_loop(text)
    assert r.text == head + row
    assert r.periods == [len(row)]


def test_repeats_inside_text_that_ends_normally_are_kept():
    """ช่องตาราง "-" ซ้ำกลางหน้าเป็นเนื้อหาจริง; ผลที่จบด้วย </table> ไม่มีคาบที่ท้าย"""
    text = "<table>" + "<tr><td>-</td><td>-</td></tr>" * 50 + "</table>"
    assert trim_tail_loop(text).removed_chars == 0


def test_below_min_repeats_is_not_a_loop():
    unit = "ก" * 120 + "\n"
    text = PREFIX + unit * (LOOP_MIN_REPEATS - 1)
    assert find_tail_loop(text) is None


def test_below_min_chars_is_not_a_loop():
    text = PREFIX + "-" * (LOOP_MIN_CHARS - 1) + "-"  # ซ้ำ LOOP_MIN_CHARS ตัว: ตัดได้ LOOP_MIN_CHARS - 1 < เกณฑ์
    assert find_tail_loop(text) is None
    assert find_tail_loop(PREFIX + "-" * (LOOP_MIN_CHARS + 1)) is not None


def test_trailing_whitespace_does_not_hide_loop():
    text = PREFIX + SENTENCE * 20 + "\n  "
    assert trim_tail_loop(text).text == PREFIX + SENTENCE


def test_cut_lands_on_grapheme_boundary():
    """จุดตัดต้องไม่แยกสระ/วรรณยุกต์ออกจากพยัญชนะ"""
    assert ocr_loop._grapheme_floor("กินข้าว", 1) == 0  # "กิ" เป็น grapheme เดียว
    assert ocr_loop._grapheme_floor("กินข้าว", 2) == 2
    assert ocr_loop._grapheme_floor("abc", 3) == 3


def test_does_not_invent_text():
    text = PREFIX + SENTENCE * 25
    r = trim_tail_loop(text)
    assert text.startswith(r.text)  # ตัดท้ายอย่างเดียว ไม่เติม ไม่แก้
