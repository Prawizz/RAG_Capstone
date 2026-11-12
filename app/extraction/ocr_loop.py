"""ตัวตรวจ "โมเดลพิมพ์วน" ท้ายผล OCR (ใช้กับทุก backend)

โมเดลแบบ generate (เช่น Typhoon greedy) ติดลูปแล้วพิมพ์ท่อนเดิมซ้ำจนชนเพดานโทเคน — ท้ายผลจึงเป็นท่อนที่ซ้ำเป็นคาบ
(ประโยคเดิม, ``<td></td>`` เดิม, แถวตารางเดิม) และจบกลางท่อนเพราะถูกตัดที่เพดาน

วิธีตรวจ: หาคาบ p ที่ "ส่วนท้ายสุด" ของข้อความมีคาบ p ยาวที่สุด (s[i] == s[i+p] ทุกตัวในช่วงนั้น; คำนวณด้วย Z-function
ของข้อความกลับด้าน -> ช่วงท้ายที่มีคาบ p ยาว p + Z[p]) นับเป็นลูปเมื่อ
  - ซ้ำอย่างน้อย ``LOOP_MIN_REPEATS`` รอบ และ
  - ส่วนที่จะตัดทิ้ง (ทุกรอบยกเว้นรอบแรก) ยาวอย่างน้อย ``LOOP_MIN_CHARS`` อักขระ
ถ้าหลายคาบผ่านเกณฑ์ เลือกคาบที่ช่วงซ้ำเริ่มเร็วที่สุด (ยาวสุด) เสมอกันเลือกคาบสั้นสุด — ทำให้ลูประดับแถวตารางชนะ
ลูประดับ ``<td></td>`` ข้างใน

การตัด: เก็บข้อความก่อนเกิดลูป + ท่อนที่ซ้ำ **หนึ่งรอบ** (รอบแรกคือที่โมเดลอ่านจริง) ตัดรอบที่เหลือทิ้ง
แล้วถอยจุดตัดไปที่ขอบ grapheme (ไม่ให้สระ/วรรณยุกต์ลอย) — ไม่เติม ไม่เดา ไม่แก้ข้อความที่เหลือ
ทำซ้ำได้สูงสุด ``LOOP_MAX_PASSES`` รอบ (ลูปซ้อน)

ทำไมต้องยึด "ท้ายสุด": ผลปกติจบด้วยของที่ไม่ซ้ำ (``</table>``, ``<page_number>``, ประโยคสุดท้าย) ช่วงท้ายจึงไม่มีคาบ
ส่วนผลที่วนจนชนเพดานจบกลางลูปเสมอ — กันตัดบรรทัดซ้ำที่เป็นเนื้อหาจริงกลางหน้า (เช่นช่องตาราง "-" ซ้ำ)
ข้อจำกัด: แถวที่วนแต่ไม่เหมือนกันทุกตัวอักษร (เช่นตัวเลขเพิ่มทีละ 1) ไม่ถูกจับ;
เนื้อหาจริงที่ซ้ำเหมือนกันทุกตัวอักษร >= 4 รอบแล้วถูกตัดที่เพดานพอดี จะถูกตัดเหลือรอบเดียว (หน้านั้นติด flag อยู่แล้ว)
ใช้กับผลดิบของโมเดล "ทีละชิ้นภาพ" (ก่อนต่อชิ้น) เพราะลูปอยู่ท้ายผลของชิ้นที่วน ไม่ใช่ท้ายหน้า
"""

from dataclasses import dataclass
from typing import Optional

import regex

LOOP_MIN_REPEATS = 4
LOOP_MIN_CHARS = 200
LOOP_MAX_PERIOD = 4000  # คาบยาวสุดที่ตรวจ (อักขระ) — แถวตารางยาวสุดที่เจอใน KBANK ~170
LOOP_MAX_PASSES = 3

_GRAPHEME = regex.compile(r"\X")


@dataclass
class LoopTrim:
    text: str  # ข้อความหลังตัด
    removed_chars: int  # จำนวนอักขระที่ตัดทิ้ง (0 = ไม่มีลูป)
    periods: list[int]  # คาบของลูปที่ตัดแต่ละรอบ


def _z_function(s: str) -> list[int]:
    n = len(s)
    z = [0] * n
    left = right = 0
    for i in range(1, n):
        if i < right:
            z[i] = min(right - i, z[i - left])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > right:
            left, right = i, i + z[i]
    return z


def find_tail_loop(text: str) -> Optional[tuple[int, int]]:
    """(คาบ, ตำแหน่งเริ่มของช่วงซ้ำ) ของลูปท้ายข้อความ (ไม่สนช่องว่างท้ายสุด); ไม่มีลูป -> None"""
    s = text.rstrip()
    n = len(s)
    if n < LOOP_MIN_REPEATS:
        return None
    z = _z_function(s[::-1])
    best: Optional[tuple[int, int]] = None  # (p, length)
    for p in range(1, min(LOOP_MAX_PERIOD, n // LOOP_MIN_REPEATS) + 1):
        length = p + z[p]
        if length >= LOOP_MIN_REPEATS * p and length - p >= LOOP_MIN_CHARS:
            if best is None or length > best[1]:
                best = (p, length)
    if best is None:
        return None
    return best[0], n - best[1]


def _grapheme_floor(s: str, pos: int) -> int:
    """ขอบ grapheme ที่ <= pos"""
    last = 0
    for m in _GRAPHEME.finditer(s):
        if m.end() > pos:
            break
        last = m.end()
    return last


def trim_tail_loop(text: str) -> LoopTrim:
    """ตัดลูปท้ายข้อความ (เก็บส่วนก่อนลูป + ท่อนซ้ำหนึ่งรอบ); ไม่มีลูป -> ข้อความเดิมทุกตัว"""
    cur = text
    periods: list[int] = []
    for _ in range(LOOP_MAX_PASSES):
        found = find_tail_loop(cur)
        if found is None:
            break
        p, start = found
        cut = _grapheme_floor(cur, start + p)
        if cut >= len(cur.rstrip()):
            break
        cur = cur[:cut]
        periods.append(p)
    return LoopTrim(text=cur, removed_chars=len(text) - len(cur), periods=periods)
