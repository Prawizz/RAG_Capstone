"""แปลงผลลัพธ์ markdown ของโมเดล OCR เป็นโครงสร้าง block (ตัวแปลงของ OCR — ไม่ใช่ normalizer)

โมเดล Typhoon OCR คืน markdown ที่มี HTML ปนอยู่ ตัวแปลงนี้แยกให้เป็นของที่ระบบใช้ต่อได้ ก่อนส่งแต่ละส่วนเข้า normalizer:
- ``<page_number>…</page_number>`` ตัดออกจากข้อความ; ถ้าเป็นตัวเลขเก็บเลขไว้ใน ``ParsedOcr.page_numbers``
- ``<table>`` HTML -> ``RawTable`` (rows / header_rows / merged_cells); ``<thead>`` หรือแถวแรกที่เป็น ``<th>`` ล้วน = header
  ตารางที่มี rowspan/colspan: ตำแหน่งในตารางถูกต้อง (เซลล์ที่คลุมอยู่เป็นช่องว่าง "" และบันทึก ``MergedCell`` ที่เซลล์ต้นทาง)
  เหมือน extractor ของ DOCX
- ``<figure>`` = คำบรรยายภาพที่โมเดลแต่งเอง ไม่ใช่ข้อความในเอกสาร -> block ชนิด image (ผู้เรียกต้องไม่รวมใน Unit.text)
- ``# หัวข้อ`` -> heading (level = จำนวน #), รายการ (``*`` ``-`` ``1.``) ติดกัน -> list หนึ่ง block, นอกนั้น -> paragraph
- HTML entity (``&amp;`` ฯลฯ) ถูกถอดก่อนส่งเข้า normalizer; ``<br>`` เป็นขึ้นบรรทัด
ข้อความที่คืนยังไม่ผ่าน normalizer
"""

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Optional

from app.models import BlockType, MergedCell

_SPECIAL = re.compile(r"(<table\b.*?(?:</table>|\Z)|<figure\b.*?(?:</figure>|\Z))", re.DOTALL | re.IGNORECASE)
_PAGE_NUMBER = re.compile(r"<page_number>(.*?)</page_number>", re.DOTALL | re.IGNORECASE)
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_LIST_ITEM = re.compile(r"^(?:[*\-+•]|\d+[.)])\s+\S")
_DIGITS = re.compile(r"\d+")


@dataclass
class RawTable:
    rows: list[list[str]]
    header_rows: int = 0
    merged_cells: list[MergedCell] = field(default_factory=list)


@dataclass
class RawBlock:
    type: BlockType
    text: str = ""
    level: Optional[int] = None
    table: Optional[RawTable] = None


@dataclass
class ParsedOcr:
    blocks: list[RawBlock] = field(default_factory=list)
    page_numbers: list[int] = field(default_factory=list)
    figure_count: int = 0


# ---------- ตาราง HTML ----------

class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int, bool]]] = []  # (ข้อความ, rowspan, colspan, เป็น th)
        self.in_head: list[bool] = []
        self._cell: Optional[list] = None
        self._head = False
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self._depth += 1
        elif self._depth > 1:
            if tag == "br" and self._cell is not None:
                self._cell[0].append(" ")
        elif tag == "thead":
            self._head = True
        elif tag == "tbody":
            self._head = False
        elif tag == "tr":
            self.rows.append([])
            self.in_head.append(self._head)
        elif tag in ("td", "th") and self.rows:
            self._cell = [[], _span(a.get("rowspan")), _span(a.get("colspan")), tag == "th"]
        elif tag == "br" and self._cell is not None:
            self._cell[0].append(" ")

    def handle_endtag(self, tag):
        if tag == "table":
            self._depth = max(self._depth - 1, 0)
        elif self._depth > 1:
            return
        elif tag == "thead":
            self._head = False
        elif tag in ("td", "th"):
            self._flush_cell()
        elif tag == "tr":
            self._flush_cell()

    def handle_data(self, data):
        if self._cell is not None:
            self._cell[0].append(data)

    def _flush_cell(self):
        if self._cell is not None and self.rows:
            parts, rs, cs, th = self._cell
            self.rows[-1].append((" ".join("".join(parts).split()), rs, cs, th))
        self._cell = None

    def close(self):
        super().close()
        self._flush_cell()


def _span(v) -> int:
    try:
        return max(int(v), 1)
    except (TypeError, ValueError):
        return 1


def parse_table(source: str) -> RawTable:
    p = _TableParser()
    p.feed(source)
    p.close()
    n_rows = len(p.rows)
    occupied: set[tuple[int, int]] = set()
    cells: dict[tuple[int, int], str] = {}
    merged: list[MergedCell] = []
    width = 0
    for r, row in enumerate(p.rows):
        c = 0
        for text, rs, cs, _ in row:
            while (r, c) in occupied:
                c += 1
            rs = min(rs, n_rows - r)
            cells[(r, c)] = text
            for dr in range(rs):
                for dc in range(cs):
                    occupied.add((r + dr, c + dc))
            if rs > 1 or cs > 1:
                merged.append(MergedCell(row=r, col=c, rowspan=rs, colspan=cs))
            c += cs
            width = max(width, c)
    rows = [[cells.get((r, c), "") for c in range(width)] for r in range(n_rows)]
    header_rows = sum(1 for h in p.in_head if h)
    if not header_rows:
        for row in p.rows:
            if row and all(th for *_, th in row):
                header_rows += 1
            else:
                break
    return RawTable(rows=rows, header_rows=min(header_rows, n_rows), merged_cells=merged)


# ---------- markdown ----------

def _clean(text: str) -> str:
    return html.unescape(_BR.sub("\n", text))


def _parse_plain(text: str, out: list[RawBlock]) -> None:
    para: list[str] = []
    items: list[str] = []

    def flush():
        if para:
            out.append(RawBlock(BlockType.PARAGRAPH, "\n".join(para)))
            para.clear()
        if items:
            out.append(RawBlock(BlockType.LIST, "\n".join(items)))
            items.clear()

    for raw in _clean(text).splitlines():
        line = raw.strip()
        if not line:
            flush()
        elif m := _HEADING.match(line):
            flush()
            out.append(RawBlock(BlockType.HEADING, m.group(2), level=len(m.group(1))))
        elif _LIST_ITEM.match(line):
            if para:
                flush()
            items.append(line)
        else:
            if items:
                flush()
            para.append(line)
    flush()


def parse_ocr_markdown(markdown: str) -> ParsedOcr:
    parsed = ParsedOcr()

    def take_page_number(m: re.Match) -> str:
        found = _DIGITS.search(html.unescape(m.group(1)))
        if found:
            parsed.page_numbers.append(int(found.group()))
        return ""

    text = _PAGE_NUMBER.sub(take_page_number, markdown)
    for seg in _SPECIAL.split(text):
        if not seg.strip():
            continue
        low = seg.lstrip().lower()
        if low.startswith("<table"):
            parsed.blocks.append(RawBlock(BlockType.TABLE, table=parse_table(seg)))
        elif low.startswith("<figure"):
            body = re.sub(r"^\s*<figure[^>]*>|</figure>\s*$", "", seg.strip(), flags=re.IGNORECASE)
            parsed.figure_count += 1
            parsed.blocks.append(RawBlock(BlockType.IMAGE, " ".join(_clean(body).split())))
        else:
            _parse_plain(seg, parsed.blocks)
    return parsed
