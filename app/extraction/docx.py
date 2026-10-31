"""DOCX extractor: หนึ่งหัวข้อ = หนึ่ง Unit (type=section, method=office_xml)

อ่าน XML ของ DOCX ตรงๆ (word/document.xml, styles.xml, numbering.xml, footnotes.xml) ด้วย lxml
เพราะต้องจำลองเลขอัตโนมัติของ Word เอง (docx2python ให้เลขผิด)

ลำดับเนื้อหา
- วนทุก element ใน body ตามลำดับ: ย่อหน้า (w:p) และตาราง (w:tbl); w:sdt ที่ครอบอยู่ถูกคลี่ออก
- กล่องข้อความ (w:txbxContent) ต่อเป็นย่อหน้าหลังย่อหน้าที่มันฝังอยู่ (ใน mc:AlternateContent เอาเฉพาะ mc:Choice
  ไม่เอา mc:Fallback ที่ซ้ำกัน) ไม่เอาหัว/ท้ายกระดาษ
- เชิงอรรถต่อท้าย unit ที่อ้างถึง เป็น block ย่อหน้าขึ้นต้นด้วย "เชิงอรรถ: "
- **index ของ element** = ลำดับของย่อหน้า/ตารางระดับ body (นับย่อหน้าว่างด้วย) เริ่มที่ 0
  ย่อหน้าในเซลล์ตาราง กล่องข้อความ และเชิงอรรถไม่มี index ของตัวเอง ``position["paragraphs"]`` = [index แรก, index สุดท้าย]
  ของ element ที่อยู่ใน unit นั้น (unit ส่วนที่ 2 เป็นต้นของหัวข้อที่ถูกแบ่ง เริ่มที่ element เนื้อหาตัวแรกของส่วนนั้น)

เลขอัตโนมัติ (``_Numbering``): คำนวณจาก numPr (ในย่อหน้าหรือใน style) -> num -> abstractNum รองรับ lvlOverride/
startOverride, start, lvlText (%1..%9), numFmt decimal/decimalZero/thaiNumbers/thaiCounting/thaiLetters/
lowerLetter/upperLetter/lowerRoman/upperRoman (bullet = ไม่ใส่เลข, ชนิดอื่นใช้ decimal) ตัวนับระดับล่างรีเซ็ตเมื่อระดับบน
เพิ่ม ระดับบนที่ยังไม่เคยใช้แสดงค่า start ของมัน ตัวนับแยกต่อ abstractNum (num หลายตัวที่ชี้ abstractNum เดียวกันนับต่อกัน
เหมือน Word ยกเว้นมี startOverride) thaiLetters ใช้พยัญชนะไทย 42 ตัวเรียงตามอักษร (ไม่รวม ฃ ฅ) เกินแล้วซ้ำตัวอักษร

หัวข้อ (``_classify_heading``) ไม่ใช้ style Heading ตัดสิน (เก็บเป็น signal ``docx:heading_style``):
ข้อความ (รวมเลขอัตโนมัติ) <= 80 ตัว, ทุก run ที่มีข้อความเป็นตัวหนา และ (ขนาด > body_size หรือขึ้นต้นด้วย
"บทที่ N" / "N.N.N " / "N.N ") ขนาดของย่อหน้า = ขนาดที่ครอบคลุมตัวอักษรมากที่สุด; body_size = ขนาดที่พบมากที่สุด
ในย่อหน้าระดับ body ที่มีข้อความ; ระดับ: บทที่ N = 1, N.N.N = 3, N.N = 2, ไม่มีเลข: ขนาด >= body + 4pt = 1 ไม่งั้น 2
"บทที่ N" ตามด้วยหัวข้อไม่มีเลขย่อหน้าถัดไป (ข้ามย่อหน้าว่าง) = รวมเป็นหัวข้อเดียว; หัวข้อ "ภาคผนวก..." = ระดับ 1 และ
หัวข้อหลังจากนั้นเลื่อนลงอีก 1 ระดับ; ``docx:heading_by_numbering`` = ตัดสินจากเลข/บทที่, ``docx:heading_by_format`` = จากขนาดล้วน

การแบ่ง unit: หัวข้อหนึ่ง + เนื้อหาจนถึงหัวข้อถัดไป (ระดับใดก็ได้); หัวข้อที่ไม่มีเนื้อหาไม่สร้าง unit แต่อยู่ใน heading_path
ของลูก; เนื้อหาก่อนหัวข้อแรก = "ส่วนต้นเอกสาร"; เกิน ``MAX_UNIT_CHARS`` แบ่งที่ขอบย่อหน้า/ตาราง (ย่อหน้าเดียวยาวเกินไม่ตัดกลาง)
ทุกส่วนขึ้นต้นด้วยบรรทัดหัวข้อ; label = heading_path ต่อด้วย " › " + " (ส่วน k/n)" ถ้าแบ่ง, label ซ้ำต่อท้าย " (ช่วงที่ k)"

Block: heading(level) / paragraph / list (ย่อหน้ามี numPr ที่ไม่ใช่หัวข้อ) / table (``Block.table``; ข้อความ = แถวละบรรทัด
ช่องคั่นด้วย " | " เซลล์ที่ถูกผสานเข้าไปไม่ซ้ำค่า) ข้อความทุกส่วนผ่าน normalizer; ผลนับรวมราย unit ใน
``Unit.signals`` ตามที่ระบุใน app/models.py; ``docx:image_count`` = จำนวนรูปใน unit (ยังไม่ OCR)
"""

import hashlib
import re
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from lxml import etree

from app.extraction.errors import ExtractionError
from app.models import (
    Block,
    BlockType,
    ErrorCode,
    ExtractionMethod,
    MergedCell,
    QualityFlag,
    QualityReport,
    Table,
    Unit,
    UnitStatus,
    UnitType,
)
from app.extraction.tally import Tally as _Tally

MAX_UNIT_CHARS = 4000
MAX_HEADING_CHARS = 80
HEADING_BIG_EXTRA_PT = 4  # หัวข้อไม่มีเลขที่ใหญ่กว่า body อย่างน้อยเท่านี้ = ระดับ 1
PREFACE_LABEL = "ส่วนต้นเอกสาร"
FOOTNOTE_PREFIX = "เชิงอรรถ: "
LABEL_SEP = " › "
DEFAULT_SIZE_HALF_PT = 20  # ถ้าไม่มี docDefaults (ค่าเริ่มต้นของ Word = 10pt)

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
V = "urn:schemas-microsoft-com:vml"
NS = {"w": W}


def _w(tag: str) -> str:
    return f"{{{W}}}{tag}"


def _val(el: Optional[etree._Element]) -> Optional[str]:
    return None if el is None else el.get(_w("val"))


def _on(el: Optional[etree._Element]) -> Optional[bool]:
    """w:b / w:i ฯลฯ: ไม่มี element = None, ไม่มี val = เปิด, val = 0/false/off = ปิด"""
    if el is None:
        return None
    v = el.get(_w("val"))
    return not (v is not None and v.lower() in ("0", "false", "off", "none"))


# ---------- styles ----------

class _Styles:
    def __init__(self, root: Optional[etree._Element]):
        self.styles: dict[str, etree._Element] = {}
        self.default_para: Optional[str] = None
        self.default_size = DEFAULT_SIZE_HALF_PT
        self.names: dict[str, str] = {}
        if root is None:
            return
        for s in root.findall("w:style", NS):
            sid = s.get(_w("styleId"))
            if not sid:
                continue
            self.styles[sid] = s
            self.names[sid] = _val(s.find("w:name", NS)) or ""
            if s.get(_w("type")) == "paragraph" and s.get(_w("default")) in ("1", "true"):
                self.default_para = sid
        sz = root.find("w:docDefaults/w:rPrDefault/w:rPr/w:sz", NS)
        if sz is None:
            sz = root.find("w:docDefaults/w:rPrDefault/w:rPr/w:szCs", NS)
        if _val(sz):
            self.default_size = int(float(_val(sz)))

    def chain(self, style_id: Optional[str]) -> Iterator[etree._Element]:
        seen = set()
        sid = style_id
        while sid and sid in self.styles and sid not in seen:
            seen.add(sid)
            yield self.styles[sid]
            sid = _val(self.styles[sid].find("w:basedOn", NS))

    def _rpr_lookup(self, style_id: Optional[str], getter):
        for s in self.chain(style_id):
            v = getter(s.find("w:rPr", NS))
            if v is not None:
                return v
        return None

    def para_bold(self, style_id: Optional[str]) -> Optional[bool]:
        return self._rpr_lookup(style_id or self.default_para, _rpr_bold)

    def para_size(self, style_id: Optional[str]) -> int:
        v = self._rpr_lookup(style_id or self.default_para, _rpr_size)
        return v if v is not None else self.default_size

    def char_bold(self, style_id: Optional[str]) -> Optional[bool]:
        return self._rpr_lookup(style_id, _rpr_bold)

    def char_size(self, style_id: Optional[str]) -> Optional[int]:
        return self._rpr_lookup(style_id, _rpr_size)

    def num_pr(self, style_id: Optional[str]) -> Optional[tuple[Optional[int], Optional[int]]]:
        """(numId, ilvl) จาก style (ไล่ตาม basedOn); None = ไม่มี numPr"""
        for s in self.chain(style_id or self.default_para):
            np_ = s.find("w:pPr/w:numPr", NS)
            if np_ is not None:
                n, lv = _val(np_.find("w:numId", NS)), _val(np_.find("w:ilvl", NS))
                return (int(n) if n is not None else None, int(lv) if lv is not None else None)
        return None

    def is_heading_style(self, style_id: Optional[str]) -> bool:
        if not style_id:
            return False
        name = self.names.get(style_id, "").lower()
        return style_id.lower().startswith(("heading", "title")) or name.startswith(("heading", "title"))


def _rpr_bold(rpr: Optional[etree._Element]) -> Optional[bool]:
    if rpr is None:
        return None
    b, bcs = _on(rpr.find("w:b", NS)), _on(rpr.find("w:bCs", NS))
    if b is None and bcs is None:
        return None
    return bool(b) or bool(bcs)


def _rpr_size(rpr: Optional[etree._Element]) -> Optional[int]:
    if rpr is None:
        return None
    for tag in ("sz", "szCs"):
        v = _val(rpr.find(f"w:{tag}", NS))
        if v:
            try:
                return int(float(v))
            except ValueError:
                pass
    return None


# ---------- เลขอัตโนมัติ ----------

THAI_LETTERS = "กขคงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
_THAI_DIGITS = str.maketrans("0123456789", "๐๑๒๓๔๕๖๗๘๙")
_ROMAN = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
          (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]


def _roman(n: int) -> str:
    out = ""
    for v, s in _ROMAN:
        while n >= v:
            out += s
            n -= v
    return out


def _repeat_letter(alphabet: str, n: int) -> str:
    n = max(n, 1) - 1
    return alphabet[n % len(alphabet)] * (n // len(alphabet) + 1)


def format_number(n: int, fmt: str) -> str:
    if fmt == "decimalZero":
        return f"{n:02d}"
    if fmt in ("thaiNumbers", "thaiCounting"):
        return str(n).translate(_THAI_DIGITS)
    if fmt == "thaiLetters":
        return _repeat_letter(THAI_LETTERS, n)
    if fmt == "lowerLetter":
        return _repeat_letter("abcdefghijklmnopqrstuvwxyz", n)
    if fmt == "upperLetter":
        return _repeat_letter("ABCDEFGHIJKLMNOPQRSTUVWXYZ", n)
    if fmt == "lowerRoman":
        return _roman(n)
    if fmt == "upperRoman":
        return _roman(n).upper()
    return str(n)


@dataclass
class _Level:
    start: int = 1
    fmt: str = "decimal"
    text: str = "%1"
    restart: Optional[int] = None
    style: Optional[str] = None


def _parse_level(lvl: etree._Element) -> _Level:
    def g(tag):
        return _val(lvl.find(f"w:{tag}", NS))

    start = g("start")
    restart = g("lvlRestart")
    return _Level(
        start=int(start) if start is not None else 1,
        fmt=g("numFmt") or "decimal",
        text=g("lvlText") if g("lvlText") is not None else "",
        restart=int(restart) if restart is not None else None,
        style=g("pStyle"),
    )


class _Numbering:
    def __init__(self, root: Optional[etree._Element]):
        self.abstract: dict[str, dict[int, _Level]] = {}
        self.nums: dict[int, tuple[str, dict[int, _Level], dict[int, int]]] = {}
        self.counters: dict[str, list[Optional[int]]] = {}
        self._applied: set[tuple[int, int]] = set()
        if root is None:
            return
        for a in root.findall("w:abstractNum", NS):
            self.abstract[a.get(_w("abstractNumId"))] = {
                int(lv.get(_w("ilvl"))): _parse_level(lv) for lv in a.findall("w:lvl", NS)
            }
        for n in root.findall("w:num", NS):
            nid = n.get(_w("numId"))
            aid = _val(n.find("w:abstractNumId", NS))
            if nid is None or aid not in self.abstract:
                continue
            levels = dict(self.abstract[aid])
            starts: dict[int, int] = {}
            for o in n.findall("w:lvlOverride", NS):
                il = int(o.get(_w("ilvl")))
                if o.find("w:lvl", NS) is not None:
                    levels[il] = _parse_level(o.find("w:lvl", NS))
                so = _val(o.find("w:startOverride", NS))
                if so is not None:
                    starts[il] = int(so)
            self.nums[int(nid)] = (aid, levels, starts)

    def style_level(self, num_id: int, style_id: Optional[str]) -> Optional[int]:
        """ilvl ที่ผูกกับ style ผ่าน w:lvl/w:pStyle (ใช้เมื่อ numPr ใน style ไม่ระบุ ilvl)"""
        if style_id and num_id in self.nums:
            for il, lv in self.nums[num_id][1].items():
                if lv.style == style_id:
                    return il
        return None

    def label(self, num_id: Optional[int], ilvl: int) -> Optional[str]:
        if not num_id or num_id not in self.nums or not 0 <= ilvl <= 8:
            return None
        aid, levels, starts = self.nums[num_id]
        lvl = levels.get(ilvl)
        if lvl is None or lvl.fmt in ("bullet", "none"):
            return None
        ctr = self.counters.setdefault(aid, [None] * 9)
        if ilvl in starts and (num_id, ilvl) not in self._applied:
            self._applied.add((num_id, ilvl))
            ctr[ilvl] = starts[ilvl]
        elif ctr[ilvl] is None:
            ctr[ilvl] = lvl.start
        else:
            ctr[ilvl] += 1
        for k in range(ilvl + 1, 9):
            if levels.get(k) is None or levels[k].restart != 0:
                ctr[k] = None

        def sub(m):
            k = int(m.group(1)) - 1
            lv = levels.get(k, _Level())
            value = ctr[k] if ctr[k] is not None else lv.start
            return format_number(value, lv.fmt if lv.fmt != "bullet" else "decimal")

        return re.sub(r"%([1-9])", sub, lvl.text)


# ---------- ย่อหน้า ----------

@dataclass
class _Run:
    text: str
    bold: bool
    size: int  # half-points


@dataclass
class _Para:
    runs: list[_Run] = field(default_factory=list)
    text: str = ""
    images: int = 0
    extras: list["_Para"] = field(default_factory=list)  # กล่องข้อความ
    footnote_ids: list[str] = field(default_factory=list)
    style_id: Optional[str] = None
    num: Optional[tuple[int, int]] = None  # (numId, ilvl) ที่ใช้จริง
    has_numpr: bool = False


class _Reader:
    def __init__(self, styles: _Styles, numbering: _Numbering, footnotes: dict[str, str]):
        self.styles = styles
        self.numbering = numbering
        self.footnotes = footnotes

    # -- run / paragraph content --
    def _run_props(self, r: etree._Element, p_style: Optional[str], p_bold: Optional[bool], p_size: int) -> tuple[bool, int]:
        rpr = r.find("w:rPr", NS)
        r_style = _val(rpr.find("w:rStyle", NS)) if rpr is not None else None
        bold = _rpr_bold(rpr)
        if bold is None:
            bold = self.styles.char_bold(r_style)
        if bold is None:
            bold = p_bold
        size = _rpr_size(rpr)
        if size is None:
            size = self.styles.char_size(r_style)
        if size is None:
            size = p_size
        return bool(bold), size

    def paragraph(self, p: etree._Element, with_numbering: bool = True) -> _Para:
        info = _Para()
        ppr = p.find("w:pPr", NS)
        info.style_id = _val(ppr.find("w:pStyle", NS)) if ppr is not None else None
        p_bold = self.styles.para_bold(info.style_id)
        p_size = self.styles.para_size(info.style_id)
        parts: list[str] = []
        self._walk(p, info, parts, info.style_id, p_bold, p_size)
        info.text = "".join(parts)
        if with_numbering:
            self._resolve_numbering(ppr, info)
        return info

    def _resolve_numbering(self, ppr: Optional[etree._Element], info: _Para) -> None:
        np_ = ppr.find("w:numPr", NS) if ppr is not None else None
        num_id: Optional[int] = None
        ilvl: Optional[int] = None
        if np_ is not None:
            n, lv = _val(np_.find("w:numId", NS)), _val(np_.find("w:ilvl", NS))
            num_id = int(n) if n is not None else None
            ilvl = int(lv) if lv is not None else None
        if num_id is None:
            st = self.styles.num_pr(info.style_id)
            if st is not None:
                num_id = st[0]
                if ilvl is None:
                    ilvl = st[1]
        if num_id is None:
            return
        info.has_numpr = num_id != 0
        if ilvl is None:
            ilvl = self.numbering.style_level(num_id, info.style_id) or 0
        info.num = (num_id, ilvl)

    def numbering_label(self, info: _Para) -> Optional[str]:
        if info.num is None:
            return None
        return self.numbering.label(*info.num)

    def _walk(self, el, info: _Para, parts: list[str], p_style, p_bold, p_size) -> None:
        for child in el:
            tag = child.tag
            if not isinstance(tag, str):
                continue
            if tag == _w("r"):
                self._run(child, info, parts, p_style, p_bold, p_size)
            elif tag in (_w("pPr"), _w("del"), _w("moveFrom"), _w("proofErr"), _w("bookmarkStart"), _w("bookmarkEnd")):
                continue
            elif tag == f"{{{MC}}}AlternateContent":
                choice = child.find(f"{{{MC}}}Choice")
                node = choice if choice is not None else child.find(f"{{{MC}}}Fallback")
                if node is not None:
                    self._walk(node, info, parts, p_style, p_bold, p_size)
            else:
                self._walk(child, info, parts, p_style, p_bold, p_size)  # hyperlink, ins, smartTag, sdt, fldSimple ...

    def _run(self, r, info: _Para, parts: list[str], p_style, p_bold, p_size) -> None:
        bold, size = self._run_props(r, p_style, p_bold, p_size)
        buf: list[str] = []
        for c in r:
            tag = c.tag
            if tag == _w("t"):
                buf.append(c.text or "")
            elif tag == _w("tab"):
                buf.append("\t")
            elif tag in (_w("br"), _w("cr")):
                buf.append("\n")
            elif tag == _w("noBreakHyphen"):
                buf.append("-")
            elif tag == _w("footnoteReference"):
                fid = c.get(_w("id"))
                if fid is not None:
                    info.footnote_ids.append(fid)
            elif tag == f"{{{MC}}}AlternateContent":
                choice = c.find(f"{{{MC}}}Choice")
                node = choice if choice is not None else c.find(f"{{{MC}}}Fallback")
                if node is not None:
                    self._scan_graphics(node, info)
            elif tag in (_w("drawing"), _w("pict"), _w("object")):
                self._scan_graphics(c, info)
        text = "".join(buf)
        if text:
            parts.append(text)
            info.runs.append(_Run(text, bold, size))

    def _scan_graphics(self, el, info: _Para) -> None:
        """นับรูป และเก็บย่อหน้าในกล่องข้อความ (ไม่ลงลึกซ้ำใน txbxContent)"""
        for tx in el.iter(_w("txbxContent")):
            for p in tx.findall("w:p", NS):
                info.extras.append(self.paragraph(p, with_numbering=False))
        info.images += len(list(el.iter(f"{{{A}}}blip"))) + len(list(el.iter(f"{{{V}}}imagedata")))


# ---------- ตาราง ----------

def _is_true(el: Optional[etree._Element]) -> bool:
    return el is not None and _on(el) is not False


# ---------- โครงสร้างกลาง ----------

@dataclass
class _Item:
    index: int
    kind: str  # empty | heading | para | list | table
    text: str = ""  # ข้อความหลัง normalize รวมเลขอัตโนมัติ (ตาราง = text ของตาราง)
    blocks: list[Block] = field(default_factory=list)  # block ของเนื้อหา (ไม่รวมหัวข้อ); order ใส่ตอนประกอบ unit
    blocks_extra: list[Block] = field(default_factory=list)  # ย่อหน้าจากกล่องข้อความ ต่อหลังย่อหน้านี้
    footnotes: list[str] = field(default_factory=list)
    images: int = 0
    tally: _Tally = field(default_factory=_Tally)
    # หัวข้อ
    level: int = 0
    method: str = ""  # numbering | format
    heading_style: bool = False
    font_size: Optional[float] = None
    bold: Optional[bool] = None
    size_half: int = 0
    all_bold: bool = False
    numbered: bool = False
    end_index: Optional[int] = None  # หัวข้อที่รวมสองย่อหน้า


_RE_CHAPTER = re.compile(r"^บทที่\s*\d+")
_RE_CHAPTER_ONLY = re.compile(r"^บทที่\s*\d+$")
_RE_NNN = re.compile(r"^\d+\.\d+\.\d+\s")
_RE_NN = re.compile(r"^\d+\.\d+\s")
APPENDIX = "ภาคผนวก"


def _dominant_size(runs: list[_Run]) -> int:
    weight: Counter = Counter()
    for r in runs:
        if r.text.strip():
            weight[r.size] += len(r.text.strip())
    return weight.most_common(1)[0][0] if weight else 0


def _classify_heading(item: _Item, body_half: int) -> bool:
    """ตั้ง level/method ให้ item ถ้าเป็นหัวข้อ; คืนว่าเป็นหัวข้อหรือไม่"""
    t = item.text
    if not t or len(t) > MAX_HEADING_CHARS or not item.all_bold:
        return False
    chapter = bool(_RE_CHAPTER.match(t))
    nnn = bool(_RE_NNN.match(t))
    nn = bool(_RE_NN.match(t))
    item.numbered = chapter or nnn or nn
    if not (item.size_half > body_half or item.numbered):
        return False
    if chapter:
        item.level = 1
    elif nnn:
        item.level = 3
    elif nn:
        item.level = 2
    else:
        item.level = 1 if item.size_half >= body_half + 2 * HEADING_BIG_EXTRA_PT else 2
    item.method = "numbering" if item.numbered else "format"
    return True


@dataclass
class DocxResult:
    units: list[Unit] = field(default_factory=list)
    has_outline: bool = False
    producer: Optional[str] = None
    page_equivalent: Optional[int] = None


def _parser() -> etree.XMLParser:
    return etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True)


def _load(z: zipfile.ZipFile, name: str, required: bool = False) -> Optional[etree._Element]:
    try:
        data = z.read(name)
    except KeyError:
        if required:
            raise ExtractionError(ErrorCode.CORRUPT, f"DOCX ไม่มี {name}")
        return None
    try:
        return etree.fromstring(data, _parser())
    except etree.XMLSyntaxError as e:
        raise ExtractionError(ErrorCode.CORRUPT, f"อ่าน {name} ไม่ได้: {e}") from e


def _body_elements(body: etree._Element) -> Iterator[etree._Element]:
    for child in body:
        if child.tag in (_w("p"), _w("tbl")):
            yield child
        elif child.tag == _w("sdt"):
            content = child.find("w:sdtContent", NS)
            if content is not None:
                yield from _body_elements(content)


class _Builder:
    def __init__(self, reader: _Reader):
        self.r = reader

    # -- footnotes --
    def footnote_texts(self, ids: list[str], tally: _Tally) -> list[str]:
        out = []
        for fid in ids:
            raw = self.r.footnotes.get(fid)
            if raw is None:
                continue
            text = tally.norm(raw)
            if text:
                out.append(FOOTNOTE_PREFIX + text)
        return out

    # -- table --
    def table(self, tbl: etree._Element, tally: _Tally) -> tuple[Optional[Table], str, int]:
        rows: list[list[str]] = []
        lines: list[str] = []
        merged: dict[tuple[int, int], MergedCell] = {}
        origin: dict[int, tuple[int, int]] = {}
        header_rows = 0
        header_open = True
        images = 0
        for r_i, tr in enumerate(tbl.findall("w:tr", NS)):
            if header_open and _is_true(tr.find("w:trPr/w:tblHeader", NS)):
                header_rows += 1
            else:
                header_open = False
            row: list[str] = []
            line: list[str] = []
            col = 0
            for tc in tr.findall("w:tc", NS):
                tcpr = tc.find("w:tcPr", NS)
                span = int(_val(tcpr.find("w:gridSpan", NS)) or 1) if tcpr is not None else 1
                vm = tcpr.find("w:vMerge", NS) if tcpr is not None else None
                if vm is not None and _val(vm) != "restart":  # ต่อจากเซลล์ข้างบน
                    key = origin.get(col)
                    if key is not None:
                        merged[key].rowspan += 1
                    row.extend([""] * span)
                else:
                    text, imgs = self._cell_text(tc, tally)
                    images += imgs
                    row.append(text)
                    row.extend([""] * (span - 1))
                    if text:
                        line.append(text)
                    if span > 1 or vm is not None:
                        merged[(r_i, col)] = MergedCell(row=r_i, col=col, rowspan=1, colspan=span)
                        origin[col] = (r_i, col)
                    else:
                        origin.pop(col, None)
                col += span
            rows.append(row)
            if line:
                lines.append(" | ".join(line))
        text = "\n".join(lines)
        if not text:
            return None, "", images
        return Table(rows=rows, header_rows=header_rows, merged_cells=list(merged.values())), text, images

    def _cell_text(self, tc: etree._Element, tally: _Tally) -> tuple[str, int]:
        pieces: list[str] = []
        images = 0
        for child in tc:
            if child.tag == _w("p"):
                info = self.r.paragraph(child)
                images += info.images
                label = self.r.numbering_label(info)
                raw = info.text if not label else (f"{label} {info.text}" if info.text.strip() else "")
                t = tally.norm(raw)
                if t:
                    pieces.append(t)
            elif child.tag == _w("tbl"):
                _, text, imgs = self.table(child, tally)
                images += imgs
                if text:
                    pieces.append(text.replace("\n", " "))
        return " ".join(pieces), images

    # -- body --
    def items(self, body: etree._Element) -> list[_Item]:
        items: list[_Item] = []
        for index, el in enumerate(_body_elements(body)):
            item = _Item(index=index, kind="empty")
            if el.tag == _w("tbl"):
                table, text, images = self.table(el, item.tally)
                item.images = images
                if table is not None:
                    item.kind = "table"
                    item.text = text
                    item.blocks = [Block(type=BlockType.TABLE, text=text, order=0, table=table)]
            else:
                info = self.r.paragraph(el)
                label = self.r.numbering_label(info)
                item.images = info.images
                raw = info.text.strip() if not label else (f"{label} {info.text}" if info.text.strip() else "")
                text = item.tally.norm(raw)
                item.text = text
                if text:
                    item.kind = "para"
                    item.all_bold = bool(info.runs) and all(r.bold for r in info.runs if r.text.strip())
                    item.size_half = _dominant_size(info.runs)
                    item.heading_style = self.r.styles.is_heading_style(info.style_id)
                    item.font_size = item.size_half / 2 if item.size_half else None
                    item.bold = item.all_bold if info.runs else None
                    if info.has_numpr:
                        item.kind = "list"
                for extra in info.extras:  # รูปในกล่องข้อความนับรวมอยู่ใน info.images แล้ว
                    et = item.tally.norm(extra.text.strip())
                    if et:
                        if item.kind == "empty":
                            item.kind = "para"
                        item.blocks_extra.append(Block(type=BlockType.PARAGRAPH, text=et, order=0))
                item.footnotes = self.footnote_texts(info.footnote_ids, item.tally)
                if item.kind == "empty" and item.footnotes:
                    item.kind = "para"
                    item.text = ""
            items.append(item)
        return items


def _paragraph_block(item: _Item, kind: BlockType) -> Block:
    return Block(type=kind, text=item.text, order=0, font_size=item.font_size, bold=item.bold)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_docx(path: str | Path) -> DocxResult:
    t_start = time.perf_counter()
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        raise ExtractionError(ErrorCode.CORRUPT, f"DOCX เสียหาย: {e}") from e
    with z:
        document = _load(z, "word/document.xml", required=True)
        styles = _Styles(_load(z, "word/styles.xml"))
        numbering = _Numbering(_load(z, "word/numbering.xml"))
        footnotes = _read_footnotes(_load(z, "word/footnotes.xml"))
        app = _load(z, "docProps/app.xml")
    body = document.find("w:body", NS)
    if body is None:
        raise ExtractionError(ErrorCode.CORRUPT, "DOCX ไม่มี body")

    items = _Builder(_Reader(styles, numbering, footnotes)).items(body)

    # body_size = ขนาดที่พบมากที่สุดในย่อหน้าระดับ body ที่มีข้อความ
    sizes = Counter(i.size_half for i in items if i.kind in ("para", "list") and i.size_half)
    body_half = sizes.most_common(1)[0][0] if sizes else styles.default_size

    headings = _mark_headings(items, body_half)
    units = _build_units(items, headings)
    for i, u in enumerate(units):
        u.index = i
    elapsed = (time.perf_counter() - t_start) * 1000
    per = elapsed / max(len(units), 1)
    for u in units:
        u.elapsed_ms = per
    producer, pages = _app_props(app)
    return DocxResult(units=units, has_outline=bool(headings), producer=producer, page_equivalent=pages)


def _read_footnotes(root: Optional[etree._Element]) -> dict[str, str]:
    out: dict[str, str] = {}
    if root is None:
        return out
    for fn in root.findall("w:footnote", NS):
        if fn.get(_w("type")) in ("separator", "continuationSeparator", "continuationNotice"):
            continue
        texts = []
        for t in fn.iter(_w("t")):
            texts.append(t.text or "")
        out[fn.get(_w("id"))] = "".join(texts).strip()
    return out


def _app_props(app: Optional[etree._Element]) -> tuple[Optional[str], Optional[int]]:
    if app is None:
        return None, None
    ns = {"a": app.nsmap.get(None, "")}
    def text(tag):
        el = app.find(f"a:{tag}", ns) if ns["a"] else None
        return el.text.strip() if el is not None and el.text else None
    pages = text("Pages")
    return text("Application"), int(pages) if pages and pages.isdigit() else None


def _mark_headings(items: list[_Item], body_half: int) -> list[_Item]:
    """ตัดสินหัวข้อ รวม "บทที่ N" + ชื่อบท ปรับระดับหลังภาคผนวก; คืนรายการ item ที่เป็นหัวข้อ (ตามลำดับ)"""
    for it in items:
        if it.kind in ("para", "list") and it.text and _classify_heading(it, body_half):
            it.kind = "heading"
    content = [i for i in items if i.kind != "empty"]
    # รวม "บทที่ N" กับชื่อบทที่ไม่มีเลขในย่อหน้าถัดไป
    merged_away: set[int] = set()
    for pos, it in enumerate(content):
        if it.kind == "heading" and _RE_CHAPTER_ONLY.match(it.text) and pos + 1 < len(content):
            nxt = content[pos + 1]
            if nxt.kind == "heading" and not nxt.numbered:
                it.text = f"{it.text} {nxt.text}"
                it.end_index = nxt.index
                it.images += nxt.images
                it.tally.add(nxt.tally)
                it.footnotes += nxt.footnotes
                merged_away.add(id(nxt))
    headings = [i for i in items if i.kind == "heading" and id(i) not in merged_away]
    for it in items:
        if id(it) in merged_away:
            it.kind = "empty"
            it.text = ""
    shift = 0
    for it in headings:
        if it.text.startswith(APPENDIX):
            it.level = 1
            shift = 1
        else:
            it.level += shift
    return headings


def _label_unique(label: str, seen: Counter) -> str:
    seen[label] += 1
    return label if seen[label] == 1 else f"{label} (ช่วงที่ {seen[label]})"


def _build_units(items: list[_Item], headings: list[_Item]) -> list[Unit]:
    units: list[Unit] = []
    seen: Counter = Counter()
    stack: list[tuple[int, _Item]] = []
    heading_ids = {id(h) for h in headings}

    sections: list[list] = []  # [heading, path items, content items, images ในย่อหน้าที่ไม่มีข้อความ]
    current: list = [None, [], [], 0]
    for it in items:
        if id(it) in heading_ids:
            sections.append(current)
            while stack and stack[-1][1].level >= it.level:
                stack.pop()
            stack.append((it.level, it))
            current = [it, [h for _, h in stack], [], 0]
        elif it.kind in ("para", "list", "table"):
            current[2].append(it)
        else:
            current[3] += it.images
    sections.append(current)
    for s in sections:
        s[2] = [c for c in s[2] if c.text or c.blocks or c.footnotes or c.blocks_extra]

    # หัวข้อที่ไม่มีเนื้อหาและไม่ได้อยู่ใน heading_path ของ unit ใดเลย (หัวข้อระดับเดียวกันตามมาทันที) จะทำให้ข้อความหาย
    # จึงยกข้อความของมันไปไว้ต้นเนื้อหาของ unit ถัดไป (ไม่สร้าง unit ให้มัน) / ท้ายสุดไปต่อท้าย unit สุดท้าย
    used = {id(h) for s in sections if s[2] for h in s[1]}
    pending: list[_Item] = []
    last_with_content = max((n for n, s in enumerate(sections) if s[2]), default=None)
    for n, (heading, path_items, content, loose_images) in enumerate(sections):
        if not content:
            if heading is not None and id(heading) not in used:
                pending.append(heading)
            continue
        carried = [_orphan_item(h) for h in pending]
        pending = []
        if n == last_with_content:
            content = carried + content + [_orphan_item(h) for h in _trailing_orphans(sections[n + 1:], used)]
        else:
            content = carried + content
        path = [h.text for h in path_items]
        for part_no, (part, total) in enumerate(_split(heading, content), 1):
            units.append(_make_unit(heading, path, part, part_no, total, seen, loose_images if part_no == 1 else 0))
    return units


def _trailing_orphans(sections: list[list], used: set) -> list[_Item]:
    return [s[0] for s in sections if not s[2] and s[0] is not None and id(s[0]) not in used]


def _orphan_item(h: _Item) -> _Item:
    """หัวข้อเปล่าที่ตกหล่น -> ย่อหน้าธรรมดาที่ index เดิม (ยังนับผล normalizer ของมันไว้)"""
    return _Item(index=h.index, kind="para", text=h.text, images=h.images, tally=h.tally, footnotes=h.footnotes,
                 font_size=h.font_size, bold=h.bold)


def _split(heading: Optional[_Item], content: list[_Item]):
    head_len = len(heading.text) + 1 if heading else 0
    parts: list[list[_Item]] = []
    cur: list[_Item] = []
    cur_len = head_len
    for it in content:
        size = _item_len(it)
        if cur and cur_len + size > MAX_UNIT_CHARS:
            parts.append(cur)
            cur, cur_len = [], head_len
        cur.append(it)
        cur_len += size
    if cur:
        parts.append(cur)
    return [(p, len(parts)) for p in parts]


def _item_len(it: _Item) -> int:
    return len(it.text) + 1 + sum(len(b.text) + 1 for b in it.blocks_extra) + sum(len(f) + 1 for f in it.footnotes)


def _make_unit(heading: Optional[_Item], path: list[str], part: list[_Item], part_no: int, total: int,
               seen: Counter, loose_images: int = 0) -> Unit:
    blocks: list[Block] = []
    tally = _Tally()
    images = loose_images

    def add(b: Block) -> None:
        blocks.append(b.model_copy(update={"order": len(blocks)}))

    if heading is not None:
        add(Block(type=BlockType.HEADING, text=heading.text, order=0, level=heading.level,
                  font_size=heading.font_size, bold=heading.bold))
        tally.add(heading.tally)  # หัวข้อซ้ำในทุกส่วนของ unit ที่ถูกแบ่ง ผลนับจึงซ้ำตาม
        if part_no == 1:
            images += heading.images
    footnote_blocks: list[Block] = []
    for it in part:
        if it.kind == "table":
            add(it.blocks[0])
        else:
            add(_paragraph_block(it, BlockType.LIST if it.kind == "list" else BlockType.PARAGRAPH))
        for extra in it.blocks_extra:
            add(extra)
        footnote_blocks += [Block(type=BlockType.PARAGRAPH, text=f, order=0) for f in it.footnotes]
        tally.add(it.tally)
        images += it.images
    for fb in footnote_blocks:
        add(fb)

    first = min(it.index for it in part)
    if heading is not None and part_no == 1:
        first = min(first, heading.index)
    last = max(it.index for it in part)
    label = LABEL_SEP.join(path) if path else PREFACE_LABEL
    if total > 1:
        label += f" (ส่วน {part_no}/{total})"
    label = _label_unique(label, seen)
    text = "\n".join(b.text for b in blocks)
    signals: dict[str, float] = {
        "char_count": float(len(text.strip())),
        "block_count": float(len(blocks)),
        "docx:image_count": float(images),
    }
    if heading is not None:
        signals[f"docx:heading_by_{heading.method}"] = 1.0
        if heading.heading_style:
            signals["docx:heading_style"] = 1.0
    for name, n in tally.fixes.items():
        signals[f"norm_fix:{name}"] = float(n)
    for name, n in tally.signals.items():
        signals[f"norm_sig:{name}"] = float(n)
    return Unit(
        index=0,
        type=UnitType.SECTION,
        position={"heading_path": list(path), "paragraphs": [first, last]},
        label=label,
        status=UnitStatus.OK,
        method=ExtractionMethod.OFFICE_XML,
        blocks=blocks,
        text=text,
        text_hash=_hash(text),
        quality=QualityReport(score=None, flags=[], normalization_changes=sum(tally.fixes.values())),
        signals=signals,
    )
