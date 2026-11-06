"""[M3] Thai-Aware Hierarchical Chunking with Sentence Boundary Disambiguation

สถาปัตยกรรม:
- ตัด Unit.text เป็นประโยค (Child Chunks) ด้วย Thai-aware SBD
- รวมประโยคที่อยู่ติดกันเป็น Parent Chunks (3–5 ประโยค ไม่ข้ามหน้า)
- จัดเก็บ char_start / char_end offset ครบทุก chunk เพื่อ citation tracking ระดับประโยค
- ไม่ข้ามขอบเขต Unit (กฎเหล็กของ contract)

สองชั้น chunking:
  Child  — ประโยคเดี่ยว (ใช้ embed + retrieve เพื่อ citation แม่นระดับประโยค)
  Parent — หน้าต่าง 3–5 ประโยค (ส่งให้ LLM เพื่อบริบทครบถ้วน)

กลยุทธ์ตัดประโยค (Thai SBD ไม่มี whitespace ระหว่างคำ):
  ลำดับที่ 1 — ใช้ pythainlp.sent_tokenize (crfcut / thaisumcut) เป็น baseline
  ลำดับที่ 2 — ถ้า pythainlp ไม่พร้อมหรือตัดสั้นเกินไป fallback ไป Rule-based SBD
  Rule-based SBD:
    • ตัดที่ newline (\\n) ซึ่งในเอกสาร PDF มักเป็นขอบเขตประโยคจริง
    • ตัดที่เครื่องหมายวรรคตอนไทย/ลาติน (.!?。)
    • Merge ชิ้นที่สั้นกว่า MIN_SENT_CHARS เข้าชิ้นถัดไป
  ทั้งสองเส้นทางรับประกันว่า:
    • ข้อความที่ต่อกันใน Child ไม่ข้ามขอบเขต Unit
    • join ประโยคทั้งหมดกลับคืนได้ (ไม่ตัดอักขระหาย)

Noisy Whitespace Tolerance:
  ใช้ normalizer กลางก่อนตัดแล้วจึง map offset กลับสู่ text ดั้งเดิม
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional, Sequence

from app.models import Unit, UnitStatus
from app.thai.normalizer import normalize

# ---------- ค่าคงที่ ----------

# ขนาด Child (ตัวอักษร ไม่ใช่คำ เพราะภาษาไทยไม่มีช่องว่าง)
MIN_SENT_CHARS = 10       # ประโยคสั้นกว่านี้ให้ merge เข้าถัดไป
MAX_CHILD_CHARS = 400     # ถ้าประโยคยาวเกิน ตัดที่ขอบ grapheme cluster ที่ใกล้ที่สุด

# ขนาด Parent (หน้าต่าง)
PARENT_WINDOW_MIN = 3     # ประโยคต่ำสุดใน parent
PARENT_WINDOW_MAX = 5     # ประโยคสูงสุดใน parent
PARENT_OVERLAP = 1        # ประโยคที่ซ้อนทับระหว่าง parent ถัดกัน (sliding window)

# ---------- โครงสร้างข้อมูล ----------


@dataclass
class ChildChunk:
    """ประโยคเดี่ยว — หน่วยที่เล็กที่สุดสำหรับ embedding + citation"""
    chunk_id: str
    parent_id: str            # FK → ParentChunk.chunk_id
    document_id: str
    unit_index: int           # Unit.index ที่มาจาก
    sentence_index: int       # ลำดับประโยคใน unit เริ่มที่ 0
    content: str              # ข้อความหลัง normalize
    char_start: int           # offset ใน Unit.text ดั้งเดิม (ก่อน normalize)
    char_end: int
    position: dict            # copy จาก Unit.position (เช่น {"page": 5})
    label: str                # copy จาก Unit.label (เช่น "หน้า 5")
    metadata: dict = field(default_factory=dict)

    @property
    def contextual_text(self) -> str:
        """ข้อความพร้อมบริบทเอกสาร (Anthropic Contextual Retrieval 2024)
        ผนวกหัวข้อ โครงสร้างเอกสาร และชื่อไฟล์ เพื่อให้ค้นหาได้แม่นยำระดับองค์กรโดยไม่สูญเสียบริบท
        ในขณะที่ content ดั้งเดิมยังคงความสมบูรณ์สำหรับทำ Citation Tracking
        """
        parts = []
        if self.document_id:
            # แปลงชื่อไฟล์เป็นชื่อเอกสารสะอาด เช่น E_ONE_REPORT_CPALLT.PDF -> E ONE REPORT CPALLT
            doc_clean = re.sub(r"\.[a-zA-Z0-9]+$", "", self.document_id).replace("_", " ")
            parts.append(f"[{doc_clean}]")
        if self.label:
            parts.append(f"[{self.label}]")
        if "heading_path" in self.position and self.position["heading_path"]:
            headings = " > ".join(self.position["heading_path"])
            parts.append(f"[{headings}]")
        parts.append(self.content)
        return " ".join(parts)


@dataclass
class ParentChunk:
    """หน้าต่าง 3–5 ประโยค — ส่งให้ LLM เพื่อบริบทครบ"""
    chunk_id: str
    document_id: str
    unit_index: int
    content: str              # ข้อความรวม (join ด้วย separator ที่เหมาะสม)
    char_start: int           # offset เริ่มต้น (จาก child แรก)
    char_end: int             # offset สิ้นสุด (จาก child สุดท้าย)
    child_ids: list[str]      # ลำดับ chunk_id ของ child ที่อยู่ใน parent
    sentence_count: int
    position: dict
    label: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkResult:
    """ผลลัพธ์ chunking ของ Unit เดียว"""
    unit_index: int
    children: list[ChildChunk]
    parents: list[ParentChunk]
    skipped_reason: Optional[str] = None  # ถ้า unit ถูกข้าม


# ---------- Thai SBD (Sentence Boundary Detection) ----------

# เครื่องหมายจบประโยค ภาษาไทยและลาติน
_SENT_END_RE = re.compile(
    r"(?:"
    r"[.!?。！？]"                      # จุดลาติน / ญี่ปุ่น
    r")"
    r"(?=\s|$)"                          # ตามด้วยช่องว่าง/จบข้อความ
)

# newline ตรงๆ — ใน PDF มักเป็นจุดตัดที่ดีที่สุด
_NEWLINE_RE = re.compile(r"\n+")


def _split_by_newlines_and_punct(text: str) -> list[str]:
    """Rule-based SBD: ตัดด้วย newline ก่อน แล้วตัดด้วยเครื่องหมายวรรคตอน

    คืน list ของ span (ข้อความย่อย) ที่ join กลับได้ครบ
    """
    # ขั้นที่ 1: แบ่งที่ newline
    lines = _NEWLINE_RE.split(text)

    sentences: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # ขั้นที่ 2: แบ่งที่เครื่องหมายวรรคตอน (รักษา delimiter ไว้)
        sub_sentences: list[str] = []
        prev = 0
        for m in _SENT_END_RE.finditer(line):
            end = m.end()
            part = line[prev:end].strip()
            if part:
                sub_sentences.append(part)
            prev = end
        tail = line[prev:].strip()
        if tail:
            sub_sentences.append(tail)

        if sub_sentences:
            sentences.extend(sub_sentences)
        elif line.strip():
            sentences.append(line.strip())

    return sentences if sentences else ([text.strip()] if text.strip() else [])


def _try_pythainlp_sent_tokenize(text: str) -> Optional[list[str]]:
    """พยายามใช้ pythainlp.sent_tokenize ถ้ามีใน environment

    คืน list[str] ถ้าสำเร็จ หรือ None ถ้า library ไม่พร้อม
    """
    try:
        from pythainlp.tokenize import sent_tokenize  # type: ignore[import]
        result = sent_tokenize(text, engine="crfcut")
        return [s for s in result if s.strip()] or None
    except Exception:
        return None


def segment_sentences(text: str) -> list[str]:
    """ตัดข้อความเป็นประโยค (Thai-aware SBD)

    เส้นทาง 1: pythainlp crfcut (ถ้าพร้อม)
    เส้นทาง 2: rule-based (newline + punct)

    คืน list ประโยค (strip แล้ว) — ไม่รวมสตริงว่าง
    """
    text = text.strip()
    if not text:
        return []

    # ลองเส้นทาง 1
    sents = _try_pythainlp_sent_tokenize(text)
    if sents and len(sents) > 0:
        return _merge_short_sentences(sents)

    # เส้นทาง 2: rule-based
    sents = _split_by_newlines_and_punct(text)
    return _merge_short_sentences(sents)


def _merge_short_sentences(sents: list[str]) -> list[str]:
    """Merge ประโยคที่สั้นกว่า MIN_SENT_CHARS เข้ากับประโยคก่อนหน้า (หรือถัดไป)"""
    if not sents:
        return []

    merged: list[str] = []
    buffer = ""

    for s in sents:
        s = s.strip()
        if not s:
            continue
        buffer = (buffer + " " + s).strip() if buffer else s
        if len(buffer) >= MIN_SENT_CHARS:
            merged.append(buffer)
            buffer = ""

    # tail
    if buffer:
        if merged:
            merged[-1] = (merged[-1] + " " + buffer).strip()
        else:
            merged.append(buffer)

    return merged


def _truncate_at_grapheme(text: str, max_chars: int) -> tuple[str, str]:
    """ตัดข้อความที่ขอบ grapheme cluster ที่ใกล้กับ max_chars ที่สุด

    คืน (ส่วนแรก, ส่วนที่เหลือ)
    ใช้ regex \\X สำหรับ grapheme cluster (ไทยสระ+พยัญชนะ+วรรณยุกต์ = 1 cluster)
    """
    try:
        import regex  # pypi package 'regex' รองรับ \\X
        clusters = regex.findall(r"\X", text)
    except ImportError:
        # fallback: ตัดที่ code point ธรรมดา
        clusters = list(text)

    total = 0
    cut = 0
    for i, c in enumerate(clusters):
        total += len(c)
        if total > max_chars:
            break
        cut = i + 1

    head = "".join(clusters[:cut])
    tail = "".join(clusters[cut:])
    return head, tail


# ---------- Offset Mapping ----------


def _find_offsets_in_original(
    sentences: list[str],
    original_text: str,
) -> list[tuple[int, int]]:
    """หา char_start / char_end ของแต่ละประโยคใน original_text

    ใช้การค้นหาแบบ forward-scan เพื่อรองรับกรณีที่ normalize อาจเปลี่ยน
    ช่องว่างหรือตัวอักขระบางตัว

    คืน list[(start, end)] ขนาดเดียวกับ sentences
    """
    offsets: list[tuple[int, int]] = []
    search_start = 0

    for sent in sentences:
        # ลองหา substring ตรงๆ ก่อน
        idx = original_text.find(sent, search_start)
        if idx >= 0:
            offsets.append((idx, idx + len(sent)))
            search_start = idx + len(sent)
        else:
            # ไม่เจอ substring ตรง → approximate จากตำแหน่ง offset ล่าสุด
            last_end = offsets[-1][1] if offsets else search_start
            approx_end = min(last_end + len(sent), len(original_text))
            offsets.append((last_end, approx_end))
            search_start = approx_end

    return offsets


# ---------- Core Chunking ----------


def chunk_unit(
    unit: Unit,
    document_id: str,
    *,
    parent_window: int = PARENT_WINDOW_MAX,
    parent_overlap: int = PARENT_OVERLAP,
) -> ChunkResult:
    """ตัด Unit หนึ่งออกเป็น Child + Parent Chunks

    กฎ:
    - ข้ามถ้า unit.status != ok หรือ text ว่าง
    - ไม่ข้ามขอบเขต Unit เด็ดขาด (ป้องกัน context leak ข้ามหน้า)
    - chunk_id ใหม่ทุกครั้ง (UUID4)
    - parent_id ของ child ชี้ไปยัง parent ที่ครอบมัน (parent แรก)
    """
    if unit.status != UnitStatus.OK:
        return ChunkResult(
            unit_index=unit.index,
            children=[],
            parents=[],
            skipped_reason=f"unit.status={unit.status.value}",
        )

    original_text = unit.text
    if not original_text.strip():
        return ChunkResult(
            unit_index=unit.index,
            children=[],
            parents=[],
            skipped_reason="empty_text",
        )

    # normalize ก่อนตัด
    norm_text = normalize(original_text)

    # ตัดประโยค
    raw_sentences = segment_sentences(norm_text)
    if not raw_sentences:
        return ChunkResult(
            unit_index=unit.index,
            children=[],
            parents=[],
            skipped_reason="no_sentences",
        )

    # ตัดประโยคที่ยาวเกิน MAX_CHILD_CHARS (sliding truncation)
    sentences: list[str] = []
    for s in raw_sentences:
        if len(s) <= MAX_CHILD_CHARS:
            sentences.append(s)
        else:
            remaining = s
            while len(remaining) > MAX_CHILD_CHARS:
                head, remaining = _truncate_at_grapheme(remaining, MAX_CHILD_CHARS)
                if head:
                    sentences.append(head)
            if remaining.strip():
                sentences.append(remaining.strip())

    if not sentences:
        return ChunkResult(
            unit_index=unit.index,
            children=[],
            parents=[],
            skipped_reason="no_sentences_after_truncation",
        )

    # หา offset ของแต่ละประโยคใน original_text
    offsets = _find_offsets_in_original(sentences, original_text)

    # กำหนด Parent windows (sliding window)
    # stride = parent_window - parent_overlap (ต้องมากกว่า 0)
    stride = max(1, parent_window - parent_overlap)
    n = len(sentences)

    # (sent_start_idx, sent_end_idx_excl, parent_chunk_id)
    parent_windows: list[tuple[int, int, str]] = []
    i = 0
    while i < n:
        end = min(i + parent_window, n)
        pid = str(uuid.uuid4())
        parent_windows.append((i, end, pid))
        if end == n:
            break
        i += stride

    # primary parent ของแต่ละ sentence (parent แรกที่ครอบ)
    sent_to_parent: list[str] = [""] * n
    for pw_start, pw_end, pid in parent_windows:
        for si in range(pw_start, pw_end):
            if sent_to_parent[si] == "":
                sent_to_parent[si] = pid

    # สร้าง Children
    children: list[ChildChunk] = []
    for si, (sent, (cs, ce)) in enumerate(zip(sentences, offsets)):
        cid = str(uuid.uuid4())
        children.append(
            ChildChunk(
                chunk_id=cid,
                parent_id=sent_to_parent[si],
                document_id=document_id,
                unit_index=unit.index,
                sentence_index=si,
                content=sent,
                char_start=cs,
                char_end=ce,
                position=dict(unit.position),
                label=unit.label,
            )
        )

    # สร้าง Parents
    parents: list[ParentChunk] = []
    child_id_by_si = {c.sentence_index: c.chunk_id for c in children}

    for pw_start, pw_end, pid in parent_windows:
        window_sents = sentences[pw_start:pw_end]
        window_offsets = offsets[pw_start:pw_end]
        window_child_ids = [child_id_by_si[si] for si in range(pw_start, pw_end)]

        combined = " ".join(window_sents)
        char_start = window_offsets[0][0]
        char_end = window_offsets[-1][1]

        parents.append(
            ParentChunk(
                chunk_id=pid,
                document_id=document_id,
                unit_index=unit.index,
                content=combined,
                char_start=char_start,
                char_end=char_end,
                child_ids=window_child_ids,
                sentence_count=len(window_sents),
                position=dict(unit.position),
                label=unit.label,
            )
        )

    return ChunkResult(
        unit_index=unit.index,
        children=children,
        parents=parents,
    )


def chunk_document(
    units: list[Unit],
    document_id: str,
    *,
    parent_window: int = PARENT_WINDOW_MAX,
    parent_overlap: int = PARENT_OVERLAP,
) -> tuple[list[ChildChunk], list[ParentChunk]]:
    """Chunk ทุก Unit ในเอกสาร

    คืน (all_children, all_parents) — รวมจากทุก unit ที่ status=ok
    unit ที่ status != ok จะถูกข้ามโดยอัตโนมัติ
    """
    all_children: list[ChildChunk] = []
    all_parents: list[ParentChunk] = []

    for unit in units:
        result = chunk_unit(
            unit,
            document_id,
            parent_window=parent_window,
            parent_overlap=parent_overlap,
        )
        all_children.extend(result.children)
        all_parents.extend(result.parents)

    return all_children, all_parents


# ---------- Utility สำหรับ Citation ----------


def get_parent_for_child(
    child: ChildChunk,
    parents: list[ParentChunk],
) -> Optional[ParentChunk]:
    """หา ParentChunk ที่ตรงกับ child.parent_id"""
    for p in parents:
        if p.chunk_id == child.parent_id:
            return p
    return None


def format_citation_context(
    children: list[ChildChunk],
    *,
    start_index: int = 1,
) -> list[dict]:
    """แปลง ChildChunks เป็น format สำหรับส่งให้ LLM พร้อม sentence tag [S1], [S2]...

    ตาม ALCE / LongCite framework — แต่ละ sentence มี sentence_id, text, metadata
    """
    context = []
    for i, child in enumerate(children):
        context.append(
            {
                "sentence_id": f"S{start_index + i}",
                "text": child.content,
                "metadata": {
                    "doc_id": child.document_id,
                    "unit_index": child.unit_index,
                    "label": child.label,
                    "char_start": child.char_start,
                    "char_end": child.char_end,
                    "position": child.position,
                },
            }
        )
    return context


def build_system_prompt() -> str:
    """สร้าง System Prompt มาตรฐานสำหรับ citation-grounded RAG

    ตาม ALCE + LongCite framework พร้อม inline sentence citation
    """
    return (
        "คุณคือระบบตอบคำถามองค์กรที่ต้องให้ข้อมูลที่ถูกต้องและสามารถตรวจสอบอ้างอิงได้เท่านั้น\n\n"
        "ข้อบังคับในการตอบ:\n"
        "1. ตอบคำถามโดยใช้เฉพาะข้อมูลจากบริบท [S1], [S2], ... ที่กำหนดให้เท่านั้น "
        "ห้ามใช้ความรู้ภายนอกเด็ดขาด\n"
        "2. ทุกๆ ประโยคในคำตอบของคุณ จะต้องปิดท้ายด้วยแท็กอ้างอิง เช่น [S1] หรือ [S1][S2] "
        "ที่สนับสนุนข้อเท็จจริงในประโยคนั้นๆ\n"
        "3. ห้ามสร้างประโยคที่ไม่สามารถหาแท็กอ้างอิงรองรับได้\n"
        "4. หากข้อมูลในบริบทไม่เพียงพอสำหรับการตอบคำถาม ให้ตอบว่า "
        "\"ไม่มีข้อมูลเพียงพอในเอกสารสำหรับการตอบคำถามนี้\" และห้ามใส่แท็กอ้างอิงใดๆ"
    )


# =============================================================================
# chunk ตามขนาด สำหรับ index ลง Qdrant (hybrid)
#
# ต่างจาก Child/Parent ด้านบน (prototype in-memory): chunk เดียวมีขนาดเป้าหมาย ``target_chars`` (CHUNK_TARGET_CHARS)
# ตัดตามขอบประโยคเท่านั้น ไม่ข้าม unit ซ้อนกัน 1 ประโยค และ offset ทุกตัวเป็น offset ใน ``Unit.text`` **ดั้งเดิม** (ใช้ cite ได้ตรง)
# ไม่ normalize ซ้ำก่อนตัด — ``Unit.text`` ผ่าน normalizer กลางมาจาก extractor แล้ว (contract) เลยตัดบนข้อความเดิมได้ offset ตรงเสมอ
#
# context line ของ unit ("[แถวแรกของตาราง: …]" / "[ป้ายก่อนหน้า: …]") ที่อยู่ก่อน chunk ถูกแปะหน้า chunk ตอน embed
# (``Chunk.embed_text``) แต่ไม่อยู่ใน ``Chunk.text`` และไม่นับใน offset
# =============================================================================

CONTEXT_LINE_PREFIXES = ("แถวแรกของตาราง", "ป้ายก่อนหน้า")
_CONTEXT_LINE_RE = re.compile(r"^\[(?:" + "|".join(CONTEXT_LINE_PREFIXES) + r"):.*\]$")
_LINE_RE = re.compile(r"[^\n]+")


@dataclass(frozen=True)
class SentenceSpan:
    """ประโยคหนึ่งใน chunk — offset ใน ``Unit.text`` ดั้งเดิม (``unit.text[char_start:char_end]``)"""

    char_start: int
    char_end: int


@dataclass(frozen=True)
class Chunk:
    unit_index: int
    chunk_index: int  # ลำดับ chunk ใน unit เริ่มที่ 0
    text: str  # = unit.text[char_start:char_end] (ไม่รวม context line)
    char_start: int
    char_end: int
    sentences: tuple[SentenceSpan, ...]
    context: str  # context line ที่อยู่ก่อน chunk ("" ถ้าไม่มี)
    position: dict
    label: str

    @property
    def embed_text(self) -> str:
        """ข้อความที่ส่งเข้า embedding: context line (ถ้ามี) ตามด้วยเนื้อ chunk"""
        return f"{self.context}\n{self.text}" if self.context else self.text


def _line_sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """แบ่งบรรทัด ``text[start:end]`` เป็นประโยคด้วย crfcut; ถ้า crfcut ใช้ไม่ได้/หา substring ไม่เจอ = ทั้งบรรทัดเป็นประโยคเดียว"""
    line = text[start:end]
    sents = _try_pythainlp_sent_tokenize(line) if len(line) >= 2 * MIN_SENT_CHARS else None
    if not sents or len(sents) == 1:
        return [(start, end)]
    spans, cursor = [], 0
    for sent in sents:
        sent = sent.strip()
        idx = line.find(sent, cursor)
        if not sent or idx < 0:
            return [(start, end)]
        spans.append((start + idx, start + idx + len(sent)))
        cursor = idx + len(sent)
    return spans


def split_sentence_spans(text: str) -> list[tuple[int, int]]:
    """ประโยคของ ``text`` เป็นช่วง (start, end) ใน text เดิม: ตัดที่ newline ก่อน แล้ว crfcut ภายในบรรทัด

    ช่วงที่สั้นกว่า ``MIN_SENT_CHARS`` รวมเข้ากับช่วงถัดไป (ช่วงสุดท้ายรวมเข้ากับช่วงก่อนหน้า) โดยช่วงที่รวมแล้วคือ
    ``text[start_แรก:end_สุดท้าย]`` ต่อเนื่อง ไม่มีอักขระถูกทิ้งหรือแต่งเพิ่ม
    """
    raw: list[tuple[int, int]] = []
    for m in _LINE_RE.finditer(text):
        seg = m.group()
        lead = len(seg) - len(seg.lstrip())
        stripped = seg.strip()
        if stripped:
            raw.extend(_line_sentence_spans(text, m.start() + lead, m.start() + lead + len(stripped)))
    merged: list[tuple[int, int]] = []
    buffer_start: Optional[int] = None
    for start, end in raw:
        begin = start if buffer_start is None else buffer_start
        if end - begin >= MIN_SENT_CHARS:
            merged.append((begin, end))
            buffer_start = None
        else:
            buffer_start = begin
    if buffer_start is not None and raw:
        tail_end = raw[-1][1]
        if merged:
            merged[-1] = (merged[-1][0], tail_end)
        else:
            merged.append((buffer_start, tail_end))
    return merged


def _split_long_span(text: str, start: int, end: int, limit: int) -> list[tuple[int, int]]:
    """ประโยคยาวเกิน ``limit`` ตัวอักษร -> ตัดที่ขอบ grapheme cluster เป็นชิ้น ≤ limit (offset ตรงกับ text เดิม)"""
    if end - start <= limit:
        return [(start, end)]
    import regex

    pieces, piece_start, pos = [], start, start
    for m in regex.finditer(r"\X", text[start:end]):
        cluster_end = start + m.end()
        if cluster_end - piece_start > limit and pos > piece_start:
            pieces.append((piece_start, pos))
            piece_start = pos
        pos = cluster_end
    pieces.append((piece_start, end))
    return pieces


def sentence_spans_for_chunking(text: str, target_chars: int) -> list[tuple[int, int]]:
    """``split_sentence_spans`` แล้วตัดประโยคที่ยาวเกิน ``target_chars`` — ไม่ขึ้นกับขนาดอื่น ใช้ซ้ำข้ามหลายขนาด chunk ไม่ได้
    (ขีดตัดขึ้นกับ target) ให้เรียก ``split_sentence_spans`` ครั้งเดียวแล้วส่งเข้า ``pack_chunks`` เมื่อต้องการ cache"""
    out: list[tuple[int, int]] = []
    for start, end in split_sentence_spans(text):
        out.extend(_split_long_span(text, start, end, target_chars))
    return out


def unit_context_lines(text: str) -> list[tuple[int, int, str]]:
    """context line ทั้งหมดใน ``Unit.text``: (start, end, ข้อความบรรทัด) เรียงตามตำแหน่ง"""
    out = []
    for m in _LINE_RE.finditer(text):
        line = m.group().strip()
        if _CONTEXT_LINE_RE.match(line):
            lead = len(m.group()) - len(m.group().lstrip())
            out.append((m.start() + lead, m.start() + lead + len(line), line))
    return out


def pack_chunks(unit: Unit, spans: list[tuple[int, int]], target_chars: int) -> list[Chunk]:
    """รวมประโยค (``spans`` ใน ``unit.text``) เป็น chunk ขนาด ≤ ``target_chars`` (ยกเว้นประโยคเดียวที่ยาวกว่า — ซึ่ง
    ``sentence_spans_for_chunking`` ตัดให้แล้ว) chunk ถัดไปเริ่มจากประโยคสุดท้ายของ chunk ก่อนหน้า (ซ้อน 1 ประโยค)
    และต้องเดินหน้าอย่างน้อย 1 ประโยคเสมอ; unit ที่สั้นกว่าเป้าหมายได้ chunk เดียว"""
    if target_chars <= 0:
        raise ValueError("target_chars ต้องมากกว่า 0")
    text, contexts = unit.text, unit_context_lines(unit.text)
    chunks: list[Chunk] = []
    i, n = 0, len(spans)
    while i < n:
        j = i + 1
        while j < n and spans[j][1] - spans[i][0] <= target_chars:
            j += 1
        start, end = spans[i][0], spans[j - 1][1]
        before = [line for s, e, line in contexts if e <= start]
        chunks.append(Chunk(
            unit_index=unit.index, chunk_index=len(chunks), text=text[start:end], char_start=start, char_end=end,
            sentences=tuple(SentenceSpan(s, e) for s, e in spans[i:j]),
            context=before[-1] if before else "", position=dict(unit.position), label=unit.label,
        ))
        if j >= n:
            break
        i = max(j - 1, i + 1)
    return chunks


def make_chunks(unit: Unit, target_chars: int) -> list[Chunk]:
    """chunk ของ unit หนึ่งตามขนาดเป้าหมาย — unit ที่ status != ok หรือไม่มีข้อความ = ไม่มี chunk"""
    if unit.status != UnitStatus.OK or not unit.text.strip():
        return []
    return pack_chunks(unit, sentence_spans_for_chunking(unit.text, target_chars), target_chars)


def make_document_chunks(units: Sequence[Unit], target_chars: int) -> list[Chunk]:
    return [c for unit in units for c in make_chunks(unit, target_chars)]
