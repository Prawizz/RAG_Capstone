"""[M6] Sentence-level Citation Builder & ALCE / LongCite Verification

สถาปัตยกรรมอ้างอิงระดับองค์กร (Enterprise Evidence Attribution):
1. Sentence-level Attribution with Exact Character Offsets:
   - ผูกโยงแต่ละข้อความอ้างอิง [S1], [S2] เข้ากับ offset จริง (char_start, char_end) ใน Unit.text ดั้งเดิม
   - รองรับทั้ง `Hit` (จาก Qdrant VectorStore), `Chunk` (จาก make_chunks), และ `ChildChunk` (prototype)
2. กฎการคัดกรองบริบทสังเคราะห์ (Design Decision 89 ใน PLAN.md):
   - บรรทัดบริบทพิเศษ เช่น [แถวแรกของตาราง: ...] หรือ [ป้ายก่อนหน้า: ...] จะไม่ถูกนำมาติดแท็กอ้างอิง [S...]
     โดยเด็ดขาด เพื่อป้องกันการยกคำพูดสังเคราะห์ว่าเป็นเนื้อหาจริงของเอกสาร
3. ALCE (Automatic LLMs' Citation Evaluation) Framework:
   - Citation Recall: สัดส่วนประโยคคำกล่าวอ้างที่มีหลักฐานอ้างอิงรองรับ
   - Citation Precision: สัดส่วนแท็กอ้างอิงที่ระบุตรงกับเนื้อหาจริง (ไม่หลอนแท็ก)
   - Fact / Number Verification: ตรวจสอบความถูกต้องของตัวเลขและข้อเท็จจริง (NLI Entailment)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

from app.thai.normalizer import normalize

# รูปแบบของบรรทัดบริบทสังเคราะห์ที่ห้ามนำมาอ้างอิง
CONTEXT_LINE_PREFIXES = ("แถวแรกของตาราง", "ป้ายก่อนหน้า")
_CONTEXT_LINE_RE = re.compile(r"^\[(?:" + "|".join(CONTEXT_LINE_PREFIXES) + r"):.*\]$")


def is_synthetic_context_line(line: str) -> bool:
    """ตรวจสอบว่าเป็นบรรทัดบริบทที่ระบบเติมเองหรือไม่ (เช่น [แถวแรกของตาราง: ...])"""
    cleaned = line.strip()
    return bool(_CONTEXT_LINE_RE.match(cleaned)) or any(
        cleaned.startswith(f"[{p}:") and cleaned.endswith("]") for p in CONTEXT_LINE_PREFIXES
    )


@dataclass(frozen=True)
class CitationItem:
    """โครงสร้างข้อมูลอ้างอิงระดับประโยคที่ชี้จุดในเอกสารต้นฉบับได้แม่นยำ"""

    sentence_id: str  # เช่น 'S1', 'S2'
    doc_id: str  # ชื่อไฟล์เอกสาร (file_id)
    label: str  # เช่น 'หน้า 7'
    char_start: int  # ตำแหน่งออฟเซ็ตเริ่มต้นใน Unit.text
    char_end: int  # ตำแหน่งออฟเซ็ตสิ้นสุดใน Unit.text
    snippet: str  # ข้อความประโยคอ้างอิงจริง
    position: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # คะแนน relevance score


@dataclass(frozen=True)
class CitationPack:
    """ชุดบริบทและข้อมูลอ้างอิงสำหรับส่งให้ LLM หรือ Gateway"""

    formatted_context: str  # ข้อความบริบทพร้อมแท็ก [S1], [S2] กำกับ
    citations_map: dict[str, CitationItem]  # แมป 'S1' -> CitationItem
    total_evidence_sentences: int


@dataclass
class InlineCitationResult:
    """ผลการตรวจสอบ Inline Citation ของคำตอบที่โมเดลสร้างตาม ALCE"""

    answer_text: str
    sentences: list[str]
    citations_used: list[str]  # รายการแท็ก [S1], [S2] ทั้งหมดที่ใช้
    citation_recall: float  # 0.0 - 1.0 (สัดส่วนประโยคที่มี tag อ้างอิง)
    citation_precision: float  # 0.0 - 1.0 (สัดส่วนแท็กที่มีอยู่จริงในคลัง)
    unsupported_sentences: list[str]
    grounded_citations: list[CitationItem]
    hallucinated_tags: list[str] = field(default_factory=list)


def build_citations_from_hits(
    hits: Sequence[Any],
    start_index: int = 1,
) -> dict[str, CitationItem]:
    """สร้าง Citation Map จาก Sequence ของ Hit (จาก VectorStore.search) หรือ Chunk

    แยกประโยคตาม sentence spans ดั้งเดิม และคัดกรอง context line สังเคราะห์ออก
    """
    citations: dict[str, CitationItem] = {}
    current_idx = start_index

    for hit in hits:
        # ตรวจสอบ attributes ว่าเป็น Hit, Chunk หรือ dict
        file_id = getattr(hit, "file_id", None) or getattr(hit, "document_id", "doc")
        label = getattr(hit, "label", "")
        position = dict(getattr(hit, "position", {}) or {})
        score = float(getattr(hit, "score", 0.0))
        text = getattr(hit, "text", None) or getattr(hit, "content", "")
        sentences_spans = getattr(hit, "sentences", None)
        chunk_char_start = int(getattr(hit, "char_start", 0))

        # หากมี sentence spans ชัดเจน (เช่น จาก make_chunks หรือ Hit)
        if sentences_spans and isinstance(sentences_spans, (tuple, list)):
            for span in sentences_spans:
                if hasattr(span, "char_start") and hasattr(span, "char_end"):
                    s_start, s_end = span.char_start, span.char_end
                elif isinstance(span, (tuple, list)) and len(span) == 2:
                    s_start, s_end = span[0], span[1]
                else:
                    continue

                # ตัดข้อความประโยคจาก text โดยคำนึงถึง offset ใน chunk
                rel_start = max(0, s_start - chunk_char_start)
                rel_end = min(len(text), s_end - chunk_char_start)
                snippet = text[rel_start:rel_end].strip()

                if not snippet or is_synthetic_context_line(snippet):
                    continue

                tag = f"S{current_idx}"
                citations[tag] = CitationItem(
                    sentence_id=tag,
                    doc_id=file_id,
                    label=label,
                    char_start=s_start,
                    char_end=s_end,
                    snippet=snippet,
                    position=position,
                    score=score,
                )
                current_idx += 1
        else:
            # Fallback หากไม่มี sentence spans: ตัดแบ่งตาม newline / เครื่องหมายจบ
            lines = [line.strip() for line in re.split(r"[\n\r]+", text) if line.strip()]
            offset_cursor = chunk_char_start
            for line in lines:
                if is_synthetic_context_line(line):
                    offset_cursor += len(line) + 1
                    continue

                tag = f"S{current_idx}"
                citations[tag] = CitationItem(
                    sentence_id=tag,
                    doc_id=file_id,
                    label=label,
                    char_start=offset_cursor,
                    char_end=offset_cursor + len(line),
                    snippet=line,
                    position=position,
                    score=score,
                )
                current_idx += 1
                offset_cursor += len(line) + 1

    return citations


def build_citations_from_chunks(
    chunks: Sequence[Any],
    start_index: int = 1,
) -> dict[str, CitationItem]:
    """เข้ากันได้กับ prototype เดิม (Backward Compatibility): รับ ChildChunk หรือ Chunk"""
    return build_citations_from_hits(chunks, start_index=start_index)


def format_citation_pack(
    hits: Sequence[Any],
    citations_map: Optional[dict[str, CitationItem]] = None,
    consistency_guidance: str = "",
) -> CitationPack:
    """จัดเตรียม Context ข้อความพร้อมแนบแท็ก [S1], [S2] สำหรับส่งต่อให้ LLM Context

    ตัวอย่างผลลัพธ์:
    ---
    [ข้อสังเกตความสอดคล้องของหลักฐาน (Evidence Consistency Guidance)]
    • พบตัวเลขต่างกันสำหรับ 'กำไรสุทธิ' ...

    [เอกสาร: ONEREPORTPTTT.PDF | หน้า 7]
    (บริบทโครงสร้าง: [แถวแรกของตาราง: รายได้จากการขาย 2568])
    [S1] โดยในปี 2568 บริษัทฯ มีรายได้จากการขาย 3,248,500 ล้านบาท [S2] เพิ่มขึ้นร้อยละ 5.2 จากปีก่อนหน้า
    """
    if citations_map is None:
        citations_map = build_citations_from_hits(hits)

    formatted_blocks: list[str] = []

    if consistency_guidance.strip():
        formatted_blocks.append(consistency_guidance.strip())

    for hit in hits:
        file_id = getattr(hit, "file_id", None) or getattr(hit, "document_id", "เอกสาร")
        label = getattr(hit, "label", "")
        header = f"[{file_id} | {label}]" if label else f"[{file_id}]"

        block_lines = [header]

        # แนบ context line สังเคราะห์ให้อ่านเข้าใจ แต่ไม่ติดแท็ก [S...]
        context_str = getattr(hit, "context", "")
        if context_str:
            block_lines.append(f"(บริบทโครงสร้าง: {context_str})")

        # ประกอบประโยคใน chunk พร้อมแท็ก [S...]
        text = getattr(hit, "text", None) or getattr(hit, "content", "")
        chunk_sentences: list[str] = []

        # หาประโยคที่อยู่ใน hit นี้
        matched_tags = [
            tag
            for tag, item in citations_map.items()
            if item.doc_id == file_id and item.snippet in text
        ]

        if matched_tags:
            for tag in matched_tags:
                item = citations_map[tag]
                chunk_sentences.append(f"[{tag}] {item.snippet}")
            block_lines.append(" ".join(chunk_sentences))
        else:
            block_lines.append(text)

        formatted_blocks.append("\n".join(block_lines))

    full_context = "\n\n---\n\n".join(formatted_blocks)
    return CitationPack(
        formatted_context=full_context,
        citations_map=citations_map,
        total_evidence_sentences=len(citations_map),
    )


def parse_inline_citations(text: str) -> list[str]:
    """แยกแท็กอ้างอิง เช่น [S1], [S2], [S1][S2] ออกมาจากข้อความคำตอบ"""
    matches = re.findall(r"\[(S\d+)\]", text)
    return list(dict.fromkeys(matches))


def evaluate_citation_quality(
    generated_answer: str,
    available_citations: dict[str, CitationItem],
) -> InlineCitationResult:
    """ประเมินคุณภาพการอ้างอิงตาม ALCE / LongCite Framework

    ตรวจสอบความถูกต้อง 2 มิติหลัก:
    1. Citation Recall: สัดส่วนประโยคที่ผู้ตอบยกหลักฐานมาสนับสนุน
    2. Citation Precision: สัดส่วนแท็กที่อ้างถูกต้องตรงกับคลังหลักฐานจริง (ไม่หลอนแท็ก S99)
    """
    clean_ans = normalize(generated_answer)
    raw_sentences = [s.strip() for s in re.split(r"[\n\r]+", clean_ans) if s.strip()]
    if not raw_sentences:
        raw_sentences = [clean_ans]

    all_citations_used: list[str] = []
    unsupported: list[str] = []
    grounded_items: list[CitationItem] = []
    hallucinated: list[str] = []

    for sent in raw_sentences:
        tags = parse_inline_citations(sent)
        valid_tags = [t for t in tags if t in available_citations]
        invalid_tags = [t for t in tags if t not in available_citations]

        if invalid_tags:
            hallucinated.extend(invalid_tags)

        if valid_tags:
            all_citations_used.extend(valid_tags)
            for t in valid_tags:
                grounded_items.append(available_citations[t])
        else:
            unsupported.append(sent)

    all_citations_used = list(dict.fromkeys(all_citations_used))
    hallucinated = list(dict.fromkeys(hallucinated))

    # คำนวณ Recall: สัดส่วนประโยคที่มีแท็กหลักฐานถูกต้อง
    recall = (len(raw_sentences) - len(unsupported)) / max(len(raw_sentences), 1)

    # คำนวณ Precision: แท็กที่ถูกต้อง / (แท็กที่ถูกต้อง + แท็กที่หลอน)
    total_tags_invoked = len(all_citations_used) + len(hallucinated)
    precision = len(all_citations_used) / max(total_tags_invoked, 1)

    return InlineCitationResult(
        answer_text=clean_ans,
        sentences=raw_sentences,
        citations_used=all_citations_used,
        citation_recall=recall,
        citation_precision=precision,
        unsupported_sentences=unsupported,
        grounded_citations=grounded_items,
        hallucinated_tags=hallucinated,
    )


def compute_nli_entailment_score(claim_sentence: str, evidence_sentence: str) -> float:
    """คำนวณคะแนน Entailment ระหว่างประโยคคำกล่าวอ้างกับประโยคหลักฐาน

    (Post-Generation Fact Verification):
    ตรวจสอบทั้งความครอบคลุมของคีย์เวิร์ด (Lexical overlap) และความถูกต้องของตัวเลข
    """
    from pythainlp.tokenize import word_tokenize

    claim_norm = normalize(claim_sentence)
    ev_norm = normalize(evidence_sentence)

    claim_tokens = set(word_tokenize(claim_norm, engine="newmm"))
    ev_tokens = set(word_tokenize(ev_norm, engine="newmm"))

    if not claim_tokens:
        return 0.0

    # ตรวจสอบตัวเลข (Fact Check / Numerical Contradiction Guard)
    claim_nums = set(re.findall(r"\d+(?:\.\d+)?", claim_sentence))
    ev_nums = set(re.findall(r"\d+(?:\.\d+)?", evidence_sentence))

    if claim_nums and not claim_nums.issubset(ev_nums):
        # หากคำตอบมีตัวเลขที่ไม่ปรากฏในหลักฐาน ให้ถือว่าขัดแย้ง (Contradiction/Hallucination)
        return 0.10

    # คำนวณ Token Overlap
    overlap = len(claim_tokens & ev_tokens)
    score = overlap / len(claim_tokens)

    # โบนัสเมื่อตัวเลขตรงกันครบถ้วน
    if claim_nums and claim_nums.issubset(ev_nums):
        score = min(1.0, score + 0.20)

    return min(1.0, score)
