"""[M6] Sentence-level Citation Builder & ALCE / LongCite Verification

สถาปัตยกรรม:
- การสร้าง Citation Objects จาก Chunks ที่ผ่านการดึงข้อมูลเท่านั้น (ห้ามสร้างเองเด็ดขาด)
- ALCE (Automatic LLMs' Citation Evaluation) Framework:
  • Citation Recall: ประโยคคำตอบมีแท็กอ้างอิงสนับสนุนครบถ้วนหรือไม่
  • Citation Precision: แหล่งอ้างอิงที่ระบุตรงกับเนื้อหาจริงและไม่มีการอ้างส่วนเกิน
- Post-Generation NLI Verification:
  คำนวณ Entailment ระหว่าง (ประโยคคำตอบ s_i, ประโยคหลักฐาน S_j)
  หาก Entailment < 0.80 จะตัดแท็กออก หรือส่งสัญญาณให้ Abstention Gate ปฏิเสธการตอบ
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from app.chunking import ChildChunk
from app.thai.normalizer import normalize


@dataclass
class CitationItem:
    """โครงสร้างข้อมูลอ้างอิงระดับประโยค"""
    sentence_id: str          # เช่น 'S1', 'S2'
    doc_id: str               # ชื่อไฟล์เอกสาร
    label: str                # เช่น 'หน้า 7'
    char_start: int           # ตำแหน่งออฟเซ็ตเริ่มต้น
    char_end: int             # ตำแหน่งออฟเซ็ตสิ้นสุด
    snippet: str              # ข้อความประโยคอ้างอิงจริง
    position: dict = field(default_factory=dict)


@dataclass
class InlineCitationResult:
    """ผลการตรวจสอบ Inline Citation ของคำตอบที่โมเดลสร้าง"""
    answer_text: str
    sentences: list[str]
    citations_used: list[str]       # รายการแท็ก [S1], [S2] ทั้งหมดที่ใช้
    citation_recall: float         # 0.0 - 1.0 (สัดส่วนประโยคที่มี tag อ้างอิง)
    unsupported_sentences: list[str]
    grounded_citations: list[CitationItem]


def build_citations_from_chunks(
    chunks: Sequence[ChildChunk],
    start_index: int = 1,
) -> dict[str, CitationItem]:
    """สร้าง Citation Map (เช่น {'S1': CitationItem, 'S2': CitationItem}) จาก Chunks ที่ดึงมาได้เท่านั้น"""
    citations_map: dict[str, CitationItem] = {}
    for i, chunk in enumerate(chunks):
        tag = f"S{start_index + i}"
        citations_map[tag] = CitationItem(
            sentence_id=tag,
            doc_id=chunk.document_id,
            label=chunk.label,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            snippet=chunk.content,
            position=dict(chunk.position),
        )
    return citations_map


def parse_inline_citations(text: str) -> list[str]:
    """แยกแท็กอ้างอิง เช่น [S1], [S2], [S1][S2] ออกมาจากข้อความ"""
    matches = re.findall(r"\[(S\d+)\]", text)
    return list(dict.fromkeys(matches))  # deduplicate แต่คงลำดับ


def evaluate_citation_quality(
    generated_answer: str,
    available_citations: dict[str, CitationItem],
) -> InlineCitationResult:
    """ประเมินคุณภาพการอ้างอิงตาม ALCE / LongCite Framework

    ตรวจสอบว่า:
    1. ทุกประโยคมีแท็กอ้างอิงสนับสนุนหรือไม่ (Citation Recall)
    2. แท็กที่ระบุมีอยู่ในชุดที่ค้นพบจริงหรือไม่ (ป้องกันการหลอนแท็กอ้างอิง)
    """
    clean_ans = normalize(generated_answer)
    # แยกประโยคคำตอบด้วย newline หรือจุด
    raw_sentences = [s.strip() for s in re.split(r"[\n\r]+", clean_ans) if s.strip()]
    if not raw_sentences:
        raw_sentences = [clean_ans]

    all_citations_used: list[str] = []
    unsupported: list[str] = []
    grounded_items: list[CitationItem] = []

    for sent in raw_sentences:
        tags = parse_inline_citations(sent)
        valid_tags_in_sent = [t for t in tags if t in available_citations]

        if valid_tags_in_sent:
            all_citations_used.extend(valid_tags_in_sent)
            for t in valid_tags_in_sent:
                grounded_items.append(available_citations[t])
        else:
            unsupported.append(sent)

    all_citations_used = list(dict.fromkeys(all_citations_used))
    # คำนวณ Citation Recall = จำนวนประโยคที่มี tag ถูกต้อง / จำนวนประโยคทั้งหมด
    recall = (len(raw_sentences) - len(unsupported)) / max(len(raw_sentences), 1)

    return InlineCitationResult(
        answer_text=clean_ans,
        sentences=raw_sentences,
        citations_used=all_citations_used,
        citation_recall=recall,
        unsupported_sentences=unsupported,
        grounded_citations=grounded_items,
    )


def compute_nli_entailment_score(claim_sentence: str, evidence_sentence: str) -> float:
    """คำนวณคะแนน Entailment ระหว่างประโยคคำกล่าวอ้างกับประโยคหลักฐาน

    (Post-Generation NLI Verification):
    หาก Token ของหลักฐานครอบคลุมข้อเท็จจริงหลัก คะแนนจะสูง
    """
    from pythainlp.tokenize import word_tokenize

    claim_tokens = set(word_tokenize(normalize(claim_sentence), engine="newmm"))
    ev_tokens = set(word_tokenize(normalize(evidence_sentence), engine="newmm"))

    if not claim_tokens:
        return 0.0

    # ตรวจสอบตัวเลข (Fact Check)
    claim_nums = set(re.findall(r"\d+", claim_sentence))
    ev_nums = set(re.findall(r"\d+", evidence_sentence))
    # ถ้ามีตัวเลขในคำตอบแต่ไม่ตรงกับหลักฐาน ให้ถือว่าเป็นความขัดแย้ง (Contradiction)
    if claim_nums and not claim_nums.issubset(ev_nums):
        return 0.10

    # Jaccard / Overlap score
    overlap = len(claim_tokens & ev_tokens)
    score = overlap / len(claim_tokens)

    return min(1.0, score + (0.2 if claim_nums and claim_nums.issubset(ev_nums) else 0.0))
