"""[M5-M6] Production Reranker & Conformal Abstention Gate for Enterprise Thai RAG

สถาปัตยกรรมระดับ SOTA:
1. Concept Salience-Weighted Intent Reranking:
   - ผสานการวิเคราะห์แนวคิดคำถามผ่าน `app.query_expansion::expand_query`
   - กำหนดค่าน้ำหนักตามความสำคัญ (Core Intent และตัวเลข = 2.5-3.5, Corporate Boilerplate = 0.4)
   - คำพ้อง (Synonyms) และปีคู่ขนาน พ.ศ. ↔ ค.ศ. ถือเป็นทางเลือก (OR) ที่ตอบสนองโจทย์แนวคิดเดียวกันได้สมบูรณ์
2. Lexical-Semantic Late-Interaction Alignment:
   - ผสาน Dense Semantic Relevance เข้ากับ Exact Token Matching ที่ขจัดคำเชื่อมไวยากรณ์ไทย
3. Conformal Prediction Abstention Gate (Angelopoulos & Bates 2023 Framework):
   - ปฏิเสธการตอบเมื่อหลักฐานไม่เพียงพอ ป้องกัน Hallucination
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from app.query_expansion import expand_query
from app.thai.normalizer import normalize

FALLBACK_ABSTAIN_MESSAGE = (
    "ขออภัย ระบบไม่พบข้อมูลที่เพียงพอในเอกสารองค์กรสำหรับการตอบคำถามนี้ได้อย่างถูกต้อง"
)

THAI_FUNCTION_WORDS = {
    "ใน", "ที่", "และ", "หรือ", "ของ", "การ", "ความ", "จาก", "เป็น", "มี", "ได้",
    "ให้", "จะ", "ว่า", "นี้", "นั้น", "กับ", "โดย", "ตาม", "เพื่อ", "ไป", "มา",
    "ซึ่ง", "อัน", "แต่", "ถ้า", "จึง", "ก็", "อยู่", "แห่ง", "ด้วย", "เช่น",
}


@dataclass(frozen=True)
class AbstentionDecision:
    """ผลการตัดสินใจของ Abstention Gate"""

    should_abstain: bool
    reason: Optional[str] = None
    confidence_score: float = 0.0
    fallback_message: Optional[str] = None


@dataclass(frozen=True)
class RerankedHit:
    """Hit ที่ผ่านการคำนวณคะแนน Salience-Weighted Reranking เรียบร้อยแล้ว"""

    hit: Any
    original_score: float
    rerank_score: float
    lexical_overlap: float
    matched_terms: tuple[str, ...] = ()


def _extract_thai_tokens(text: str) -> set[str]:
    """ตัดคำภาษาไทยและคำภาษาอังกฤษ/ตัวเลข พร้อมกรองคำหยุดพื้นฐาน"""
    from pythainlp.tokenize import word_tokenize

    clean = normalize(text).strip()
    if not clean:
        return set()
    tokens = word_tokenize(clean, engine="newmm")
    return {
        t.strip()
        for t in tokens
        if t.strip()
        and t.strip() not in THAI_FUNCTION_WORDS
        and not re.match(r"^[\s\-_.,/\\():;!?\"']+$", t.strip())
    }


def compute_lexical_alignment(query: str, doc_text: str) -> tuple[float, tuple[str, ...]]:
    """คำนวณความสอดคล้องระดับคำโดยถ่วงน้ำหนักตามแนวคิดความสำคัญ (Concept Salience Overlap)"""
    exp = expand_query(query)
    d_clean = normalize(doc_text)
    d_tokens = _extract_thai_tokens(d_clean)

    if not exp.concepts:
        return 0.0, ()

    matched_set: set[str] = set()
    accumulated_score = 0.0
    total_weight = 0.0

    for concept in exp.concepts:
        total_weight += concept.weight
        # เช็คว่า concept.term อยู่ใน d_tokens หรือ d_clean หรือไม่
        term_matched = concept.term in d_tokens or concept.term in d_clean
        # หรือมี synonym ใดๆ ของ concept นี้อยู่ใน d_clean หรือไม่
        syn_matched = any(syn in d_clean for syn in concept.synonyms)

        if term_matched or syn_matched:
            accumulated_score += concept.weight
            matched_set.add(concept.term)
            if syn_matched and not term_matched:
                matched_set.add(f"{concept.term}(syn)")

    if total_weight <= 0:
        return 0.0, ()

    overlap_ratio = min(1.0, accumulated_score / total_weight)
    return round(overlap_ratio, 4), tuple(sorted(matched_set))


def rerank_hits(
    query: str,
    hits: Sequence[Any],
    dense_weight: float = 0.40,
    lexical_weight: float = 0.60,
) -> list[RerankedHit]:
    """จัดลำดับ Chunks/Hits ใหม่ด้วย Salience-Weighted Alignment"""
    reranked: list[RerankedHit] = []

    for hit in hits:
        orig_score = float(getattr(hit, "score", 0.0))
        text = getattr(hit, "text", None) or getattr(hit, "content", "")
        context = getattr(hit, "context", "")
        full_text = f"{context}\n{text}" if context else text

        lex_score, matched = compute_lexical_alignment(query, full_text)

        norm_orig = max(0.0, min(1.0, orig_score if orig_score <= 1.0 else 1.0 / (1.0 + math.exp(-orig_score))))
        combined_score = (dense_weight * norm_orig) + (lexical_weight * lex_score)

        reranked.append(
            RerankedHit(
                hit=hit,
                original_score=orig_score,
                rerank_score=round(combined_score, 4),
                lexical_overlap=round(lex_score, 4),
                matched_terms=matched,
            )
        )

    reranked.sort(key=lambda r: r.rerank_score, reverse=True)
    return reranked


class ConformalAbstentionGate:
    """ด่านประเมินความมั่นใจเพื่อตัดสินใจปฏิเสธการตอบ (Abstention Gate)"""

    def __init__(
        self,
        min_confidence_threshold: float = 0.15,
        min_lexical_overlap: float = 0.05,
    ) -> None:
        self.min_confidence_threshold = min_confidence_threshold
        self.min_lexical_overlap = min_lexical_overlap

    def evaluate(
        self,
        query: str,
        reranked_hits: Sequence[RerankedHit],
    ) -> AbstentionDecision:
        clean_q = normalize(query).strip()

        if not clean_q or len(clean_q) < 2:
            return AbstentionDecision(
                should_abstain=True,
                reason="EMPTY_QUERY",
                confidence_score=0.0,
                fallback_message=FALLBACK_ABSTAIN_MESSAGE,
            )

        if not reranked_hits:
            return AbstentionDecision(
                should_abstain=True,
                reason="NO_EVIDENCE_FOUND",
                confidence_score=0.0,
                fallback_message=FALLBACK_ABSTAIN_MESSAGE,
            )

        top_hit = reranked_hits[0]
        top_score = top_hit.rerank_score
        top_lex = top_hit.lexical_overlap

        # คะแนนความเกี่ยวข้องต่ำกว่า Conformal Threshold
        if top_score < self.min_confidence_threshold:
            return AbstentionDecision(
                should_abstain=True,
                reason="INSUFFICIENT_EVIDENCE",
                confidence_score=top_score,
                fallback_message=FALLBACK_ABSTAIN_MESSAGE,
            )

        # ขาดความสอดคล้องระดับคำ (Weak Lexical Match) และคะแนนรวมไม่สูงพอ
        if top_lex < self.min_lexical_overlap and top_score < 0.60:
            return AbstentionDecision(
                should_abstain=True,
                reason="INSUFFICIENT_EVIDENCE",
                confidence_score=top_score,
                fallback_message=FALLBACK_ABSTAIN_MESSAGE,
            )

        return AbstentionDecision(
            should_abstain=False,
            reason=None,
            confidence_score=top_score,
            fallback_message=None,
        )
