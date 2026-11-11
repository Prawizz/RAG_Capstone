"""[M6] Multi-Evidence Consistency & Numerical Conflict Detector for Enterprise RAG

สถาปัตยกรรมระดับ SOTA สำหรับตรวจจับข้อเท็จจริงและตัวเลขที่ขัดแย้งกันในเอกสารองค์กร:
1. Cross-Chunk Metric & Numerical Extraction:
   - ตรวจจับคู่ตัวเลขทางสถิติและการเงิน (เช่น กำไรสุทธิ, รายได้รวม, EBITDA, จำนวนสาขา) ควบคู่กับปี พ.ศ./ค.ศ.
2. Contradiction & Discrepancy Detection:
   - หาก Chunks ที่ดึงมามีตัวเลขสำหรับตัวชี้วัดเดียวกันและปีเดียวกันแต่มีค่าต่างกัน
     (เช่น หน้างบการเงินรวม vs หน้างบการเงินเฉพาะกิจการ)
   - ระบบจะตรวจจับและสร้าง ConflictNotice แจ้งเตือนอย่างเป็นทางการ
3. Context Disambiguation Guidance for LLM:
   - เสริมข้อความกำกับบริบทให้ LLM อธิบายความแตกต่างของทั้ง 2 แหล่ง แทนที่จะตอบสุ่มตัวเลขตัวใดตัวหนึ่ง
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.thai.normalizer import normalize

# คีย์เวิร์ดตัวชี้วัดทางการเงินและสถิติองค์กรที่พบบ่อย
TRACKED_METRICS = {
    "รายได้รวม": r"รายได้(?:รวม|จากการขาย)",
    "กำไรสุทธิ": r"กำไร(?:สุทธิ|สำหรับปี)",
    "ebitda": r"(?:ebitda|กำไรก่อนดอกเบี้ย)",
    "จำนวนสาขา": r"(?:สาขารวม|ขยายสาขา|จำนวนสาขา)",
    "สินทรัพย์รวม": r"สินทรัพย์รวม",
    "หนี้สินรวม": r"หนี้สินรวม",
}


@dataclass(frozen=True)
class NumericalFact:
    """ข้อเท็จจริงเชิงตัวเลขที่สกัดได้จากหลักฐานหนึ่งชิ้น"""

    doc_id: str
    label: str
    metric: str
    value: str
    year: Optional[str] = None
    raw_sentence: str = ""


@dataclass(frozen=True)
class ConflictNotice:
    """การแจ้งเตือนความขัดแย้งของข้อมูลระหว่างหลักฐาน"""

    metric: str
    year: Optional[str]
    records: tuple[dict[str, str], ...]  # รายการ [{'source': 'หน้า 5', 'value': '28,206 ล้านบาท'}, ...]
    guidance: str  # คำแนะนำสำหรับ LLM ให้ตอบโดยจำแนกที่มา


@dataclass(frozen=True)
class ConsistencyReport:
    """รายงานผลการตรวจสอบความสอดคล้องของหลักฐานทั้งหมด"""

    has_conflict: bool
    conflicts: tuple[ConflictNotice, ...]
    guidance_text: str  # ข้อความสรุปสำหรับแนบให้ LLM Context (ถ้ามีข้อขัดแย้ง)


def extract_numerical_facts(
    hits: Sequence[Any],
) -> list[NumericalFact]:
    """สกัดข้อเท็จจริงเชิงตัวเลขจาก Sequence ของ Hit หรือ Chunk"""
    facts: list[NumericalFact] = []

    for hit in hits:
        doc_id = getattr(hit, "file_id", None) or getattr(hit, "document_id", "doc")
        label = getattr(hit, "label", "")
        text = getattr(hit, "text", None) or getattr(hit, "content", "")

        # แยกเป็นประโยคย่อย
        sentences = [s.strip() for s in re.split(r"[\n\r]+", text) if s.strip()]

        for sent in sentences:
            sent_norm = normalize(sent)
            # ค้นหาปี (ถ้ามี)
            years = re.findall(r"\b(25\d{2}|20\d{2})\b", sent_norm)
            year = years[0] if years else None

            # ค้นหาตัวเลขหลักในประโยค (เช่น 1,022,143 หรือ 700 หรือ 28,206)
            numbers = re.findall(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?|\b\d+(?:\.\d+)?", sent)

            for metric_name, pattern in TRACKED_METRICS.items():
                if re.search(pattern, sent_norm, re.IGNORECASE):
                    for num in numbers:
                        # กรองปีออกจากค่าตัวเลข
                        if year and num == year:
                            continue
                        # กรองตัวเลขสั้นๆ ที่เป็นลำดับหัวข้อ เช่น 1. หรือ 2.
                        if len(num) <= 2 and sent.startswith(num):
                            continue

                        facts.append(
                            NumericalFact(
                                doc_id=doc_id,
                                label=label,
                                metric=metric_name,
                                value=num,
                                year=year,
                                raw_sentence=sent,
                            )
                        )
                        break

    return facts


def detect_evidence_conflicts(
    hits: Sequence[Any],
) -> ConsistencyReport:
    """ตรวจจับความขัดแย้งเชิงตัวเลขระหว่างหลักฐานหลายชิ้นที่ดึงมาได้"""
    facts = extract_numerical_facts(hits)
    if len(facts) < 2:
        return ConsistencyReport(has_conflict=False, conflicts=(), guidance_text="")

    # จัดกลุ่มตาม (metric, year)
    groups: dict[tuple[str, Optional[str]], list[NumericalFact]] = {}
    for f in facts:
        key = (f.metric, f.year)
        groups.setdefault(key, []).append(f)

    conflicts: list[ConflictNotice] = []

    for (metric, year), items in groups.items():
        # ตรวจสอบว่ามีค่าตัวเลขที่ต่างกันในตัวชี้วัดเดียวกันหรือไม่
        unique_values = {it.value for it in items}
        if len(unique_values) > 1:
            records = tuple(
                {
                    "source": f"{it.doc_id} ({it.label})",
                    "value": it.value,
                    "sentence": it.raw_sentence[:90] + "...",
                }
                for it in items
            )
            year_str = f" ปี {year}" if year else ""
            guidance = (
                f"พบตัวเลขต่างกันสำหรับ '{metric}'{year_str}: "
                + " เทียบกับ ".join(f"{r['source']} ระบุ {r['value']}" for r in records[:2])
                + " (แนะนำให้ตอบโดยระบุที่มาทั้งสองจุด เช่น งบรวมเทียบกับงบเฉพาะกิจการ)"
            )
            conflicts.append(
                ConflictNotice(
                    metric=metric,
                    year=year,
                    records=records,
                    guidance=guidance,
                )
            )

    has_conflict = len(conflicts) > 0
    if has_conflict:
        guidance_lines = ["[ข้อสังเกตความสอดคล้องของหลักฐาน (Evidence Consistency Guidance)]"]
        for c in conflicts:
            guidance_lines.append(f"• {c.guidance}")
        guidance_text = "\n".join(guidance_lines)
    else:
        guidance_text = ""

    return ConsistencyReport(
        has_conflict=has_conflict,
        conflicts=tuple(conflicts),
        guidance_text=guidance_text,
    )

