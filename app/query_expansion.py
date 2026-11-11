"""[M5-M6] Enterprise Thai Query Reformulation & Semantic Expansion

สถาปัตยกรรมระดับ SOTA สำหรับแก้ไขปัญหา Retrieval Misses ในเอกสารองค์กร:
1. Dual-Calendar Year Normalization (พ.ศ. ↔ ค.ศ.):
   - แปลงคู่ปี พ.ศ. 2500-2600 กับ ค.ศ. 1957-2057 (เช่น 2568 ↔ 2025, 2567 ↔ 2024)
   - ป้องกันการสืบค้นงบการเงินไม่เจอเมื่อคำถามถามเป็น พ.ศ. แต่ตารางสรุปในเอกสารลงเป็น ค.ศ.
2. Enterprise Acronym & Jargon Expansion:
   - แปลงคำย่อทางการเงินและธุรกิจ เช่น EBITDA, O2O, SMEs, ESG, Contact Center
   - ขยายเป็นคำศัพท์ทางการที่มักปรากฏในรายงานประจำปี (One Report)
3. Salience Concept Decomposition:
   - จัดกลุ่ม Query เป็นแนวคิดหลัก (Query Concepts) พร้อมค่าน้ำหนักเฉพาะตัว
   - กำหนดให้คำพ้อง (Synonyms) และปีคู่ขนานเป็นทางเลือก (OR) ที่ตอบสนองโจทย์แนวคิดเดียวกันได้ 100%
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from app.thai.normalizer import normalize

# ตารางจับคู่คำย่อและศัพท์องค์กรที่พบบ่อยในเอกสารธุรกิจและงบการเงิน
ENTERPRISE_ACRONYMS: dict[str, tuple[str, ...]] = {
    "ebitda": ("กำไรก่อนดอกเบี้ย ภาษี ค่าเสื่อมราคา", "กำไรจากการดำเนินงานก่อนดอกเบี้ย"),
    "o2o": ("online to offline", "7delivery", "all online", "หน้าร้านและออนไลน์"),
    "sme": ("วิสาหกิจขนาดกลางและขนาดย่อม", "ผู้ประกอบการขนาดกลางและขนาดย่อม"),
    "smes": ("วิสาหกิจขนาดกลางและขนาดย่อม", "ผู้ประกอบการขนาดกลางและขนาดย่อม"),
    "esg": ("สิ่งแวดล้อม สังคม บรรษัทภิบาล", "การพัฒนาอย่างยั่งยืน"),
    "cogs": ("ต้นทุนขาย", "ต้นทุนการขายและการให้บริการ"),
    "contact center": ("ศูนย์บริการข้อมูล", "1365", "คอลเซ็นเตอร์"),
    "call center": ("ศูนย์บริการข้อมูล", "คอลเซ็นเตอร์"),
    "net zero": ("การปล่อยก๊าซเรือนกระจกสุทธิเป็นศูนย์", "ลดการปล่อยคาร์บอน"),
    "7-11": ("เซเว่น อีเลฟเว่น", "ร้านเซเว่น", "7-eleven"),
    "7-eleven": ("เซเว่น อีเลฟเว่น", "ร้านเซเว่น", "7-11"),
    "เซเว่น": ("เซเว่น อีเลฟเว่น", "ร้านเซเว่น", "7app"),
    "รายได้รวม": ("รายได้จากการขายและการให้บริการ", "รายได้จากการดำเนินงาน"),
    "กำไรสุทธิ": ("กำไรสำหรับปี", "กำไรสุทธิส่วนที่เป็นของบริษัทใหญ่"),
}

# คำสามัญประจำองค์กร (Corporate Boilerplate) ที่มีเกลื่อนทุกหน้าและมักกลบคำค้นหลัก
CORPORATE_BOILERPLATE_WORDS: set[str] = {
    "บริษัท", "บริษัทฯ", "จำกัด", "มหาชน", "กลุ่ม", "องค์กร", "เครือ", "ธุรกิจ",
    "ซีพี", "ออลล์", "ซีพีออลล์", "ปตท", "ปตท.", "เคแบงก์", "กสิกรไทย",
    "ปี", "พ.ศ.", "ค.ศ.", "ประจำปี", "แห่ง", "นี้", "มี", "ของ", "ใน", "ที่",
}


@dataclass(frozen=True)
class QueryConcept:
    """แนวคิดหนึ่งคำในคำถาม พร้อมค่าน้ำหนักความสำคัญและคำพ้องความหมาย"""

    term: str
    weight: float
    synonyms: tuple[str, ...] = ()
    is_salient: bool = True


@dataclass(frozen=True)
class ExpandedQuery:
    """ผลการวิเคราะห์และขยายคำถามเชิงอรรถศาสตร์"""

    original_query: str
    expanded_text: str  # ข้อความคำถามที่ขยายคำพ้องและปีคู่ขนานสำหรับทำ Dense/BM25
    canonical_terms: tuple[str, ...]  # รายการคำทั้งหมดหลังขยาย
    salient_terms: tuple[str, ...]  # คำสำคัญแก่นเรื่อง (Core Intent)
    boilerplate_terms: tuple[str, ...]  # คำบริบททั่วไปขององค์กร
    year_mappings: dict[str, str] = field(default_factory=dict)  # คู่ปี เช่น {'2568': '2025'}
    concepts: tuple[QueryConcept, ...] = ()  # แนวคิดระดับคำสำหรับ Salience Weighted Scoring


def convert_buddhist_to_christian_year(year_str: str) -> Optional[str]:
    """แปลง พ.ศ. เป็น ค.ศ. หรือ ค.ศ. เป็น พ.ศ. (ปี 2500-2600 ↔ 1957-2057)"""
    try:
        val = int(year_str)
        if 2500 <= val <= 2650:
            return str(val - 543)
        elif 1957 <= val <= 2107:
            return str(val + 543)
    except ValueError:
        pass
    return None


def expand_query(query: str) -> ExpandedQuery:
    """ขยายคำถามด้วยปีคู่ขนาน คำย่อธุรกิจ และแยกคำแก่นเรื่องออกจากคำสามัญองค์กร"""
    clean_q = normalize(query).strip()
    if not clean_q:
        return ExpandedQuery(
            original_query="",
            expanded_text="",
            canonical_terms=(),
            salient_terms=(),
            boilerplate_terms=(),
            year_mappings={},
            concepts=(),
        )

    from pythainlp.tokenize import word_tokenize

    tokens = word_tokenize(clean_q, engine="newmm")
    clean_tokens = [t.strip() for t in tokens if t.strip()]

    year_mappings: dict[str, str] = {}
    expansion_addons: list[str] = []

    # 1. ค้นหาปี พ.ศ. / ค.ศ. ในคำถาม และสร้างคู่ปีคู่ขนาน
    years_found = re.findall(r"\b(25\d{2}|20\d{2})\b", clean_q)
    for y in years_found:
        alt_y = convert_buddhist_to_christian_year(y)
        if alt_y:
            year_mappings[y] = alt_y
            expansion_addons.append(alt_y)

    # 2. ขยายคำย่อทางธุรกิจ (Acronym Expansion)
    lower_q = clean_q.lower()
    matched_acronym_synonyms: dict[str, tuple[str, ...]] = {}
    for acronym, expansions in ENTERPRISE_ACRONYMS.items():
        if re.search(r"\b" + re.escape(acronym) + r"\b", lower_q) or acronym in lower_q:
            expansion_addons.extend(expansions)
            matched_acronym_synonyms[acronym] = expansions

    # 3. ประกอบข้อความขยายผลสำหรับส่งเข้า Embedder / Sparse BM25
    if expansion_addons:
        expanded_text = f"{clean_q} {' '.join(expansion_addons)}"
    else:
        expanded_text = clean_q

    # 4. แยก Salient Terms (Core Intent) ออกจาก Boilerplate และสร้าง QueryConcept
    salient_list: list[str] = []
    boilerplate_list: list[str] = []
    concepts_list: list[QueryConcept] = []

    seen_tokens: set[str] = set()

    for token in clean_tokens:
        if token in seen_tokens:
            continue
        seen_tokens.add(token)

        # กรองเครื่องหมายวรรคตอนเดี่ยว
        if re.match(r"^[\s\-_.,/\\():;!?\"']+$", token):
            continue

        token_lower = token.lower()

        # ตรวจสอบว่าเป็นปีหรือไม่
        if token in year_mappings:
            alt_year = year_mappings[token]
            salient_list.append(token)
            salient_list.append(alt_year)
            concepts_list.append(
                QueryConcept(term=token, weight=3.5, synonyms=(alt_year,), is_salient=True)
            )
            continue

        # ตรวจสอบว่าเป็นคำย่อหรือไม่
        if token_lower in matched_acronym_synonyms:
            syns = matched_acronym_synonyms[token_lower]
            salient_list.append(token)
            salient_list.extend(syns)
            concepts_list.append(
                QueryConcept(term=token, weight=3.0, synonyms=syns, is_salient=True)
            )
            continue

        # ตรวจสอบว่าเป็นตัวเลขทั่วไปหรือไม่
        if re.match(r"^\d+(?:,\d{3})*(?:\.\d+)?$", token):
            salient_list.append(token)
            concepts_list.append(
                QueryConcept(term=token, weight=3.0, is_salient=True)
            )
            continue

        # ตรวจสอบว่าเป็นคำสามัญประจำองค์กร (Boilerplate) หรือไม่
        if token in CORPORATE_BOILERPLATE_WORDS:
            boilerplate_list.append(token)
            concepts_list.append(
                QueryConcept(term=token, weight=0.4, is_salient=False)
            )
        else:
            salient_list.append(token)
            concepts_list.append(
                QueryConcept(term=token, weight=2.5, is_salient=True)
            )

    unique_salient = tuple(dict.fromkeys(salient_list))
    unique_boilerplate = tuple(dict.fromkeys(boilerplate_list))
    all_terms = tuple(dict.fromkeys(clean_tokens + [t for add in expansion_addons for t in word_tokenize(add)]))

    return ExpandedQuery(
        original_query=clean_q,
        expanded_text=expanded_text,
        canonical_terms=all_terms,
        salient_terms=unique_salient,
        boilerplate_terms=unique_boilerplate,
        year_mappings=year_mappings,
        concepts=tuple(concepts_list),
    )
