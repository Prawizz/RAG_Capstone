"""จำลอง Pipeline RAG ภาษาไทยแบบ End-to-End:
1. สร้างชุดข้อมูลเอกสารองค์กรจำลอง (3 หน้า/Units พร้อม Noisy Whitespace จริง)
2. รัน Hierarchical Chunking (Child + Parent)
3. ทดสอบสืบค้นด้วยคำถามจริงจาก rag_evaluation_1000.jsonl
4. จัดทำ Sentence-level Citations ([S1], [S2]) และ Context Window

รัน: python scripts/simulate_rag_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.chunking import (
    ChildChunk,
    ParentChunk,
    chunk_document,
    format_citation_context,
    build_system_prompt,
    get_parent_for_child,
)
from app.models import (
    Block,
    BlockType,
    ExtractionMethod,
    QualityReport,
    Unit,
    UnitStatus,
    UnitType,
)
from app.thai.normalizer import normalize


def make_simulated_unit(page_num: int, raw_text: str) -> Unit:
    """สร้าง Unit จำลองสำหรับหน้าเอกสาร PDF"""
    return Unit(
        index=page_num - 1,
        type=UnitType.PAGE,
        position={"page": page_num},
        label=f"หน้า {page_num}",
        status=UnitStatus.OK,
        method=ExtractionMethod.TEXT_LAYER,
        blocks=[Block(type=BlockType.PARAGRAPH, text=raw_text, order=0)],
        text=raw_text,
        text_hash=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        quality=QualityReport(),
        signals={"char_count": float(len(raw_text))},
    )


# -------------------------------------------------------------
# 1. สร้างชุดข้อมูลเอกสารองค์กรจำลอง (Enterprise Document Pages)
# -------------------------------------------------------------
def build_simulated_corpus() -> list[Unit]:
    """จำลอง 3 หน้าของรายงานประจำปี One Report (มี Noisy whitespace ปะปนเหมือน PDF จริง)"""

    # หน้าที่ 1: ข้อมูลองค์กรและผลการดำเนินงานทางการเงิน
    page_1_raw = (
        "บริษัท ซีพี ออลล์ จำกัด  (มหาชน)  รายงานประจำปี 2568\n"
        "วิสัยทัศน์ขององค์กรคือ   เราให้บริการความสะดวกกับทุกชุมชน\n"
        "ในปี 2568 บริษัทฯ  มีรายได้รวม 1,022,143 ล้านบาท  และมีกำไรสุทธิ 28,206 ล้านบาท เพิ่มขึ้นจากปีก่อนหน้าร้อยละ 11\n"
        "การเติบโตของยอดขายมาจากทั้งสาขาเดิมและการขยายสาขาใหม่อย่างต่อเนื่องในทำเลยุทธศาสตร์ทั่วประเทศ\n"
        "อัตราส่วนกำไรขั้นต้นปรับตัวดีขึ้นจากประสิทธิภาพในการบริหารจัดการห่วงโซ่อุปทานและการควบคุมต้นทุนสินค้า"
    )

    # หน้าที่ 2: การขยายสาขา เครือข่าย และบริการสุขภาพดิจิทัล
    page_2_raw = (
        "เครือข่ายร้านสะดวกซื้อและการพัฒนาบริการดิจิทัล\n"
        "โดยในปี 2568 บริษัทฯ ขยายสาขาใหม่ในไทยเพิ่มอีก 700 สาขา ทำให้สิ้นปีมีสาขารวมทั้งสิ้น 15,945 สาขา และร้อยละ 90 ของร้านเปิดเป็น Stand-Alone\n"
        "เพื่อรองรับการขายผ่านช่องทาง Online (7App) ส่งสินค้าด่วนถึงบ้านทันที และส่งตามรอบ เพื่อตอบสนองพฤติกรรมผู้บริโภคยุคใหม่\n"
        "บริษัทฯ ได้ขยายการให้บริการด้านสุขภาพผ่านการเปิดร้านจำหน่ายยาและสินค้าสุขภาพภายใต้ชื่อ All Care Pharmacy (เดิม eXta Plus)\n"
        "โดยเชื่อมโยงแอปพลิเคชัน Mor Dee (หมอดี) และ All PharmaSee เข้ากับ 7App เพื่อให้คำปรึกษาและจัดส่งยาถึงบ้าน"
    )

    # หน้าที่ 3: ความยั่งยืน ESG และนโยบายพนักงาน
    page_3_raw = (
        "เป้าหมายความยั่งยืนและการพัฒนาบุคลากร\n"
        "ซีพี ออลล์ มีเป้าหมายสร้างการเติบโตทางธุรกิจอย่างยั่งยืน ควบคู่ไปกับการพัฒนาองค์กร โดยให้ความสำคัญกับการดูแลพนักงาน ตลอดจนถึงการพัฒนาความยั่งยืนภายใต้กรอบแนวทาง ESG\n"
        "บริษัทฯ กำหนดนโยบายให้พนักงานทุกคนมีสิทธิลาพักร้อนได้ 10 วันทำการต่อปี หลังจากผ่านการทดลองงาน 120 วัน\n"
        "การยื่นใบลาพักร้อนต้องได้รับการอนุมัติจากผู้บังคับบัญชาล่วงหน้าอย่างน้อย 3 วันทำการ ผ่านระบบดิจิทัล HR Core\n"
        "บริษัทฯ ยึดมั่นในการดำเนินธุรกิจตามหลักธรรมาภิบาลและการกำกับดูแลกิจการที่ดีเพื่อผู้มีส่วนได้เสียทุกฝ่าย"
    )

    units = [
        make_simulated_unit(1, page_1_raw),
        make_simulated_unit(2, page_2_raw),
        make_simulated_unit(3, page_3_raw),
    ]
    return units


# -------------------------------------------------------------
# 2. จำลอง Simple Lexical / Hybrid Matcher สำหรับทดสอบค้นหา
# -------------------------------------------------------------
def retrieve_top_child(
    query: str,
    children: list[ChildChunk],
) -> tuple[ChildChunk, float]:
    """คำนวณคะแนนความเกี่ยวข้องระหว่าง Query และ Child Chunk (token overlap แบบ Thai aware)"""
    from pythainlp.tokenize import word_tokenize

    query_tokens = set(word_tokenize(normalize(query), engine="newmm"))
    best_score = -1.0
    best_child = children[0]

    for child in children:
        child_tokens = set(word_tokenize(child.content, engine="newmm"))
        if not child_tokens:
            continue
        # Jaccard + Exact Keyword overlap
        intersection = query_tokens.intersection(child_tokens)
        score = len(intersection) / len(query_tokens)

        # Boost พิเศษหากตัวเลขหรือคำเฉพาะตรงกัน
        query_numbers = re.findall(r"\d+", query)
        for num in query_numbers:
            if num in child.content:
                score += 0.5

        if score > best_score:
            best_score = score
            best_child = child

    return best_child, best_score


# -------------------------------------------------------------
# 3. รันการทดสอบและแสดงผล
# -------------------------------------------------------------
def main():
    print("=" * 80)
    print(" จำลองการทำงาน RAG: Thai-Aware Chunking & Sentence Citation Pipeline")
    print("=" * 80)

    # 1. โหลดเอกสารและ Chunking
    units = build_simulated_corpus()
    print(f"\n📂 เอกสารจำลอง: {len(units)} หน้า (PDF Units)")
    for u in units:
        print(f"   - {u.label}: ความยาว {len(u.text)} ตัวอักษร")

    children, parents = chunk_document(units, document_id="CPALL_ONE_REPORT.PDF")
    print(f"\n⚙️  ผลการ Chunking:")
    print(f"   - Child Chunks  (ระดับประโยคเดี่ยว) : {len(children)} chunks")
    print(f"   - Parent Chunks (ระดับหน้าต่างบริบท): {len(parents)} chunks")

    # 2. ชุดคำถามจริงจาก rag_evaluation_1000.jsonl + คำถามทดสอบ Abstention
    test_queries = [
        {
            "id": 1,
            "query": "ปี 2568 ซีพี ออลล์มีรายได้รวมเท่าไร และกำไรสุทธิเท่าไร",
            "expected_key": "1,022,143 ล้านบาท",
        },
        {
            "id": 2,
            "query": "ปี 2568 ซีพี ออลล์ขยายสาขาใหม่ในไทยกี่สาขา และมีสาขารวมทั้งสิ้นเท่าไร",
            "expected_key": "700 สาขา",
        },
        {
            "id": 3,
            "query": "บริการร้านยาใช้ชื่อว่าอะไร และเชื่อมต่อบริการสุขภาพกับแอปพลิเคชันใด",
            "expected_key": "All Care Pharmacy",
        },
        {
            "id": 4,
            "query": "พนักงานมีสิทธิลาพักร้อนได้กี่วันต่อปี และต้องผ่านการทดลองงานกี่วัน",
            "expected_key": "10 วันทำการ",
        },
        {
            "id": 5,
            "query": "อัตราดอกเบี้ยเงินกู้ซื้อบ้านและสินเชื่อส่วนบุคคลของบริษัทอยู่ที่ร้อยละเท่าไร",
            "expected_key": "ไม่มีข้อมูล (ควร Abstain)",
        },
    ]

    print("\n" + "=" * 80)
    print(" 🧪 การทดสอบค้นหากับคำถามจริง (Query Execution & Citation Verification)")
    print("=" * 80)

    for item in test_queries:
        qid = item["id"]
        q_text = item["query"]
        expected = item["expected_key"]

        print(f"\n[คำถาม #{qid}]: \"{q_text}\"")
        print(f"  👉 ข้อมูลที่ต้องการ: {expected}")

        # ค้นหา Child Chunk ที่ตรงที่สุด
        best_child, score = retrieve_top_child(q_text, children)

        # ตรวจสอบ Conformal / Retrieval Threshold (Tier 1 Gate)
        SIM_THRESHOLD = 0.25
        if score < SIM_THRESHOLD:
            print(f"  ⛔ [Tier 1 Abstention Gate]: Score = {score:.2f} < {SIM_THRESHOLD}")
            print("  🤖 คำตอบของระบบ: \"ขออภัย ระบบไม่พบข้อมูลที่เพียงพอในเอกสารองค์กรสำหรับการตอบคำถามนี้ได้อย่างถูกต้อง\"")
            continue

        # หา Parent Chunk ของประโยคนี้
        parent = get_parent_for_child(best_child, parents)

        # สร้าง Citation Context format สำหรับส่งให้ LLM
        # นำ child และเพื่อนบ้านใน parent มาจัด context
        parent_children = [c for c in children if c.chunk_id in parent.child_ids]
        citation_context = format_citation_context(parent_children, start_index=1)

        # หา tag [Sx] ของประโยคที่เป็น Evidence
        evidence_tag = "S1"
        for ctx_item in citation_context:
            if ctx_item["metadata"]["char_start"] == best_child.char_start:
                evidence_tag = ctx_item["sentence_id"]
                break

        print(f"  ✅ [ค้นพบ Evidence Sentence]: Child ID {best_child.chunk_id[:8]}... (คะแนน: {score:.2f})")
        print(f"     ตำแหน่ง: {best_child.label} | Offset: [{best_child.char_start} : {best_child.char_end}]")
        print(f"     เนื้อหาประโยค: \"{best_child.content}\"")

        print(f"\n  📦 [ดึง Parent Chunk เพื่อส่งให้ LLM ขยายบริบท]:")
        print(f"     Parent ID: {parent.chunk_id[:8]}... (ครอบคลุม {parent.sentence_count} ประโยค)")
        print(f"     บริบท: \"{parent.content}\"")

        print(f"\n  🏷️ [Citation Context ที่ส่งให้ LLM (ตัดเฉพาะข้อความสั้น)]:")
        for ctx in citation_context:
            is_evidence = " ⭐ (Evidence)" if ctx["sentence_id"] == evidence_tag else ""
            print(f"     [{ctx['sentence_id']}] {ctx['text'][:70]}...{is_evidence}")

        print(f"\n  💬 [ตัวอย่างคำตอบแบบ Inline Citation]:")
        # จำลองการสร้างคำตอบที่มีการ cite ระดับประโยค
        if qid == 1:
            ans = f"ในปี 2568 บริษัท ซีพี ออลล์ มีรายได้รวม 1,022,143 ล้านบาท และกำไรสุทธิ 28,206 ล้านบาท [{evidence_tag}]"
        elif qid == 2:
            ans = f"บริษัทฯ มีการขยายสาขาใหม่ในไทยเพิ่ม 700 สาขา ทำให้มีสาขารวมทั้งสิ้น 15,945 สาขา ณ สิ้นปี 2568 [{evidence_tag}]"
        elif qid == 3:
            ans = f"บริการร้านจำหน่ายยาและสินค้าสุขภาพดำเนินงานภายใต้ชื่อ All Care Pharmacy [{evidence_tag}] และเชื่อมโยงแอป Mor Dee เข้ากับ 7App [S3]"
        elif qid == 4:
            ans = f"พนักงานมีสิทธิลาพักร้อนได้ 10 วันทำการต่อปี โดยต้องผ่านการทดลองงาน 120 วันก่อน [{evidence_tag}]"
        else:
            ans = "..."
        print(f"     {ans}")
        print("-" * 80)


if __name__ == "__main__":
    main()

