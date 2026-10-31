"""ตัวอย่างแสดงผลลัพธ์การ Chunking จริงจาก Dataset และโครงสร้าง Parent-Child
รัน: python scripts/demo_chunking.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.chunking import (
    chunk_unit,
    format_citation_context,
    build_system_prompt,
)
from tests.test_chunking import _make_unit

EVAL_PATH = ROOT / "eval" / "datasets" / "rag_evaluation_1000.jsonl"


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def demo_from_dataset():
    if not EVAL_PATH.exists():
        print(f"File not found: {EVAL_PATH}")
        return

    # ดึง 3 ตัวอย่างที่น่าสนใจจาก dataset
    samples = []
    target_ids = {1, 3, 5}
    with EVAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("question_id") in target_ids:
                samples.append(item)
            if len(samples) == len(target_ids):
                break

    for sample in samples:
        qid = sample["question_id"]
        doc = sample.get("document_source", "doc.pdf")
        query = sample.get("query", "")
        citation_text = sample.get("expected_citation") or ""
        expected_answer = sample.get("expected_answer") or ""

        print_banner(f"ตัวอย่างข้อที่ {qid} | แหล่งที่มา: {doc}")
        print(f"คำถาม (Query): {query}")
        print(f"คำตอบที่ต้องการ: {expected_answer}")
        print("\n[ข้อความต้นฉบับในเอกสาร (Raw Text)]:")
        print(f'"""\n{citation_text}\n"""')

        unit = _make_unit(citation_text, index=0)
        result = chunk_unit(unit, document_id=doc)

        print("\n--- 1. CHILD CHUNKS (ระดับประโยค: ใช้ทำ Dense Vector + Exact Citation) ---")
        for c in result.children:
            print(f" [Child ID]: {c.chunk_id[:8]}... (Sentence #{c.sentence_index + 1})")
            print(f"   - เนื้อหา      : \"{c.content}\"")
            print(f"   - Parent FK    : {c.parent_id[:8]}...")
            print(f"   - Char Offsets : [{c.char_start} : {c.char_end}]")
            # ดึงข้อความจริงจาก original ด้วย offset มาเช็คความแม่นยำ
            slice_text = citation_text[c.char_start:c.char_end]
            print(f"   - Match Verify : \"{slice_text}\"")

        print("\n--- 2. PARENT CHUNKS (ระดับบริบท 3-5 ประโยค: ส่งให้ LLM สร้างคำตอบ) ---")
        for p in result.parents:
            print(f" [Parent ID]: {p.chunk_id[:8]}... (รวม {p.sentence_count} ประโยค)")
            print(f"   - สมาชิก Child IDs : {[cid[:8] + '...' for cid in p.child_ids]}")
            print(f"   - Char Offsets     : [{p.char_start} : {p.char_end}]")
            print(f"   - ข้อความบริบท     : \"{p.content}\"")

        print("\n--- 3. LLM CITATION CONTEXT FORMAT (รูปแบบที่ ALCE/LongCite ส่งให้โมเดล) ---")
        citation_ctx = format_citation_context(result.children)
        print(json.dumps(citation_ctx, ensure_ascii=False, indent=2))


def demo_multi_paragraph_page():
    print_banner("ตัวอย่างหน้าเอกสารจริงแบบหลายย่อหน้า (Multi-Sentence / Sliding Window)")
    text = (
        "บริษัท ซีพี ออลล์ จำกัด (มหาชน) ดำเนินธุรกิจค้าปลีกประเภทร้านสะดวกซื้อ 7-Eleven ในประเทศไทย\n"
        "ในปี 2568 บริษัทฯ มีรายได้รวมทั้งสิ้น 1,022,143 ล้านบาท เพิ่มขึ้นจากปีก่อนร้อยละ 11.2\n"
        "กำไรสุทธิสำหรับปีอยู่ที่ 28,206 ล้านบาท เติบโตอย่างแข็งแกร่งตามกำลังซื้อที่ฟื้นตัว\n"
        "บริษัทได้เปิดสาขาใหม่เพิ่มอีก 700 สาขาทั่วประเทศ ส่งผลให้มีสาขารวมทั้งสิ้น 15,945 สาขา\n"
        "นอกจากนี้ ยังมุ่งเน้นการให้บริการ All Care Pharmacy เพื่อขยายฐานสู่ธุรกิจสุขภาพครบวงจร"
    )
    unit = _make_unit(text, index=4)  # หน้า 5
    result = chunk_unit(unit, document_id="CPALL_2024.pdf", parent_window=3, parent_overlap=1)

    print(f"หน้า: {unit.label} ({len(text)} ตัวอักษร)")
    print(f"ตัดได้: {len(result.children)} Child Chunks, {len(result.parents)} Parent Chunks\n")

    print("[สรุปความสัมพันธ์ Hierarchical Mapping]:")
    for p in result.parents:
        print(f"\n📦 ParentChunk [{p.chunk_id[:8]}...]")
        print(f"   เนื้อหา: {p.content}")
        print("   ประกอบด้วยประโยค (Children):")
        for child in result.children:
            if child.parent_id == p.chunk_id or child.chunk_id in p.child_ids:
                print(f"     ├── [S{child.sentence_index + 1}] \"{child.content}\" (Offset: {child.char_start}-{child.char_end})")


if __name__ == "__main__":
    demo_from_dataset()
    demo_multi_paragraph_page()

