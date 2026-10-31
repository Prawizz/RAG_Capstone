"""สคริปต์รัน Pipeline RAG ระดับโปรดักชันที่เป็นอิสระจากโดเมน (Enterprise Domain-Agnostic RAG Pipeline)

สถาปัตยกรรมระดับ Production:
- รองรับเอกสารองค์กรทุกประเภท (PDF, DOCX, ฯลฯ) โดยไม่มีการ Hardcode คำศัพท์หรือชื่อบริษัทใดๆ
- Anthropic Contextual Retrieval 2024: พ่วงชื่อเอกสารและหัวข้อเข้ากับ Chunk สำหรับ Indexing โดยไม่ทำลายข้อความดั้งเดิม
- BM25+ (Lv & Zhai, 2011) สำหรับภาษาไทยและคำผสมหลากภาษา
- BGE-M3 Dense Semantic Embedding (1024 มิติ)
- Reciprocal Rank Fusion (RRF k=60)
- ColBERT MaxSim Token-to-Token Interaction Reranker
- Multi-Tiered Abstention Gate (Conformal Prediction)

การใช้งาน:
  python scripts/run_full_pipeline.py --query "คำถามของคุณ"
  python scripts/run_full_pipeline.py --file "path/to/any_doc.pdf" --query "คำถาม"
  python scripts/run_full_pipeline.py --pages 1-20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.ingest import IngestionService

DEFAULT_PDF = ROOT / "eval" / "datasets" / "E_ONE_REPORT_CPALLT.PDF"
DEFAULT_PAGES = [5, 6, 7, 8, 9, 10, 11, 12, 143, 144, 145, 146]


def print_header(title: str):
    print("\n" + "=" * 85)
    print(f"  {title}")
    print("=" * 85)


def print_step(step_num: int, title: str):
    print(f"\n[ขั้นตอนที่ {step_num}] {title}")
    print("-" * 60)


def run_pipeline(
    file_path: str | Path | None = None,
    custom_query: str | None = None,
    pages_arg: str | None = None,
):
    print_header("🎯 Enterprise Domain-Agnostic RAG: Production Complete Pipeline")

    doc_path = Path(file_path) if file_path else DEFAULT_PDF
    if not doc_path.exists():
        print(f"[ERROR] ไม่พบไฟล์ {doc_path}")
        return

    # กำหนดช่วงหน้า
    target_pages = DEFAULT_PAGES if doc_path == DEFAULT_PDF else None
    if pages_arg:
        if pages_arg.lower() == "all":
            target_pages = None
        elif "-" in pages_arg:
            parts = pages_arg.split("-")
            target_pages = (int(parts[0]), int(parts[1]))
        elif "," in pages_arg:
            target_pages = [int(p.strip()) for p in pages_arg.split(",")]

    service = IngestionService()

    desc_pages = (
        f"{len(target_pages)} หน้า {target_pages}"
        if isinstance(target_pages, list)
        else (f"หน้า {target_pages[0]}-{target_pages[1]}" if isinstance(target_pages, tuple) else "ทุกหน้าในไฟล์")
    )
    print_step(1, f"นำเข้าเอกสาร: {doc_path.name} ({desc_pages})")

    t_start = time.time()
    report = service.ingest_file(doc_path, pages=target_pages)
    total_ingest_time = time.time() - t_start

    print(f" ✓ สถานะเอกสาร (Document Status) : {report.document_status.value}")
    print(f" ✓ หน้าที่ประมวลผล (Units OK)     : {report.ok_units} หน้า")
    print(f" ✓ ผลการตัดประโยค (Child Chunks)   : {report.total_children} chunks (Evidence Sentence)")
    print(f" ✓ ผลรวมหน้าต่าง (Parent Chunks)   : {report.total_parents} chunks (Context Window)")
    print(f" ⏱️ สรุปเวลา Ingestion:")
    print(f"    - Text Extraction : {report.extraction_ms:.1f} ms")
    print(f"    - Thai SBD Chunk  : {report.chunking_ms:.1f} ms")
    print(f"    - Hybrid Indexing : {report.indexing_ms:.1f} ms")
    print(f"    - รวมทั้งสิ้น      : {total_ingest_time:.2f} วินาที")

    # กำหนดคำถามทดสอบ
    if custom_query:
        test_queries = [custom_query]
    else:
        test_queries = [
            "ในปี 2568 รายได้รวมเท่าไหร่ CP all",
            "ในปี 2568 บริษัทขยายสาขาใหม่ในไทยกี่สาขา และมีสาขารวมเท่าไร",
            "วิสัยทัศน์ขององค์กร คืออะไร",
            "อัตราดอกเบี้ยและสินเชื่อกู้ซื้อคอนโดมิเนียมของธนาคารพาณิชย์",  # ทดสอบกรณีอยู่นอกขอบเขตเอกสาร
        ]

    print_step(2, f"สืบค้นด้วย Contextual BM25+ + BGE-M3 Dense + RRF + ColBERT MaxSim ({len(test_queries)} คำถาม)")

    for i, q in enumerate(test_queries, 1):
        print("\n" + "=" * 85)
        print(f" ❓ คำถามที่ {i}: \"{q}\"")
        print("=" * 85)

        t0 = time.time()
        res = service.query(q, top_k=3, fusion_method="rrf")
        query_elapsed = (time.time() - t0) * 1000

        # กรณีตกเกณฑ์ Abstention Gate
        if res["status"] == "abstained":
            print(f" ⛔ [ABSTENTION GATE ACTIVATED]: ไม่ผ่านเกณฑ์ {res['tier_failed']}")
            print(f"    - คะแนนความมั่นใจ : {res['confidence_score']:.4f}")
            print(f"    - เกณฑ์ขั้นต่ำ     : {res['threshold']:.4f}")
            print(f" 🤖 คำตอบมาตรฐาน  : \"{res['fallback_message']}\"")
            print(f" ⏱️ Latency         : {query_elapsed:.2f} ms")
            continue

        # กรณีค้นหาสำเร็จ
        payload = res["payload"]
        print(f" ✅ [สืบค้นสำเร็จ]: พบข้อมูลเกี่ยวข้อง {payload['retrieved_count']} หลักฐาน (Latency: {query_elapsed:.2f} ms)")
        print(f"    - คะแนน Reranker สูงสุด : {res['confidence_score']:.4f}")

        # แสดง Top-1 Evidence Sentence
        top_evidence = payload["evidence_chunks"][0]
        print(f"\n 📌 [หลักฐานประโยค Evidence Sentence ระดับ Child]:")
        print(f"    - ตำแหน่งในไฟล์ : {top_evidence['label']} (Offset {top_evidence['offsets'][0]}-{top_evidence['offsets'][1]})")
        print(f"    - คะแนน RRF     : {top_evidence['rrf_score']:.4f}")
        print(f"    - คะแนน Rerank  : {top_evidence.get('reranker_score', 0.0):.4f}")
        print(f"    - คะแนน BM25+   : {top_evidence['bm25_score']:.2f} | Dense: {top_evidence['dense_score']:.3f}")

        # แสดง Parent Context ที่ส่งให้ LLM
        print(f"\n 📦 [บริบทระดับย่อหน้า Parent Context สำหรับ LLM]:")
        for idx, p_ctx in enumerate(payload["parent_contexts"][:2], 1):
            print(f"    ({idx}) \"{p_ctx[:180]}...\"")

        # แสดง Citation Context
        print(f"\n 🏷️ [ALCE Citation Tags ที่ส่งให้ Gateway]:")
        for c in payload["citation_context"][:3]:
            print(f"    [{c['sentence_id']}] \"{c['text'][:65]}...\" (Offset: {c['metadata']['char_start']}-{c['metadata']['char_end']})")

    print_header("✨ สรุปผล: ระบบรันแบบ Production-Grade Domain-Agnostic สำเร็จเรียบร้อย")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Production Enterprise RAG Pipeline")
    parser.add_argument("--file", "-f", type=str, default=None, help="Document file path (PDF, DOCX, etc.)")
    parser.add_argument("--query", "-q", type=str, default=None, help="Custom query to test")
    parser.add_argument("--pages", "-p", type=str, default=None, help="Pages to ingest (e.g. 1-20 or all)")
    args = parser.parse_args()

    run_pipeline(file_path=args.file, custom_query=args.query, pages_arg=args.pages)
