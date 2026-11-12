"""สคริปต์รัน Pipeline RAG ระดับโปรดักชันที่เป็นอิสระจากโดเมน (Enterprise Domain-Agnostic RAG Pipeline)

สถาปัตยกรรมระดับ Production:
- รองรับเอกสารองค์กรทุกประเภท (PDF, DOCX, PPTX, XLSX) ผ่าน Contract เดียวกัน
- Ingestion & Quality Gate: สกัดข้อความผ่าน normalizer กลาง และคัดแยกสถานะ unit (OK, NEEDS_OCR, ฯลฯ)
- Thai SBD Chunking (app/chunking.py::make_chunks): ตัดตามขอบประโยคเป้าหมาย พร้อม SentenceSpan offsets
- Two-Gate Enterprise ACL (app/access.py): ตรวจสอบสิทธิ์ 2 ด่าน (Pre-filter และ Post-retrieval Re-check)
- Hybrid Vector Store (app/vectorstore.py): Qdrant Store พร้อม named vectors (dense + sparse_bge + sparse_bm25)
- Lexical-Semantic Late-Interaction Reranking (app/rerank.py): จัดลำดับ Chunks ด้วย Exact Match
- Conformal Prediction Abstention Gate (app/rerank.py): ปฏิเสธการตอบเมื่อหลักฐานไม่เพียงพอ
- Sentence-level Citations & ALCE Verification (app/citations.py): สร้างแท็ก [S1], [S2] โดยกรองบริบทสังเคราะห์ออก

การใช้งาน:
  python scripts/run_full_pipeline.py --query "คำถามของคุณ"
  python scripts/run_full_pipeline.py --file "path/to/any_doc.pdf" --pages 1-20
  python scripts/run_full_pipeline.py --server  # เชื่อมต่อ Qdrant จริงจาก Docker
  python scripts/run_full_pipeline.py --hash-embedding  # ใช้ Hash Vector เมื่อเครื่องไม่มี PyTorch
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.access import AclStore, Box, FileAcl, Visibility
from app.citations import build_citations_from_hits, format_citation_pack
from app.chunking import make_chunks
from app.config import Settings
from app.embedding import EmbeddingUnavailable, get_embedder
from app.extraction.pipeline import extract_file
from app.models import Document, UnitStatus
from app.rerank import ConformalAbstentionGate, FALLBACK_ABSTAIN_MESSAGE, rerank_hits
from app.vectorstore import IndexedChunk, Query, VectorStore, embed_chunks, embed_query

DEFAULT_PDF = ROOT / "eval" / "datasets" / "E_ONE_REPORT_CPALLT.PDF"
DEFAULT_PAGES = (6, 8)
BOX_ID = "box_enterprise_demo"


def parse_pages(spec: str) -> tuple[int, int] | None:
    if not spec or spec.lower() == "all":
        return None
    if "-" in spec:
        p1, p2 = spec.split("-", 1)
        return (int(p1), int(p2))
    if "," in spec:
        nums = [int(p.strip()) for p in spec.split(",")]
        return (min(nums), max(nums))
    p = int(spec.strip())
    return (p, p)


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
    use_server: bool = False,
    use_hash_embedding: bool = False,
):
    print_header("🎯 Enterprise Domain-Agnostic RAG: Production Complete Pipeline")

    doc_path = Path(file_path) if file_path else DEFAULT_PDF
    if not doc_path.exists():
        print(f"[ERROR] ไม่พบไฟล์ {doc_path}")
        return 1

    target_pages = parse_pages(pages_arg) if pages_arg else (DEFAULT_PAGES if doc_path == DEFAULT_PDF else None)

    # 1. โหลด Embedder
    try:
        embedder = get_embedder(test_mode=use_hash_embedding)
    except EmbeddingUnavailable:
        print("⚠ ไม่พบ PyTorch/CUDA ในระบบ — สลับไปใช้ HashEmbedder สำหรับการสาธิต Pipeline")
        embedder = get_embedder(test_mode=True)
        use_hash_embedding = True

    settings = Settings(
        vector_size=embedder.dense_size,
        collection_public="pipeline_demo_public",
        collection_private="pipeline_demo_private",
        search_mode="all",
    )

    # 2. ตั้งค่า ACL Store & User Principal
    acl = AclStore()
    acl.create_box(Box(box_id=BOX_ID, visibility=Visibility.PUBLIC), owner_user_id="demo_user")

    user_principal = acl.principal_for("demo_user", {"management", "sales"}, 2)  # box_ids จากสมาชิกภาพใน AclStore

    # 3. เตรียม Vector Store
    if use_server:
        try:
            store = VectorStore.connect(acl, settings)
            print(" ✓ เชื่อมต่อ Qdrant Live Server สำเร็จ")
        except Exception as e:
            print(f" ⚠ ไม่สามารถเชื่อมต่อ Qdrant Server ({e}) — สลับไปใช้ In-Memory Qdrant")
            store = VectorStore.in_memory(acl, settings)
    else:
        store = VectorStore.in_memory(acl, settings)

    # 4. Ingestion & Quality Gate
    desc_pages = (
        f"{len(target_pages)} หน้า {target_pages}"
        if isinstance(target_pages, list)
        else (f"หน้า {target_pages[0]}-{target_pages[1]}" if isinstance(target_pages, tuple) else "ทุกหน้าในไฟล์")
    )
    print_step(1, f"นำเข้าเอกสาร: {doc_path.name} ({desc_pages})")

    t_start = time.time()
    doc = extract_file(doc_path, pages=target_pages)
    t_extract = time.time() - t_start

    if doc.error:
        print(f" ❌ Extract ไม่ผ่าน: {doc.error.code.value} — {doc.error.message}")
        return 1

    # ลงทะเบียนสิทธิ์ของไฟล์
    acl.register_file(FileAcl(file_id=doc.file_id, box_id=BOX_ID))

    # 5. ทำ Chunking ด้วย make_chunks (ขอบประโยค + Context Line)
    t_chunk_start = time.time()
    all_chunks = []
    ok_units = [u for u in doc.units if u.status == UnitStatus.OK and u.text.strip()]
    for u in ok_units:
        all_chunks.extend(make_chunks(u, target_chars=1000))
    t_chunk = time.time() - t_chunk_start

    # 6. Embed Chunks & Upsert ลง VectorStore
    t_index_start = time.time()
    indexed_chunks = embed_chunks(all_chunks, embedder)
    upsert_report = store.upsert_chunks(doc, indexed_chunks)
    t_index = time.time() - t_index_start
    total_ingest_time = time.time() - t_start

    print(f" ✓ สถานะเอกสาร (Document Status) : {doc.status.value}")
    print(f" ✓ หน้าที่ประมวลผล (Units OK)     : {len(ok_units)} หน้า")
    print(f" ✓ ผลการตัดแบ่ง (Chunks Index)    : {len(all_chunks)} chunks (Sentence-level Spans)")
    print(f" ⏱️ สรุปเวลา Ingestion:")
    print(f"    - Extraction & Normalizer : {t_extract * 1000:.1f} ms")
    print(f"    - Thai SBD Chunking       : {t_chunk * 1000:.1f} ms")
    print(f"    - Vector Embedding & Index: {t_index * 1000:.1f} ms")
    print(f"    - รวมทั้งสิ้น              : {total_ingest_time:.2f} วินาที")

    # 7. ทำการสืบค้นและสร้าง Citations
    if custom_query:
        test_queries = [custom_query]
    else:
        test_queries = [
            "ในปี 2568 รายได้รวมเท่าไหร่ CP all",
            "ในปี 2568 บริษัทขยายสาขาใหม่ในไทยกี่สาขา และมีสาขารวมเท่าไร",
            "วิสัยทัศน์ขององค์กร คืออะไร",
            "การคำนวณวงโคจรดาวเทียมในอวกาศ",  # ทดสอบกรณีไม่มีข้อมูลในเอกสาร
        ]

    print_step(2, f"ทดสอบสืบค้น + 2-Gate ACL + Reranker + Abstention Gate + Citations ({len(test_queries)} คำถาม)")

    gate = ConformalAbstentionGate()

    for i, q in enumerate(test_queries, 1):
        print("\n" + "=" * 85)
        print(f" ❓ คำถามที่ {i}: \"{q}\"")
        print("=" * 85)

        t0 = time.time()
        # แปลง query เป็นเวกเตอร์ dense + sparse
        query_obj = embed_query(q, embedder)

        # สืบค้นผ่าน Qdrant พร้อมตรวจสิทธิ์ 2 ด่าน
        search_res = store.search(
            query=query_obj,
            principal=user_principal,
            box_id=BOX_ID,
            limit=4,
            mode=settings.search_mode,
        )

        candidate_hits = list(search_res.hits)

        # Rerank
        reranked = rerank_hits(q, candidate_hits)
        active_hits = [r.hit for r in reranked]

        # Conformal Abstention Gate
        decision = gate.evaluate(q, reranked)
        query_elapsed = (time.time() - t0) * 1000

        if decision.should_abstain:
            print(f" ⛔ [ABSTENTION GATE ACTIVATED]: ปฏิเสธการตอบ (เหตุผล: {decision.reason})")
            print(f"    - คะแนนความเชื่อมั่น : {decision.confidence_score:.4f}")
            print(f" 🤖 คำตอบมาตรฐาน  : \"{decision.fallback_message or FALLBACK_ABSTAIN_MESSAGE}\"")
            print(f" ⏱️ Latency         : {query_elapsed:.2f} ms")
            continue

        print(f" ✅ [สืบค้นสำเร็จ]: พบข้อมูลเกี่ยวข้อง {len(active_hits)} Chunks (Latency: {query_elapsed:.2f} ms)")
        print(f"    - คะแนน Reranker สูงสุด : {reranked[0].rerank_score:.4f} (Original: {reranked[0].original_score:.4f}, Lexical: {reranked[0].lexical_overlap:.4f})")
        print(f"    - คำสำคัญที่ตรวจพบ     : {', '.join(reranked[0].matched_terms) or '-'}")

        # สร้าง Citations
        citations_map = build_citations_from_hits(active_hits)
        pack = format_citation_pack(active_hits, citations_map)

        # แสดง Top Hit
        top_hit = active_hits[0]
        snippet = " ".join(top_hit.text.split())[:120]
        print(f"\n 📌 [หลักฐานระดับ Top Hit]:")
        print(f"    - ตำแหน่งในไฟล์ : {top_hit.label} (Offset {top_hit.char_start}-{top_hit.char_end})")
        print(f"    - ข้อความย่อ   : \"{snippet}...\"")
        if top_hit.context:
            print(f"    - บริบทโครงสร้าง: \"{top_hit.context}\"")

        # แสดง Citation Tags
        print(f"\n 🏷️ [ALCE Citation Tags ที่สร้างขึ้นจริง (Sentence-level)]: ({len(citations_map)} ประโยค)")
        for tag, item in list(citations_map.items())[:3]:
            print(f"    [{tag}] \"{item.snippet[:65]}...\" (Offset: {item.char_start}-{item.char_end})")

        print(f"\n 📦 [LLM Formatted Context Preview]:")
        preview_lines = pack.formatted_context.split("\n")[:4]
        for line in preview_lines:
            print(f"    {line}")

    print_header("✨ สรุปผล: ระบบรันแบบ Production-Grade Domain-Agnostic สำเร็จสมบูรณ์")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Production Enterprise RAG Pipeline")
    parser.add_argument("--file", "-f", type=str, default=None, help="Document file path (PDF, DOCX, etc.)")
    parser.add_argument("--query", "-q", type=str, default=None, help="Custom query to test")
    parser.add_argument("--pages", "-p", type=str, default=None, help="Pages to ingest (e.g. 1-20 or all)")
    parser.add_argument("--server", action="store_true", help="Connect to Qdrant Docker server")
    parser.add_argument("--hash-embedding", action="store_true", help="Use hash vectors instead of neural model")
    args = parser.parse_args()

    sys.exit(
        run_pipeline(
            file_path=args.file,
            custom_query=args.query,
            pages_arg=args.pages,
            use_server=args.server,
            use_hash_embedding=args.hash_embedding,
        )
    )
