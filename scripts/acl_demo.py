"""เดโมสิทธิ์: ingest ไฟล์จริงเข้า vector store แล้วถามคำถามเดียวกันด้วยคนละ persona

    python scripts/acl_demo.py                                  # ใช้ mock_corpus.pdf + Qdrant ในกระบวนการ
    python scripts/acl_demo.py --file eval/datasets/ONEREPORTPTTT.PDF --pages 1-20
    python scripts/acl_demo.py --server                         # ใช้ Qdrant จาก docker compose

สิ่งที่ควรเห็น: persona ที่มีสิทธิ์ได้ผลค้นพร้อม label อ้างอิง ("หน้า 12") ส่วน persona ที่ไม่มีสิทธิ์
ได้ผลลัพธ์ที่**หน้าตาเหมือนกันทุกประการ**กับการค้นที่ไม่เจออะไร (กฎเหล็กข้อ 9)

หมายเหตุ: ``app/embedding.py`` ยังไม่มีโมเดลจริง (ไม่มี sentence-transformers ใน requirements.txt)
สคริปต์นี้จะพิมพ์บอกว่ากำลังใช้เวกเตอร์จาก fallback — อันดับผลค้นจึงยัง**ไม่มีความหมายเชิงความหมาย**
เดโมนี้แสดงเรื่องสิทธิ์และการอ้างอิง ไม่ได้แสดงคุณภาพการค้น
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.access import AclStore, Box, FileAcl, Principal, Visibility  # noqa: E402
from app.config import Settings  # noqa: E402
from app.embedding import EmbeddingModel  # noqa: E402
from app.extraction.pipeline import extract_file  # noqa: E402
from app.vectorstore import VectorStore  # noqa: E402
from scripts.extract_cli import parse_pages  # noqa: E402

BOX_ID = "hr_private"
QUERY = "การลาป่วย"


def build_acl(file_id: str) -> tuple[AclStore, Principal, Principal]:
    """box เดียวแบบ private จำกัดแผนก hr + persona สองคนที่ต่างกันแค่แผนก"""
    acl = AclStore()
    acl.register_box(
        Box(box_id=BOX_ID, visibility=Visibility.PRIVATE, departments=frozenset({"hr"}))
    )
    acl.register_file(FileAcl(file_id=file_id, box_id=BOX_ID))
    allowed = Principal(
        user_id="somchai (hr)", departments=frozenset({"hr"}), security_level=2,
        box_ids=frozenset({BOX_ID}),
    )
    denied = Principal(  # เป็นสมาชิก box เดียวกัน ระดับเท่ากัน ต่างแค่แผนก
        user_id="wanida (finance)", departments=frozenset({"finance"}), security_level=2,
        box_ids=frozenset({BOX_ID}),
    )
    return acl, allowed, denied


def show(store: VectorStore, principal: Principal, query_vector, limit: int) -> None:
    result = store.search(query_vector, principal, BOX_ID, limit=limit)
    print(f"\n--- persona: {principal.user_id}  (แผนก {', '.join(sorted(principal.departments))})")
    if not result.hits:
        print("  ไม่พบข้อมูลที่เกี่ยวข้อง")  # ข้อความเดียวกันทั้งกรณีไม่มีสิทธิ์และกรณีไม่เจอ
        return
    for hit in result.hits:
        snippet = " ".join(hit.text.split())[:70]
        print(f"  [{hit.score:.3f}] {hit.filename} · {hit.label} — {snippet}…")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="เดโม ACL บน vector store")
    ap.add_argument("--file", default="mock_corpus.pdf")
    ap.add_argument("--pages", type=parse_pages, help="ช่วงหน้า เช่น 1-20 (PDF)")
    ap.add_argument("--query", default=QUERY)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--server", action="store_true", help="ใช้ Qdrant จาก docker compose")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    doc = extract_file(args.file, pages=args.pages)
    if doc.error:
        print(f"extract ไม่ผ่าน: {doc.error.code.value} — {doc.error.message}")
        return 1
    print(f"ไฟล์: {doc.filename} · unit {doc.unit_count} · สถานะ {doc.status.value}")
    print("unit ต่อสถานะ: " + "  ".join(f"{k}={v}" for k, v in doc.unit_summary.items()))

    embedder = EmbeddingModel()
    # เข้าถึง attribute ภายในเพราะ EmbeddingModel ยังไม่บอกว่าใช้เส้นทางไหน (ดูรายงานท้ายงาน)
    if getattr(embedder, "_st_model", None) is None:
        print("⚠ embedding: ใช้เวกเตอร์ fallback (ไม่ใช่ BGE-M3) — อันดับผลค้นยังไม่มีความหมาย")
    settings = Settings(vector_size=embedder.dimension)

    acl, allowed, denied = build_acl(doc.file_id)
    store = (
        VectorStore.connect(acl, settings) if args.server else VectorStore.in_memory(acl, settings)
    )
    if args.server:
        store.ensure_collections()
        store.delete_file(doc.file_id)  # ให้รันซ้ำได้ผลเดิม

    ok_units = [u for u in doc.units if u.status.value == "ok" and u.text.strip()]
    vectors = {
        u.index: v for u, v in zip(ok_units, embedder.embed_texts([u.text for u in ok_units]))
    }
    report = store.upsert_units(doc, vectors)
    print(
        f"index: {report.indexed} unit เข้า collection '{report.collection}' "
        f"(ข้าม {report.skipped_not_ok} ที่ไม่ใช่ ok, {report.skipped_empty} ที่ว่าง)"
    )

    print(f"\nคำถาม: {args.query!r}  ใน box {BOX_ID} (private, เฉพาะแผนก hr)")
    query_vector = embedder.embed_query(args.query)
    show(store, allowed, query_vector, args.limit)
    show(store, denied, query_vector, args.limit)
    print("\nสองบล็อกข้างบนต่างกันเพราะสิทธิ์เท่านั้น — ฝั่งที่ไม่มีสิทธิ์ตอบเหมือนหาไม่เจอ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
