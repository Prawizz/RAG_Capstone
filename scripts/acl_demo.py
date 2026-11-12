"""เดโมสิทธิ์: ingest ไฟล์จริงเข้า vector store แล้วถามคำถามเดียวกันด้วยคนละ persona

    python scripts/acl_demo.py                                  # ใช้ mock_corpus.pdf + Qdrant ในกระบวนการ
    python scripts/acl_demo.py --file eval/datasets/ONEREPORTPTTT.PDF --pages 1-20
    python scripts/acl_demo.py --server                         # ใช้ Qdrant จาก docker compose

สิ่งที่ควรเห็น: persona ที่มีสิทธิ์ได้ผลค้นพร้อม label อ้างอิง ("หน้า 12") ส่วน persona ที่ไม่มีสิทธิ์
ได้ผลลัพธ์ที่**หน้าตาเหมือนกันทุกประการ**กับการค้นที่ไม่เจออะไร (กฎเหล็กข้อ 9)

ใช้ BGE-M3 จริง (ต้องโหลดได้ ไม่งั้นสคริปต์ล้ม) เพิ่ม ``--hash-embedding`` ถ้าต้องการเดโมสิทธิ์โดยไม่โหลดโมเดล
(hash vector ไม่มีความหมายเชิงความหมาย — อันดับผลค้นไม่มีความหมาย) ``--mode`` เลือกโหมดค้นของ ``VectorStore.search``
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.access import AclStore, Box, BoxRole, FileAcl, Principal, Visibility  # noqa: E402
from app.config import Settings  # noqa: E402
from app.embedding import get_embedder  # noqa: E402
from app.extraction.pipeline import extract_file  # noqa: E402
from app.vectorstore import SEARCH_MODES, VectorStore, embed_query  # noqa: E402
from scripts.extract_cli import parse_pages  # noqa: E402

BOX_ID = "hr_private"
QUERY = "การลาป่วย"


def build_acl(file_id: str) -> tuple[AclStore, Principal, Principal]:
    """box เดียวแบบ private จำกัดแผนก hr + persona สองคนที่ต่างกันแค่แผนก"""
    acl = AclStore()
    acl.create_box(
        Box(box_id=BOX_ID, visibility=Visibility.PRIVATE, departments=frozenset({"hr"})), owner_user_id="box-owner"
    )
    acl.register_file(FileAcl(file_id=file_id, box_id=BOX_ID))
    for user in ("somchai (hr)", "wanida (finance)"):
        acl.set_member(BOX_ID, user, BoxRole.VIEWER)
    allowed = acl.principal_for("somchai (hr)", {"hr"}, 2)
    denied = acl.principal_for("wanida (finance)", {"finance"}, 2)  # เป็นสมาชิก box เดียวกัน ระดับเท่ากัน ต่างแค่แผนก
    return acl, allowed, denied


def show(store: VectorStore, principal: Principal, query, limit: int, mode: str) -> None:
    result = store.search(query, principal, BOX_ID, limit=limit, mode=mode)
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
    ap.add_argument("--mode", choices=sorted(SEARCH_MODES), default="all")
    ap.add_argument("--hash-embedding", action="store_true", help="ใช้ hash vector แทน BGE-M3 (เฉพาะเดโมสิทธิ์)")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    doc = extract_file(args.file, pages=args.pages)
    if doc.error:
        print(f"extract ไม่ผ่าน: {doc.error.code.value} — {doc.error.message}")
        return 1
    print(f"ไฟล์: {doc.filename} · unit {doc.unit_count} · สถานะ {doc.status.value}")
    print("unit ต่อสถานะ: " + "  ".join(f"{k}={v}" for k, v in doc.unit_summary.items()))

    embedder = get_embedder(test_mode=args.hash_embedding)  # โหลด BGE-M3 ไม่ได้ = raise
    if args.hash_embedding:
        print("⚠ embedding: hash vector (ไม่ใช่ BGE-M3) — อันดับผลค้นไม่มีความหมาย")
    settings = Settings(vector_size=embedder.dense_size)

    acl, allowed, denied = build_acl(doc.file_id)
    store = (
        VectorStore.connect(acl, settings) if args.server else VectorStore.in_memory(acl, settings)
    )
    if args.server:
        store.ensure_collections()
        store.delete_file(doc.file_id)  # ให้รันซ้ำได้ผลเดิม

    report = store.upsert_document(doc, embedder)
    print(
        f"index: {report.indexed} chunk เข้า collection '{report.collection}' "
        f"(ข้าม {report.skipped_not_ok} ที่ไม่ใช่ ok, {report.skipped_empty} ที่ว่าง)"
    )

    print(f"\nคำถาม: {args.query!r}  ใน box {BOX_ID} (private, เฉพาะแผนก hr)")
    query = embed_query(args.query, embedder)
    show(store, allowed, query, args.limit, args.mode)
    show(store, denied, query, args.limit, args.mode)
    print("\nสองบล็อกข้างบนต่างกันเพราะสิทธิ์เท่านั้น — ฝั่งที่ไม่มีสิทธิ์ตอบเหมือนหาไม่เจอ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
