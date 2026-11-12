"""ทดลอง retrieval บนชุด dev: ขนาด chunk x โหมดค้น (dense / +sparse_bge / +sparse_bm25 / ทั้ง 3)

    python scripts/retrieval_dev_eval.py                   # ต้องมี Qdrant (docker compose up -d qdrant) และ BGE-M3 โหลดได้
    python scripts/retrieval_dev_eval.py --sizes 1000      # ลองเร็วด้วยขนาดเดียว

- index เฉพาะ unit ที่ status = ok ของ PDF One Report ทั้ง 3 ไฟล์ (ไม่ OCR: หน้า needs_ocr ไม่ถูก index — รายงานจำนวนคำถามที่หลักฐานไม่อยู่ใน
  ข้อความของ unit ok) ทั้ง 3 ไฟล์อยู่ box เดียว ค้นรวมกัน (ไม่บอกระบบว่าคำถามมาจากไฟล์ไหน) ไม่มี ACL กรอง (box public เดียว)
- ชุด dev = eval/datasets/rag_evaluation_100.jsonl เฉพาะ type=question (มี expected_citation) 70 ข้อ
- hit = chunk ใน top-k มีข้อความใกล้ expected_citation: ``rapidfuzz.fuzz.partial_ratio(canon(citation), canon(chunk)) >= 80``
  (canon = eval/ocr_canon.py ตัวเดียวกับงานวัด OCR) · Recall@1 / @5 · MRR (top 10; ไม่เจอใน 10 = 0)
- index ครั้งเดียวต่อขนาด chunk แล้วสลับโหมดตอน query; embedding คำถามทำครั้งเดียวใช้ทุกขนาด
- **ไม่ปรับพารามิเตอร์อื่น** ให้ตัวเลขดีขึ้น: ใช้ค่าเริ่มต้นของ config ทั้งหมด (max_length, batch, prefetch ฯลฯ)
- collection แยกชื่อสำหรับทดลอง (``rag_dev_eval_<ขนาด>_public/private``) ลบทิ้งท้ายเสมอ
- ผลลง eval/results/retrieval_dev_eval.json + .md

กฎเลือก config: ตัวที่ห่างจากดีที่สุดไม่เกิน 2 จุด (0.02) ของตัวชี้วัดถือว่าเท่ากัน แล้วเลือก "ตัวที่ง่ายกว่า":
จำนวน vector ที่ใช้น้อยกว่า (dense 1 < +sparse 2 < all 3) -> จำนวน chunk น้อยกว่า (ขนาดใหญ่กว่า) -> MRR สูงกว่า
คำนวณให้ 3 ตัวชี้วัด (Recall@5, MRR, Recall@1) ทุกครั้ง เพราะตัวเลขต่างกันระดับ 1–3 คำถามจาก 70 และตัวชี้วัดต่างกันเลือกได้ต่างกัน
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapidfuzz import fuzz  # noqa: E402

from app.access import AclStore, Box, FileAcl, Visibility  # noqa: E402
from app.chunking import make_document_chunks  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.embedding import get_embedder, gpu_peak_memory_mb, release_gpu_memory, reset_gpu_peak  # noqa: E402
from app.extraction.pipeline import extract_file  # noqa: E402
from app.models import UnitStatus  # noqa: E402
from app.vectorstore import SEARCH_MODES, VectorStore, embed_chunks, embed_query  # noqa: E402
from eval.ocr_canon import canon  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "eval" / "datasets"
RESULTS = ROOT / "eval" / "results"
DEV_SET = DATASETS / "rag_evaluation_100.jsonl"
DOCS = {"CPALL": "E_ONE_REPORT_CPALLT.PDF", "PTT": "ONEREPORTPTTT.PDF", "KBANK": "ONEREPORTKBANKT.PDF"}
SIZES = (600, 1000, 1500)
MODES = tuple(SEARCH_MODES)
LIMIT = 10  # top-k ที่ดึงมาคิด MRR
FUZZY_THRESHOLD = 80
SELECTION_TOLERANCE = 0.02  # 2 จุดเปอร์เซ็นต์ของ Recall@5
BOX_ID = "dev"


def load_questions() -> list[dict]:
    rows = [json.loads(line) for line in DEV_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r["type"] == "question" and r["expected_citation"]]


def is_hit(citation_canon: str, text_canon: str) -> bool:
    return bool(citation_canon) and fuzz.partial_ratio(citation_canon, text_canon) >= FUZZY_THRESHOLD


def first_hit_rank(citation_canon: str, texts: list[str], cache: dict[str, str]) -> int | None:
    for rank, text in enumerate(texts, start=1):
        if text not in cache:
            cache[text] = canon(text)
        if is_hit(citation_canon, cache[text]):
            return rank
    return None


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    if n == 0:
        return {"n": 0, "recall_at_1": None, "recall_at_5": None, "mrr": None}
    return {
        "n": n,
        "recall_at_1": sum(1 for r in ranks if r is not None and r <= 1) / n,
        "recall_at_5": sum(1 for r in ranks if r is not None and r <= 5) / n,
        "mrr": sum(1 / r for r in ranks if r is not None) / n,
    }


def coverage(documents: dict, questions: list[dict]) -> dict:
    """ต่อไฟล์: จำนวน unit ต่อสถานะ + คำถามที่ expected_citation อยู่ในข้อความของ unit ok หรือไม่ (ใช้เกณฑ์ hit เดียวกัน)

    คำถามที่ไม่เจอใน unit ok ใดเลย = หลักฐานอยู่ในหน้า needs_ocr (ไม่ถูก index) **หรือ** expected_citation เองไม่ตรงกับข้อความในไฟล์
    (สองกรณีนี้แยกจากกันไม่ได้โดยไม่ OCR) — ถ้าไฟล์ไม่มี unit needs_ocr เลย กรณีแรกเป็นไปไม่ได้
    """
    out = {}
    for short, doc in documents.items():
        ok_texts = [canon(u.text) for u in doc.units if u.status == UnitStatus.OK and u.text.strip()]
        qs = [q for q in questions if q["document_source"] == DOCS[short]]
        in_ok = [q["question_id"] for q in qs if any(is_hit(canon(q["expected_citation"]), t) for t in ok_texts)]
        statuses: dict[str, int] = {}
        for u in doc.units:
            statuses[u.status.value] = statuses.get(u.status.value, 0) + 1
        out[short] = {
            "file": DOCS[short], "units": len(doc.units), "unit_status": statuses,
            "ok_chars": sum(len(u.text) for u in doc.units if u.status == UnitStatus.OK),
            "questions": len(qs), "citation_in_ok_units": len(in_ok),
            "citation_not_in_ok_units": len(qs) - len(in_ok),
            "in_ok_units_ids": in_ok,
        }
    return out


def select_best(rows: list[dict], chunk_counts: dict[int, int], metric: str = "recall_at_5") -> dict:
    """ตัวที่ ``metric`` ห่างจากดีที่สุดไม่เกิน 2 จุด (0.02) ถือว่าเท่ากัน -> เลือกตัวที่ง่ายกว่า"""
    best = max(r["all"][metric] for r in rows)
    near = [r for r in rows if r["all"][metric] >= best - SELECTION_TOLERANCE - 1e-9]
    near.sort(key=lambda r: (len(SEARCH_MODES[r["mode"]]), chunk_counts[r["size"]], -r["all"]["mrr"], -r["all"]["recall_at_5"]))
    return {"metric": metric, "best": best, "tolerance": SELECTION_TOLERANCE,
            "candidates": [(r["size"], r["mode"], r["all"][metric]) for r in near], "chosen": (near[0]["size"], near[0]["mode"])}


SELECTION_METRICS = ("recall_at_5", "mrr", "recall_at_1")


def qdrant_version(url: str) -> str:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read())["version"]
    except Exception:
        return "?"


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def render_md(out: dict) -> str:
    L = ["# ทดลอง retrieval บนชุด dev (chunk x โหมดค้น)", "",
         f"- ชุด dev: `rag_evaluation_100.jsonl` เฉพาะ type=question {out['questions']} ข้อ · hit = partial_ratio ≥ {FUZZY_THRESHOLD} บน canon · top-{LIMIT} สำหรับ MRR",
         f"- index เฉพาะ unit status=ok ของ 3 ไฟล์ใน box เดียว (ค้นรวม ไม่บอกว่าคำถามมาจากไฟล์ไหน) ไม่ OCR · embedding = BGE-M3 {out['dtype']} (ค่าเริ่มต้นของ config ทั้งหมด)",
         f"- Qdrant {out['qdrant']} · GPU {out['gpu']}", "",
         "## 1. หน้าที่ถูก index / หลักฐานของคำถาม", "",
         "| ไฟล์ | unit ทั้งหมด | ok | needs_ocr | คำถาม | citation อยู่ในข้อความ unit ok | ไม่อยู่ใน unit ok |", "|---|---:|---:|---:|---:|---:|---:|"]
    for short, c in out["coverage"].items():
        L.append(f"| {short} | {c['units']} | {c['unit_status'].get('ok', 0)} | {c['unit_status'].get('needs_ocr', 0)} | {c['questions']} "
                 f"| {c['citation_in_ok_units']} | {c['citation_not_in_ok_units']} |")
    L += ["", "> 'ไม่อยู่ใน unit ok' = หลักฐานน่าจะอยู่ในหน้า needs_ocr ที่ไม่ได้ index (หรือ expected_citation ไม่ตรงกับข้อความในไฟล์ — แยกไม่ได้โดยไม่ OCR) "
              "คำถามกลุ่มนี้ค้นเจอไม่ได้ในทุก config จึงเป็นเพดานของ recall", "",
          "## 2. ต้นทุน index ต่อขนาด chunk", "",
          "| CHUNK_TARGET_CHARS | จำนวน chunk | ขนาดเฉลี่ย (ตัวอักษร) | เวลา index รวม | chunk | embed | upsert | VRAM สูงสุด (MB) | แถว fp16 ได้ NaN (คำนวณใหม่ / ต้องใช้ fp32) |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for size, c in out["index"].items():
        L.append(f"| {size} | {c['chunks']} | {c['avg_chars']:.0f} | {c['seconds_total']:.1f}s | {c['seconds_chunk']:.1f}s | {c['seconds_embed']:.1f}s "
                 f"| {c['seconds_upsert']:.1f}s | {c['vram_peak_mb']:.0f} | {c['retries']} / {c['fp32_fallbacks']} |")
    L += ["", "VRAM สูงสุด = `torch.cuda.max_memory_allocated` ช่วง index (รวมน้ำหนักโมเดล ~1.1 GB) · ไม่มี Typhoon บน GPU ในขั้นตอนนี้", "",
          f"## 3. ผลรวมทุกข้อ ({out['questions']} ข้อ)", "",
          "| CHUNK_TARGET_CHARS | โหมด | Recall@1 | Recall@5 | MRR | query เฉลี่ย (ms) |", "|---:|---|---:|---:|---:|---:|"]
    for r in out["rows"]:
        a = r["all"]
        L.append(f"| {r['size']} | {r['mode']} | {pct(a['recall_at_1'])} | {pct(a['recall_at_5'])} | {a['mrr']:.3f} | {r['query_ms']:.0f} |")
    L += ["", "## 4. แยกตามไฟล์ (Recall@1 / Recall@5 / MRR)", "",
          "| CHUNK_TARGET_CHARS | โหมด | " + " | ".join(f"{d} (n={out['coverage'][d]['questions']})" for d in DOCS) + " |",
          "|---:|---|" + "---|" * len(DOCS)]
    for r in out["rows"]:
        cells = [f"{pct(r['by_doc'][d]['recall_at_1'])} / {pct(r['by_doc'][d]['recall_at_5'])} / {r['by_doc'][d]['mrr']:.3f}" for d in DOCS]
        L.append(f"| {r['size']} | {r['mode']} | " + " | ".join(cells) + " |")
    L += ["", "> KBANK ทั้งไฟล์เป็นหน้า needs_ocr (ไม่มีข้อความให้ index) จึงได้ 0% ทุก config — ไม่ใช่ความผิดของการค้น · truth ของ KBANK บางส่วนพังอยู่แล้ว", "",
          f"## 5. เฉพาะคำถามที่ citation อยู่ในข้อความ unit ok ({out['reachable_n']} ข้อ)", "",
          "| CHUNK_TARGET_CHARS | โหมด | Recall@1 | Recall@5 | MRR |", "|---:|---|---:|---:|---:|"]
    for r in out["rows"]:
        a = r["reachable"]
        L.append(f"| {r['size']} | {r['mode']} | {pct(a['recall_at_1'])} | {pct(a['recall_at_5'])} | {a['mrr']:.3f} |")
    L += ["", "## 6. config ที่ดีที่สุดตามกฎ \"ห่างไม่เกิน 2 จุดให้เลือกตัวที่ง่ายกว่า\"", "",
          "ง่ายกว่า = จำนวน vector ที่ใช้น้อยกว่า → จำนวน chunk น้อยกว่า (ขนาดใหญ่กว่า) → MRR สูงกว่า · 2 จุด = 0.02 ของตัวชี้วัด (Recall = 2 จุดเปอร์เซ็นต์)", ""]
    for sel in out["selections"]:
        fmt = pct if sel["metric"] != "mrr" else (lambda v: f"{v:.3f}")
        L += [f"**ตัวชี้วัด {sel['metric']}** (รวมทุกข้อ) · ดีที่สุด {fmt(sel['best'])}",
              "- เข้าเกณฑ์ (เรียงจากง่ายไปซับซ้อน): " + ", ".join(f"{s}/{m} ({fmt(v)})" for s, m, v in sel["candidates"]),
              f"- **เลือก: CHUNK_TARGET_CHARS={sel['chosen'][0]}, โหมด {sel['chosen'][1]}**", ""]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ทดลอง chunk x โหมดค้น บนชุด dev")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--no-write", action="store_true", help="พิมพ์ผลอย่างเดียว ไม่เขียน eval/results/")
    ap.add_argument("--render-only", action="store_true", help="สร้าง .md ใหม่จาก .json เดิม (ไม่รันการทดลอง)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.render_only:
        out = json.loads((RESULTS / "retrieval_dev_eval.json").read_text(encoding="utf-8"))
        out["index"] = {int(k): v for k, v in out["index"].items()}
        out.setdefault("dtype", get_settings().embedding_dtype)
        chunk_counts = {s: v["chunks"] for s, v in out["index"].items()}
        out["selections"] = [select_best(out["rows"], chunk_counts, m) for m in SELECTION_METRICS]
        out.pop("selection", None)
        (RESULTS / "retrieval_dev_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        (RESULTS / "retrieval_dev_eval.md").write_text(render_md(out), encoding="utf-8")
        return 0

    base = get_settings()
    questions = load_questions()
    print(f"คำถามที่มี citation: {len(questions)}", flush=True)

    t = time.perf_counter()
    documents = {short: extract_file(DATASETS / name, file_id=short) for short, name in DOCS.items()}
    for short, doc in documents.items():
        if doc.error:
            raise SystemExit(f"extract {short} ไม่ผ่าน: {doc.error.message}")
    print(f"extract 3 ไฟล์ {time.perf_counter() - t:.1f}s (ไม่ OCR)", flush=True)
    cov = coverage(documents, questions)
    reachable_ids = {qid for c in cov.values() for qid in c["in_ok_units_ids"]}

    acl = AclStore()
    acl.create_box(Box(box_id=BOX_ID, visibility=Visibility.PUBLIC), owner_user_id="dev")
    for short in DOCS:
        acl.register_file(FileAcl(file_id=short, box_id=BOX_ID))
    principal = acl.principal_for("dev")

    embedder = get_embedder(base)  # โหลด BGE-M3 ไม่ได้ = ล้ม (ไม่ fallback)
    q_canon = {q["question_id"]: canon(q["expected_citation"]) for q in questions}
    queries = {q["question_id"]: embed_query(q["query"], embedder) for q in questions}

    index_info: dict[int, dict] = {}
    rows: list[dict] = []
    text_canon: dict[str, str] = {}
    for size in args.sizes:
        settings = base.model_copy(update={
            "collection_public": f"rag_dev_eval_{size}_public", "collection_private": f"rag_dev_eval_{size}_private",
            "chunk_target_chars": size})
        store = VectorStore.connect(acl, settings)
        for name in store.collections:  # ทดลองต้องเริ่มสะอาด
            if store.client.collection_exists(name):
                store.client.delete_collection(name)
        store.ensure_collections()
        try:
            reset_gpu_peak()
            retries0, fallbacks0 = embedder.retries, embedder.fp32_fallbacks
            t0 = time.perf_counter()
            chunk_s = embed_s = upsert_s = 0.0
            n_chunks = total_chars = 0
            for short, doc in documents.items():
                ok_units = [u for u in doc.units if u.status == UnitStatus.OK and u.text.strip()]
                t = time.perf_counter()
                chunks = make_document_chunks(ok_units, size)
                chunk_s += time.perf_counter() - t
                t = time.perf_counter()
                items = embed_chunks(chunks, embedder)
                embed_s += time.perf_counter() - t
                t = time.perf_counter()
                store.upsert_chunks(doc, items)
                upsert_s += time.perf_counter() - t
                n_chunks += len(chunks)
                total_chars += sum(len(c.text) for c in chunks)
            index_info[size] = {
                "chunks": n_chunks, "avg_chars": total_chars / max(n_chunks, 1),
                "seconds_total": time.perf_counter() - t0, "seconds_chunk": chunk_s, "seconds_embed": embed_s,
                "seconds_upsert": upsert_s, "vram_peak_mb": gpu_peak_memory_mb(),
                "retries": embedder.retries - retries0, "fp32_fallbacks": embedder.fp32_fallbacks - fallbacks0,
            }
            print(f"[{size}] {n_chunks} chunk · index {index_info[size]['seconds_total']:.1f}s "
                  f"(embed {embed_s:.1f}s) · VRAM {index_info[size]['vram_peak_mb']:.0f} MB", flush=True)

            for mode in MODES:
                ranks: dict[int, int | None] = {}
                t = time.perf_counter()
                for q in questions:
                    result = store.search(queries[q["question_id"]], principal, BOX_ID, limit=LIMIT, mode=mode)
                    ranks[q["question_id"]] = first_hit_rank(q_canon[q["question_id"]], [h.text for h in result.hits], text_canon)
                query_ms = 1000 * (time.perf_counter() - t) / len(questions)
                by_doc = {d: metrics([ranks[q["question_id"]] for q in questions if q["document_source"] == DOCS[d]]) for d in DOCS}
                row = {
                    "size": size, "mode": mode, "query_ms": query_ms,
                    "all": metrics(list(ranks.values())), "by_doc": by_doc,
                    "reachable": metrics([r for qid, r in ranks.items() if qid in reachable_ids]),
                    "ranks": ranks,
                }
                rows.append(row)
                a = row["all"]
                print(f"  {mode:18s} R@1 {pct(a['recall_at_1'])}  R@5 {pct(a['recall_at_5'])}  MRR {a['mrr']:.3f}  ({query_ms:.0f} ms/query)", flush=True)
        finally:
            for name in store.collections:
                if store.client.collection_exists(name):
                    store.client.delete_collection(name)

    embedder.release()
    release_gpu_memory()
    import torch

    out = {
        "questions": len(questions), "reachable_n": len(reachable_ids), "coverage": cov, "index": index_info,
        "rows": rows,
        "selections": [select_best(rows, {s: index_info[s]["chunks"] for s in index_info}, m) for m in SELECTION_METRICS],
        "qdrant": qdrant_version(base.qdrant_url), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "fuzzy_threshold": FUZZY_THRESHOLD, "limit": LIMIT, "dtype": base.embedding_dtype,
    }
    print("\n" + render_md(out))
    if not args.no_write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "retrieval_dev_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        (RESULTS / "retrieval_dev_eval.md").write_text(render_md(out), encoding="utf-8")
        print(f"เขียน {RESULTS / 'retrieval_dev_eval.json'} และ .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
