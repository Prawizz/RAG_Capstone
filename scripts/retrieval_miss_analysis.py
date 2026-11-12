"""วิเคราะห์ข้อที่พลาดของ config ที่เลือก (CHUNK_TARGET_CHARS=1000 + โหมด dense+sparse_bge) — วิเคราะห์อย่างเดียว ไม่แก้ระบบค้น

    python scripts/retrieval_miss_analysis.py

- เฉพาะข้อที่ "ค้นเจอได้" = expected_citation อยู่ในข้อความของ unit ok (partial_ratio ≥ เกณฑ์ บน canon) แต่ไม่ติด top-5
- ดึง top-50 ต่อคำถาม (อันดับจริงของ chunk ที่ถูก; ไม่ติด top-50 = "-") ทำตารางทีละข้อ + สาเหตุ
- ทดลองเกณฑ์ fuzzy 70 / 80 / 90 เพื่อดูว่า Recall@5 ไวต่อเกณฑ์แค่ไหน (ตารางหลักใช้ 80 ตามที่ตกลง)
- ใช้ collection ชื่อแยก (``rag_miss_analysis_*``) และลบทิ้งท้ายเสมอ; ผลลง eval/results/retrieval_miss_analysis.json + .md

สาเหตุ (เลือกได้ข้อเดียวต่อข้อ):
  a) citation ไม่ตรงกับข้อความในไฟล์ (fuzzy เกือบถึงเกณฑ์)   b) มีหลายหน้าตอบได้แต่ truth ระบุไว้หน้าเดียว
  c) chunk ที่ถูกโดนตัดขาด   d) ค้นพลาดจริง   e) อื่นๆ
ป้ายแรกคำนวณอัตโนมัติจากหลักฐาน (คะแนน fuzzy ของ chunk ที่ดีที่สุด / คำตอบอยู่ใน top-5 หรือไม่) แล้วคนอ่านแถวต่อแถวและแก้ทับใน ``MANUAL_CAUSES``
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapidfuzz import fuzz  # noqa: E402

from app.access import AclStore, Box, FileAcl, Visibility  # noqa: E402
from app.chunking import make_document_chunks  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.embedding import get_embedder  # noqa: E402
from app.extraction.pipeline import extract_file  # noqa: E402
from app.models import UnitStatus  # noqa: E402
from app.vectorstore import VectorStore, embed_chunks, embed_query  # noqa: E402
from eval.ocr_canon import canon  # noqa: E402
from scripts.retrieval_dev_eval import BOX_ID, DATASETS, DOCS, RESULTS, load_questions, qdrant_version  # noqa: E402

SIZE = 1000
MODE = "dense+sparse_bge"
DEEP = 50  # ลึกสุดที่ดูอันดับ
THRESHOLDS = (70, 80, 90)
MAIN_THRESHOLD = 80
# ข้อที่พลาดตามเกณฑ์เข้มงวด แต่คำตอบอยู่ใน top-5 (ดู MANUAL_CAUSES): ตรงตัวอักษร / ตัดสินจากการอ่านเนื้อ
ANSWER_LITERAL_IN_TOP5 = {4, 28, 29, 31, 36, 62, 63, 59}  # 59 = chunk ที่ถูก (หน้า 5) อยู่อันดับ 3 แต่ extract สลับบรรทัด fuzzy จึงไม่ถึงเกณฑ์
ANSWER_BY_READING_IN_TOP5 = {40, 48, 49, 69}
CAUSES = {"a": "citation ไม่ตรงกับข้อความในไฟล์", "b": "หลายหน้าตอบได้ แต่ truth ระบุหน้าเดียว", "c": "chunk ที่ถูกโดนตัดขาด",
          "d": "ค้นพลาดจริง", "e": "อื่นๆ — หลักฐานอยู่ในข้อความที่ถูก flatten จากกราฟ/ตาราง/แผนผัง"}

# คนอ่านแต่ละแถว (chunk อันดับ 1-3, chunk ที่ถูก, ตำแหน่งคำตอบในข้อความ) แล้วตัดสินทับป้ายอัตโนมัติ: question_id -> (สาเหตุ, หมายเหตุ)
MANUAL_CAUSES: dict[int, tuple[str, str]] = {
    # ตรวจ top-5 ด้วยการค้นข้อความคำตอบตรงตัว ("ตรงตัว") หรืออ่านเนื้อ chunk เทียบความหมาย ("อ่านเนื้อ")
    3: ("a", "ข้อความหน้า 5 สลับลำดับตอน extract: \"รายได้รวม ล้านบาท … ร้อยละ11 1,022,143\" ตัวเลขหลุดไปท้ายประโยค (fuzzy 90.6) "
             "chunk จึงไม่มีวลี \"รายได้รวม 1,022,143\"; top-5 ไม่มี chunk ที่มีเลข 1,022,143 เลย"),
    4: ("b", "ตรงตัว: top-2/3 (หน้า 145) ระบุ \"กำไรสุทธิเท่ากับ 28,206 ล้านบาท\"; truth ระบุหน้า 5 ซึ่งมีปัญหาลำดับข้อความเหมือนข้อ 3"),
    5: ("d", "chunk ที่ถูก (หน้า 7) อันดับ 7; top-5 ไม่มี \"700 สาขา\"/\"15,945\": คำถามขึ้นต้น \"ปี 2568 ซีพี ออลล์\" ซึ่งตรงกับทุกหน้า "
             "ส่วน \"ขยายสาขาใหม่\" ถูกกลบ; top-1..3 เป็นเรื่อง SMEs/พันธกิจ"),
    6: ("d", "chunk เดียวกับข้อ 5 อันดับ 15; top-5 ไม่มี \"15,945\": top-1..3 เป็นเนื้อหาทั่วไปของบริษัท (พันธกิจ ประวัติ ที่ตั้ง)"),
    28: ("b", "ตรงตัว: หน้า 108 (อันดับ 1) 148, 241 ระบุ 28,562,996,250 หุ้น; truth ระบุหน้า 8"),
    29: ("b", "ตรงตัว: หน้า 39 (อันดับ 1) และ 108 (อันดับ 2) ระบุ \"หุ้นละ 1 บาท\"; truth ระบุหน้า 8"),
    31: ("b", "ตรงตัว: หน้า 198 (อันดับ 1) ตารางสรุปผลการดำเนินงาน \"EBITDA 396,234 332,849\"; truth ระบุตาราง/กราฟหน้า 10 (chunk นั้นถูก flatten จากกราฟ อันดับ 9)"),
    36: ("b", "ตรงตัว: หน้า 316 และ 315 (อันดับ 1-2) ระบุ \"1365 Contact Center\"; truth ระบุหน้า 8"),
    40: ("b", "อ่านเนื้อ: หน้า 20 (อันดับ 3) บรรยาย 7Delivery / All Online เชื่อม Online-Offline จัดส่งถึงปลายทาง ตอบคำถามเดียวกับหน้า 7 ที่ truth ระบุ (อันดับ 17) แต่ไม่ได้พูดถึง 7App"),
    48: ("b", "อ่านเนื้อ: หน้า 140 (อันดับ 4-5) \"SMEs โตไกลไปด้วยกัน … เติบโตของ SMEs ผู้ประกอบการชุมชน และเกษตรกร\" ตอบเรื่องเดียวกับ truth (หน้า 5 อันดับ 19)"),
    49: ("b", "อ่านเนื้อ: หน้า 8 (อันดับ 1, 3, 5) อธิบายว่าพนักงาน/บุคลากรขับเคลื่อนธุรกิจและความสุขในการทำงาน ตอบคำถามเหตุผลได้ แต่ไม่ใช่ประโยค \"หัวใจหลักของการบริการ\" ที่ truth ระบุ (หน้า 5 อันดับ 14)"),
    59: ("a", "หน้า 5 (อันดับ 3 — ติด top-5) มีประโยคนี้จริง แต่ extract สลับบรรทัดกับ tagline (\"ปตท. แข็งแรงร่วมกับสังคมไทย…\" แทรกกลาง) fuzzy ของ chunk นั้นเหลือ 63; "
             "ที่ได้ 86.7 คือหน้า 29 ที่เขียนต่างออกไป (\"และดูแล… อย่างสมดุลและยั่งยืน\") ซึ่งไม่ติด top-50"),
    61: ("e", "chunk ที่ถูก (หน้า 9 chunk 0) เป็นรายชื่อหัวข้อที่ถูก flatten จากแผนผังโครงสร้างธุรกิจ (ไม่มีประโยค) อันดับ 12; top-5 ไม่มีรายการธุรกิจครบ"),
    62: ("b", "ตรงตัว: หน้า 198 (อันดับ 2) \"ยอดขาย 3,090,453 2,662,145 (13.9)\"; truth ระบุกราฟหน้า 10 (chunk ถูก flatten อันดับ 14)"),
    63: ("b", "ตรงตัว: หน้า 198 (อันดับ 1) \"EBITDA 396,234 332,849 (16.0)\"; truth ระบุกราฟหน้า 10 (chunk ถูก flatten อันดับ 6)"),
    69: ("b", "อ่านเนื้อ: หน้า 338, 177, 137 (อันดับ 1-3) พูดเรื่องพัฒนาสังคม/ชุมชนควบคู่การดูดซับก๊าซเรือนกระจก ตอบคำถามเดียวกัน; truth ระบุคำนำหน้า 6 (อันดับ 28)"),
    70: ("e", "chunk ที่ถูก (หน้า 7 chunk 0) เป็นข้อความที่ถูก flatten จากแผนผังกลยุทธ์ (หลายกล่องปนกัน) ไม่ติด top-50; หน้าที่ติดอันดับ (95, 128, 244) เป็นเรื่อง Venture Building / ธุรกิจใหม่ที่ไม่มี \"Sustainability และ AI Tech\""),
}


def short(text: str, n: int) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[:n] + "…"


def rank_at(cit: str, texts_canon: list[str], threshold: int, limit: int) -> int | None:
    for rank, t in enumerate(texts_canon[:limit], start=1):
        if fuzz.partial_ratio(cit, t) >= threshold:
            return rank
    return None


def metrics_at(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {"n": n, "recall_at_1": sum(r == 1 for r in ranks) / n if n else None,
            "recall_at_5": sum(r is not None and r <= 5 for r in ranks) / n if n else None,
            "mrr": sum(1 / r for r in ranks if r is not None and r <= 10) / n if n else None}


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if argv and "--render-only" in argv:  # ใช้ผลที่เก็บไว้ + MANUAL_CAUSES ปัจจุบัน (ไม่ index ใหม่)
        out = json.loads((RESULTS / "retrieval_miss_analysis.json").read_text(encoding="utf-8"))
        for m in out["misses"]:
            m["cause"], m["note"] = MANUAL_CAUSES.get(m["question_id"], (m["auto_cause"], ""))
        out["sensitivity"] = {int(k): v for k, v in out["sensitivity"].items()}
        (RESULTS / "retrieval_miss_analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        (RESULTS / "retrieval_miss_analysis.md").write_text(render_md(out), encoding="utf-8")
        return 0
    base = get_settings()
    questions = load_questions()
    documents = {short_name: extract_file(DATASETS / name, file_id=short_name) for short_name, name in DOCS.items()}

    unit_canon = {s: [canon(u.text) for u in d.units if u.status == UnitStatus.OK and u.text.strip()] for s, d in documents.items()}
    chunk_texts, chunk_labels = {}, {}
    for s, d in documents.items():
        ok = [u for u in d.units if u.status == UnitStatus.OK and u.text.strip()]
        made = make_document_chunks(ok, SIZE)
        chunk_texts[s], chunk_labels[s] = [c.text for c in made], [f"{c.label} · chunk {c.chunk_index}" for c in made]
    chunk_canon = {s: [canon(t) for t in ts] for s, ts in chunk_texts.items()}

    acl = AclStore()
    acl.create_box(Box(box_id=BOX_ID, visibility=Visibility.PUBLIC), owner_user_id="dev")
    for s in DOCS:
        acl.register_file(FileAcl(file_id=s, box_id=BOX_ID))
    principal = acl.principal_for("dev")

    settings = base.model_copy(update={"collection_public": "rag_miss_analysis_public", "collection_private": "rag_miss_analysis_private",
                                       "chunk_target_chars": SIZE})
    embedder = get_embedder(settings)
    store = VectorStore.connect(acl, settings)
    for name in store.collections:
        if store.client.collection_exists(name):
            store.client.delete_collection(name)
    store.ensure_collections()
    rows = []
    try:
        t0 = time.perf_counter()
        for s, d in documents.items():
            ok = [u for u in d.units if u.status == UnitStatus.OK and u.text.strip()]
            store.upsert_chunks(d, embed_chunks(make_document_chunks(ok, SIZE), embedder))
        print(f"index {time.perf_counter() - t0:.1f}s · chunks {sum(len(v) for v in chunk_texts.values())}", flush=True)
        for q in questions:
            doc_key = next(k for k, v in DOCS.items() if v == q["document_source"])
            cit = canon(q["expected_citation"])
            hits = store.search(embed_query(q["query"], embedder), principal, BOX_ID, limit=DEEP, mode=MODE).hits
            hit_canon = [canon(h.text) for h in hits]
            row = {
                "question_id": q["question_id"], "doc": doc_key, "query": q["query"], "expected_citation": q["expected_citation"],
                "expected_answer": q["expected_answer"], "top1": hits[0].text if hits else "", "top1_label": hits[0].label if hits else "",
                "unit_score": max((fuzz.partial_ratio(cit, t) for t in unit_canon[doc_key]), default=0),
                "chunk_score": max((fuzz.partial_ratio(cit, t) for t in chunk_canon[doc_key]), default=0),
                "ranks": {str(T): rank_at(cit, hit_canon, T, DEEP) for T in THRESHOLDS},
                "top5_has_answer": False, "answer_chunks_in_top5": [],
            }
            ans = canon(q["expected_answer"])
            for i, (h, hc) in enumerate(zip(hits[:5], hit_canon[:5]), start=1):
                if ans and (ans in hc or fuzz.partial_ratio(ans, hc) >= 90):
                    row["top5_has_answer"] = True
                    row["answer_chunks_in_top5"].append({"rank": i, "label": h.label, "is_citation_chunk": fuzz.partial_ratio(cit, hc) >= MAIN_THRESHOLD})
            scores = [fuzz.partial_ratio(cit, t) for t in chunk_canon[doc_key]]
            if scores:
                j = max(range(len(scores)), key=scores.__getitem__)
                row["best_chunk"] = {"label": chunk_labels[doc_key][j], "score": scores[j], "text": chunk_texts[doc_key][j]}
            row["top5"] = [{"rank": i, "label": h.label, "unit": h.unit_index, "text": h.text,
                            "citation_score": fuzz.partial_ratio(cit, hc), "has_answer": bool(ans and (ans in hc or fuzz.partial_ratio(ans, hc) >= 90))}
                           for i, (h, hc) in enumerate(zip(hits[:5], hit_canon[:5]), start=1)]
            found = row["ranks"][str(MAIN_THRESHOLD)]
            row["hit_chunk"] = ({"rank": found, "label": hits[found - 1].label, "text": hits[found - 1].text} if found else None)
            rows.append(row)
        qdrant = qdrant_version(base.qdrant_url)
    finally:
        for name in store.collections:
            if store.client.collection_exists(name):
                store.client.delete_collection(name)
        embedder.release()

    # ---- ความไวต่อเกณฑ์
    sensitivity = {}
    for T in THRESHOLDS:
        reach = [r for r in rows if r["unit_score"] >= T]
        sensitivity[T] = {"all": metrics_at([r["ranks"][str(T)] for r in rows]),
                          "reachable": metrics_at([r["ranks"][str(T)] for r in reach]), "reachable_n": len(reach)}

    # ---- ตารางข้อที่พลาด (เกณฑ์หลัก 80)
    misses = []
    for r in rows:
        if r["unit_score"] < MAIN_THRESHOLD:
            continue
        rank = r["ranks"][str(MAIN_THRESHOLD)]
        if rank is not None and rank <= 5:
            continue
        auto = "c" if r["chunk_score"] < MAIN_THRESHOLD else ("b" if r["top5_has_answer"] else "d")
        cause, note = MANUAL_CAUSES.get(r["question_id"], (auto, ""))
        misses.append({**r, "rank": rank, "auto_cause": auto, "cause": cause, "note": note})

    out = {"size": SIZE, "mode": MODE, "qdrant": qdrant, "questions": len(rows), "thresholds": list(THRESHOLDS),
           "sensitivity": sensitivity, "misses": misses,
           "rows": [{k: v for k, v in r.items() if k not in ("top5", "top1", "best_chunk", "hit_chunk")} for r in rows]}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "retrieval_miss_analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS / "retrieval_miss_analysis.md").write_text(render_md(out), encoding="utf-8")
    print(render_md(out))
    return 0


def render_md(out: dict) -> str:
    L = [f"# วิเคราะห์ข้อที่พลาด — CHUNK_TARGET_CHARS={out['size']} + {out['mode']}", "",
         f"- ชุด dev {out['questions']} ข้อ · Qdrant {out['qdrant']} · ดึง top-{DEEP} ต่อคำถาม · วิเคราะห์อย่างเดียว ไม่ได้แก้ระบบค้น",
         f"- ข้อที่พลาด = หลักฐานอยู่ใน unit ok (partial_ratio ≥ {MAIN_THRESHOLD} บน canon) แต่ chunk ที่ถูกไม่ติด top-5", "",
         "## 1. Recall@5 ไวต่อเกณฑ์ fuzzy แค่ไหน", "",
         "| เกณฑ์ | Recall@5 ทุกข้อ (70) | ข้อที่ค้นเจอได้ | Recall@5 เฉพาะที่ค้นเจอได้ | Recall@1 (ค้นเจอได้) | MRR (ค้นเจอได้) |", "|---:|---:|---:|---:|---:|---:|"]
    for T, s in out["sensitivity"].items():
        L.append(f"| {T} | {pct(s['all']['recall_at_5'])} | {s['reachable_n']} | {pct(s['reachable']['recall_at_5'])} "
                 f"| {pct(s['reachable']['recall_at_1'])} | {s['reachable']['mrr']:.3f} |")
    L += ["", "ข้อที่ติด/ไม่ติด top-5 ต่างกันเมื่อเปลี่ยนเกณฑ์:", ""]
    for lo, hi in ((70, 80), (80, 90)):
        flips = [r for r in out["rows"] if ((r["ranks"][str(lo)] or 99) <= 5) != ((r["ranks"][str(hi)] or 99) <= 5)]
        L.append(f"- {lo} → {hi}: " + ("; ".join(f"ข้อ {r['question_id']} (อันดับ {r['ranks'][str(lo)] or '-'} → {r['ranks'][str(hi)] or '-'})" for r in flips) or "ไม่มี"))
    L += ["", f"## 2. ข้อที่พลาด ({len(out['misses'])} ข้อ, เกณฑ์ {MAIN_THRESHOLD})", "",
          "| # | query | expected_citation | อันดับที่เจอจริง | chunk อันดับ 1 | สาเหตุ |", "|---:|---|---|---:|---|---|"]
    for m in out["misses"]:
        why = f"{m['cause']}) {m['note']}" if m["note"] else f"{m['cause']}) {CAUSES[m['cause']]}"
        L.append(f"| {m['question_id']} ({m['doc']}) | {short(m['query'], 60)} | {short(m['expected_citation'], 60)} | {m['rank'] or '-'} "
                 f"| {short(m['top1'], 80)} | {why} |")
    counts = {c: sum(1 for m in out["misses"] if m["cause"] == c) for c in CAUSES}
    n = len(out["misses"]) or 1
    L += ["", "## 3. สัดส่วนตามสาเหตุ", "", "| สาเหตุ | ข้อ | สัดส่วนของข้อที่พลาด |", "|---|---:|---:|"]
    L += [f"| {c}) {CAUSES[c]} | {counts[c]} | {100 * counts[c] / n:.0f}% |" for c in CAUSES]
    reach = out["sensitivity"][MAIN_THRESHOLD]["reachable_n"]
    strict = reach - len(out["misses"])
    literal = [m for m in out["misses"] if m["question_id"] in ANSWER_LITERAL_IN_TOP5]
    read = [m for m in out["misses"] if m["question_id"] in ANSWER_BY_READING_IN_TOP5]
    rest = [m for m in out["misses"] if m["question_id"] not in ANSWER_LITERAL_IN_TOP5 | ANSWER_BY_READING_IN_TOP5]
    ids = lambda ms: ", ".join(str(m["question_id"]) for m in ms)
    L += ["", "## 4. ถ้านับว่า \"ตอบได้\" เมื่อคำตอบอยู่ใน top-5 (ไม่ต้องเป็นหน้าที่ truth ระบุ)", "",
          f"| นับแบบ | ข้อ | จาก {reach} |", "|---|---:|---:|",
          f"| เข้มงวด (chunk ที่ตรง citation ≥ {MAIN_THRESHOLD} ติด top-5 — ตัวเลข Recall@5 ข้างบน) | {strict} | {100 * strict / reach:.0f}% |",
          f"| + คำตอบตรงตัวอยู่ใน top-5 หรือ chunk ที่ถูกอยู่ใน top-5 แต่ fuzzy ไม่ถึงเกณฑ์ ({len(literal)} ข้อ: {ids(literal)}) "
          f"| {strict + len(literal)} | {100 * (strict + len(literal)) / reach:.0f}% |",
          f"| + คำตอบเทียบความหมายจากการอ่านอยู่ใน top-5 ({len(read)} ข้อ: {ids(read)}) "
          f"| {strict + len(literal) + len(read)} | {100 * (strict + len(literal) + len(read)) / reach:.0f}% |", "",
          f"ข้อที่เหลือ (ตอบไม่ได้แม้ผ่อนเกณฑ์ {len(rest)} ข้อ): {ids(rest)}", "",
          "> สองบรรทัดหลังเป็นการตัดสินด้วยคนอ่าน ไม่ใช่เกณฑ์อัตโนมัติ ใช้ดูว่าตัวเลข Recall@5 ต่ำกว่าความจริงแค่ไหน — ไม่ใช่ตัวเลขที่ใช้เทียบ config"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
