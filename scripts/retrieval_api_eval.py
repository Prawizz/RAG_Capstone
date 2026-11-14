"""วัด retrieval ชุด dev ผ่าน ``POST /v1/retrieve`` ของ API ที่รันอยู่ (กล่อง annual-reports, top_k 10)

    python scripts/bootstrap_demo.py                                   # ตั้งกล่อง + ingest ก่อน
    python scripts/retrieval_api_eval.py --api-log path/to/api.log     # log ของ API (ไม่บังคับ) ใช้แยกเวลา embed/search

- นิยาม hit และตัวชี้วัด **import จาก scripts/retrieval_dev_eval.py ตรงๆ** (load_questions, canon, is_hit / partial_ratio ≥ 80,
  first_hit_rank, metrics = Recall@1 / @5 / MRR top 10) — ไม่ปรับพารามิเตอร์ใดๆ ให้ตัวเลขดีขึ้น
- รายงาน (ก) 50 ข้อที่ 6 พ.ย. ค้นเจอได้ (``coverage.*.in_ok_units_ids`` ใน eval/results/retrieval_dev_eval.json) เทียบกับผลเดิม
  ของ config ที่เลือกใช้ (CHUNK_TARGET_CHARS=1000, dense+sparse_bge) รายข้อ · (ข) 70 ข้อทั้งหมด (ตอนนี้ KBANK OCR แล้ว)
- latency ต่อ request (ฝั่ง client, p50/p95) + แยก embed/search/map จาก log ของ API (จับคู่ด้วย X-Trace-Id) ถ้าให้ ``--api-log``
- การกระจายของ top_score: type=question แยกตามผล (hit@1 / hit อันดับ 2–10 / ไม่ hit ใน 10) และ type อื่นใน rag_evaluation_100
  (unanswerable, chitchat) — เป็นข้อมูลไว้ตั้ง RETRIEVE_MIN_SCORE ภายหลัง **สคริปต์นี้ไม่เปลี่ยนค่าเกณฑ์**
- เรียกในนาม persona ``somchai.demo`` (viewer ของ annual-reports; กล่อง public ไม่จำกัดแผนก/ระดับ)
- ผลลง eval/results/<--name>.json + .md (ค่าเริ่มต้น retrieval_api_eval · ``--no-write`` = พิมพ์อย่างเดียว)
- API เปิด ``RERANKER`` -> log มี ``rerank_ms`` / ``rerank_top`` (คะแนน reranker ของอันดับ 1) -> รายงาน latency ของ rerank
  และการกระจายคะแนน reranker ต่อกลุ่ม (เหมือน top_score) — ใช้ ``--label`` บอก config ในรายงาน
- **ห้ามวัดโมดูลของเพื่อน (QUERY_EXPANSION / SALIENCE_RERANK) บนชุด dev** (คำย่อ/น้ำหนักถูกตั้งจากข้อที่พลาดในชุดนี้ = วัดวน):
  ค่าตั้งในเครื่องเปิดอยู่ หรือ log ของ API บอกว่าเปิด (``extras``) -> หยุด ไม่เขียนผล
- ``--questions eval/datasets/questions.jsonl`` = วัดชุดคำถามที่ไม่ได้ใช้ออกแบบ (ตรวจด้วย ``eval/questions.py`` ก่อน) แทนชุด dev:
  hit = ผลที่ filename ตรง ``file`` และ citation.page อยู่ใน ``pages`` · Recall@1/@5/MRR ด้วย ``metrics`` ชุดเดียวกัน
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app.config import build_settings, load_env_file  # noqa: E402
from eval.ocr_canon import canon  # noqa: E402
from scripts.retrieval_dev_eval import (  # noqa: E402 — นิยามเดียวกันทุกอย่าง
    DEV_SET,
    DOCS,
    FUZZY_THRESHOLD,
    LIMIT,
    first_hit_rank,
    load_questions,
    metrics,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
OLD_RESULTS = RESULTS / "retrieval_dev_eval.json"
BOX_ID = "annual-reports"
USER = ("somchai.demo", "marketing", 1)
OLD_CONFIG = (1000, "dense+sparse_bge")  # config ที่เลือกเมื่อ 6 พ.ย. = ค่าเริ่มต้นของ API ตอนนี้
SCORE_THRESHOLDS = (0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)  # ไว้ดูการกระจายเท่านั้น ไม่ใช่การตั้งเกณฑ์
LOG_RE = re.compile(r"retrieve trace_id=(\S+) .*?embed_ms=([\d.]+) search_ms=([\d.]+) map_ms=([\d.]+)")
RERANK_RE = re.compile(r"rerank_ms=([\d.]+) rerank_top=(\S+) extras=(\S+)")
FRIEND_EXTRAS = ("query_expansion", "salience_rerank")  # โมดูลของเพื่อน — ห้ามวัดบนชุด dev


def service_key() -> str:
    key = os.environ.get("SERVICE_KEY") or load_env_file(ROOT / ".env").get("SERVICE_KEY")
    if not key:
        raise SystemExit("ไม่มี SERVICE_KEY (env หรือ .env)")
    return key


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": min(values), "p10": quantile(values, 0.1), "p25": quantile(values, 0.25),
            "median": quantile(values, 0.5), "p75": quantile(values, 0.75), "p90": quantile(values, 0.9),
            "max": max(values), "mean": statistics.fmean(values),
            "share_at_or_above": {str(t): sum(1 for v in values if v >= t) / len(values) for t in SCORE_THRESHOLDS}}


def latency(values: list[float]) -> dict:
    return {"n": len(values), "p50": quantile(values, 0.5), "p95": quantile(values, 0.95), "max": max(values) if values else None,
            "mean": statistics.fmean(values) if values else None}


def parse_api_log(path: Path, trace_ids: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LOG_RE.search(line)
        if m and m.group(1) in trace_ids:
            row = {"embed_ms": float(m.group(2)), "search_ms": float(m.group(3)), "map_ms": float(m.group(4)),
                   "rerank_ms": None, "rerank_top": None, "extras": []}
            r = RERANK_RE.search(line)
            if r:
                row["rerank_ms"] = float(r.group(1))
                row["rerank_top"] = None if r.group(2) == "-" else float(r.group(2))
                row["extras"] = [] if r.group(3) == "-" else r.group(3).split(",")
            out[m.group(1)] = row
    return out


def refuse_friend_modules_on_dev(settings, parsed: dict[str, dict] | None) -> None:
    on = [x for x in FRIEND_EXTRAS if getattr(settings, x)]
    if parsed:
        on += sorted({x for p in parsed.values() for x in p["extras"] if x in FRIEND_EXTRAS})
    if on:
        raise SystemExit(f"ห้ามวัด {sorted(set(on))} บนชุด dev (ออกแบบจากข้อที่พลาดในชุดนี้) — ใช้ --questions "
                         "eval/datasets/questions.jsonl")


def score_table(rows: dict[str, dict]) -> list[str]:
    L = ["| กลุ่ม | n | min | p10 | p25 | median | p75 | p90 | max |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, d in rows.items():
        if d.get("n"):
            L.append(f"| {name} | {d['n']} | {d['min']:.3f} | {d['p10']:.3f} | {d['p25']:.3f} | {d['median']:.3f} | "
                     f"{d['p75']:.3f} | {d['p90']:.3f} | {d['max']:.3f} |")
        else:
            L.append(f"| {name} | 0 | | | | | | | |")
    return L


def render_md(out: dict) -> str:
    old, new = out["reachable"]["old"], out["reachable"]["new"]
    L = ["# วัด retrieval ชุด dev ผ่าน POST /v1/retrieve" + (f" — {out['label']}" if out.get("label") else ""), "",
         f"- API {out['base_url']} · กล่อง `{BOX_ID}` (public: KBANK, CPALL, PTT) · top_k {LIMIT} · ผู้เรียก `{USER[0]}`",
         f"- hit = partial_ratio ≥ {FUZZY_THRESHOLD} บน canon (นิยามจาก scripts/retrieval_dev_eval.py ตรงๆ) · Recall@1 / @5 · MRR (top {LIMIT})",
         f"- ค่าตั้งของ API: SEARCH_MODE={out['search_mode']} · CHUNK_TARGET_CHARS={out['chunk_target_chars']} · "
         f"RETRIEVE_MIN_SCORE={out['retrieve_min_score']} (ไม่ได้เปลี่ยน) · score = dense cosine",
         f"- วัดเมื่อ {out['measured_at']} · คำถาม {out['n_requests']} request (warm-up 1 ครั้งไม่นับ)", "",
         "**ต่างจากการวัด 6 พ.ย.:** (1) index มีหน้าที่ OCR แล้ว (KBANK ทั้งเล่ม, CPALL 156 หน้า, PTT 4 หน้า) → chunk มากขึ้น "
         "คู่แข่งในการค้นมากขึ้น (2) คำถามผ่าน normalizer กลางก่อน embed (3) embed คำถามบน CPU fp32 (เดิม GPU bf16; ฝั่งเอกสารยัง bf16) "
         "(4) ค้นผ่าน API + ACL 2 ด่าน", "",
         f"## (ก) {old['n']} ข้อที่ 6 พ.ย. ค้นเจอได้ (หลักฐานอยู่ใน unit ok ตอนนั้น) — เทียบผลเดิม", "",
         "| | Recall@1 | Recall@5 | MRR |", "|---|---:|---:|---:|",
         f"| 6 พ.ย. ({OLD_CONFIG[0]}, {OLD_CONFIG[1]}, ค้นตรง ไม่ OCR) | {pct(old['recall_at_1'])} | {pct(old['recall_at_5'])} | {old['mrr']:.3f} |",
         f"| ตอนนี้ ผ่าน /v1/retrieve | {pct(new['recall_at_1'])} | {pct(new['recall_at_5'])} | {new['mrr']:.3f} |", "",
         f"รายข้อ (อันดับแรกที่ hit): ดีขึ้น {out['per_question']['better']} · แย่ลง {out['per_question']['worse']} · "
         f"เท่าเดิม {out['per_question']['same']}", ""]
    if out["per_question"]["changed"]:
        L += ["| question_id | ไฟล์ | อันดับเดิม | อันดับใหม่ |", "|---|---|---:|---:|"]
        for c in out["per_question"]["changed"]:
            L.append(f"| {c['question_id']} | {c['doc']} | {c['old'] or '-'} | {c['new'] or '-'} |")
        L.append("")
    L += [f"## (ข) {out['all']['n']} ข้อทั้งหมด (type=question)", "",
          "| ชุด | n | Recall@1 | Recall@5 | MRR |", "|---|---:|---:|---:|---:|",
          f"| 6 พ.ย. ทั้งหมด (KBANK ยังไม่ OCR = 0) | {out['all_old']['n']} | {pct(out['all_old']['recall_at_1'])} | "
          f"{pct(out['all_old']['recall_at_5'])} | {out['all_old']['mrr']:.3f} |",
          f"| ตอนนี้ ทั้งหมด | {out['all']['n']} | {pct(out['all']['recall_at_1'])} | {pct(out['all']['recall_at_5'])} | "
          f"{out['all']['mrr']:.3f} |"]
    for d, m in out["by_doc"].items():
        o = out["by_doc_old"][d]
        L.append(f"| {d} (6 พ.ย. → ตอนนี้) | {m['n']} | {pct(o['recall_at_1'])} → {pct(m['recall_at_1'])} | "
                 f"{pct(o['recall_at_5'])} → {pct(m['recall_at_5'])} | {o['mrr']:.3f} → {m['mrr']:.3f} |")
    lat = out["latency"]
    L += ["", "## latency ต่อ request (ms)", "",
          "| ช่วง | n | p50 | p95 | max | เฉลี่ย |", "|---|---:|---:|---:|---:|---:|"]
    for name, v in lat.items():
        if v and v.get("n"):
            L.append(f"| {name} | {v['n']} | {v['p50']:.0f} | {v['p95']:.0f} | {v['max']:.0f} | {v['mean']:.0f} |")
    L += ["", "client = เวลารอบ HTTP ทั้งหมด (เครื่องเดียวกับ API) · embed / search / map จาก log ของ API "
              "(search = Qdrant ค้น + ACL ด่าน 2, map = แปลงผล + คำนวณ cosine)", "",
          "## การกระจายของ top_score (ข้อมูลไว้ตั้ง RETRIEVE_MIN_SCORE ภายหลัง — ไม่ได้เปลี่ยนค่าเกณฑ์)", ""]
    L += score_table(out["top_score"])
    if out.get("rerank_score"):
        L += ["", "## การกระจายคะแนน reranker ของอันดับ 1 (rerank_top จาก log — ไม่ได้ส่งออก API)", ""]
        L += score_table(out["rerank_score"])
    L += ["", "สัดส่วนของแต่ละกลุ่มที่ top_score ≥ ค่า t (ถ้าตั้งเกณฑ์ที่ t กลุ่มนี้จะได้ status ok เท่านี้)", "",
          "| กลุ่ม | " + " | ".join(f"≥{t}" for t in SCORE_THRESHOLDS) + " |", "|---|" + "---:|" * len(SCORE_THRESHOLDS)]
    for name, d in out["top_score"].items():
        if d.get("n"):
            L.append(f"| {name} | " + " | ".join(pct(d["share_at_or_above"][str(t)]) for t in SCORE_THRESHOLDS) + " |")
    L += ["", "กลุ่ม: hit@1 / hit อันดับ 2–10 / ไม่ hit ใน 10 = type=question แยกตามอันดับแรกที่ hit · unanswerable / chitchat = "
              "คำถามที่ไม่มีคำตอบในเอกสาร / คุยเล่น (ไม่มี expected_citation)", "",
          f"status ที่ได้จริง (เกณฑ์ปัจจุบัน {out['retrieve_min_score']}): " +
          " · ".join(f"{t}: {', '.join(f'{k}={v}' for k, v in c.items())}" for t, c in out["status_by_type"].items())]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="วัดชุด dev ผ่าน /v1/retrieve")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-log", type=Path, help="log ของ API (แยกเวลา embed/search/map ด้วย trace_id)")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--render-only", action="store_true", help="สร้าง .md ใหม่จาก .json เดิม")
    ap.add_argument("--name", default="retrieval_api_eval", help="ชื่อไฟล์ผลใน eval/results (ไม่มีนามสกุล)")
    ap.add_argument("--label", default="", help="บอก config ของ API ในรายงาน เช่น 'RERANKER=bge-v2-m3 (cpu)'")
    ap.add_argument("--questions", type=Path, help="วัดชุดนี้ (schema eval/datasets/questions.schema.json) แทนชุด dev")
    ap.add_argument("--box-id", default=BOX_ID, help="กล่องที่ค้น (ใช้กับ --questions)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.render_only:
        out = json.loads((RESULTS / f"{args.name}.json").read_text(encoding="utf-8"))
        md = render_md(out)
        if not args.no_write:
            (RESULTS / f"{args.name}.md").write_text(md, encoding="utf-8")
        else:
            print(md)
        return 0
    if args.questions:
        return run_questions(args)

    settings = build_settings()
    refuse_friend_modules_on_dev(settings, None)
    old = json.loads(OLD_RESULTS.read_text(encoding="utf-8"))
    old_row = next(r for r in old["rows"] if (r["size"], r["mode"]) == OLD_CONFIG)
    reachable = {str(qid) for c in old["coverage"].values() for qid in c["in_ok_units_ids"]}

    rows = [json.loads(line) for line in DEV_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = load_questions()
    qids = {str(q["question_id"]) for q in questions}
    others = [r for r in rows if str(r["question_id"]) not in qids]

    key = service_key()
    headers = {"X-Service-Key": key, "X-User-Id": USER[0], "X-User-Departments": USER[1],
               "X-User-Security-Level": str(USER[2])}
    http = httpx.Client(base_url=args.base_url.rstrip("/"), timeout=60.0)

    def call(query: str, trace: str) -> tuple[dict, float]:
        t0 = time.perf_counter()
        r = http.post("/v1/retrieve", headers={**headers, "X-Trace-Id": trace},
                      json={"query": query, "box_id": BOX_ID, "top_k": LIMIT})
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            raise SystemExit(f"{trace}: {r.status_code} {r.text}")
        return r.json(), ms

    call("ทดสอบระบบก่อนวัด", "eval-warmup")  # warm-up (ไม่นับ)
    stamp = time.strftime("%Y%m%d%H%M%S")
    responses: dict[str, dict] = {}
    client_ms: dict[str, float] = {}
    for r in rows:
        trace = f"eval-{stamp}-{r['question_id']}"
        body, ms = call(r["query"], trace)
        responses[str(r["question_id"])] = {"trace": trace, "body": body, "type": r["type"]}
        client_ms[trace] = ms
    print(f"เรียก /v1/retrieve {len(rows)} ครั้ง", flush=True)

    text_canon: dict[str, str] = {}
    ranks: dict[str, int | None] = {}
    for q in questions:
        qid = str(q["question_id"])
        texts = [x["text"] for x in responses[qid]["body"]["results"]]
        ranks[qid] = first_hit_rank(canon(q["expected_citation"]), texts, text_canon)

    old_ranks = {str(k): v for k, v in old_row["ranks"].items()}
    changed, better, worse, same = [], 0, 0, 0
    for q in questions:
        qid = str(q["question_id"])
        if qid not in reachable:
            continue
        o, n = old_ranks.get(qid), ranks[qid]
        ov, nv = (o or 10**6), (n or 10**6)
        if nv < ov:
            better += 1
        elif nv > ov:
            worse += 1
        else:
            same += 1
        if o != n:
            doc = next(d for d, f in DOCS.items() if f == q["document_source"])
            changed.append({"question_id": qid, "doc": doc, "old": o, "new": n})

    def top_scores(ids) -> list[float]:
        return [responses[i]["body"]["top_score"] for i in ids if responses[i]["body"]["top_score"] is not None]

    groups = {
        "question: hit@1": [i for i, r in ranks.items() if r == 1],
        "question: hit อันดับ 2–10": [i for i, r in ranks.items() if r is not None and r > 1],
        "question: ไม่ hit ใน 10": [i for i, r in ranks.items() if r is None],
    }
    for t in sorted({r["type"] for r in others}):
        groups[t] = [str(r["question_id"]) for r in others if r["type"] == t]
    status_by_type: dict[str, dict[str, int]] = {}
    for qid, v in responses.items():
        c = status_by_type.setdefault(v["type"], {})
        c[v["body"]["status"]] = c.get(v["body"]["status"], 0) + 1

    lat = {"client": latency(list(client_ms.values()))}
    rerank_tops: dict[str, float] = {}
    if args.api_log:
        parsed = parse_api_log(args.api_log, set(client_ms))
        refuse_friend_modules_on_dev(settings, parsed)
        lat.update(server_latency(parsed))
        by_trace = {v["trace"]: qid for qid, v in responses.items()}
        rerank_tops = {by_trace[t]: p["rerank_top"] for t, p in parsed.items() if p["rerank_top"] is not None}
        print(f"จับคู่ log ได้ {len(parsed)}/{len(client_ms)} request")

    out = {
        "base_url": args.base_url, "measured_at": time.strftime("%Y-%m-%d %H:%M"), "n_requests": len(rows),
        "search_mode": settings.search_mode, "chunk_target_chars": settings.chunk_target_chars,
        "retrieve_min_score": settings.retrieve_min_score, "fuzzy_threshold": FUZZY_THRESHOLD, "limit": LIMIT,
        "reachable": {"old": old_row["reachable"], "new": metrics([ranks[i] for i in sorted(reachable)])},
        "all": metrics(list(ranks.values())), "all_old": old_row["all"],
        "by_doc": {d: metrics([ranks[str(q["question_id"])] for q in questions if q["document_source"] == f])
                   for d, f in DOCS.items()},
        "by_doc_old": old_row["by_doc"],
        "per_question": {"better": better, "worse": worse, "same": same, "changed": changed},
        "latency": lat,
        "label": args.label,
        "top_score": {name: describe(top_scores(ids)) for name, ids in groups.items()},
        "rerank_score": ({name: describe([rerank_tops[i] for i in ids if i in rerank_tops]) for name, ids in groups.items()}
                         if rerank_tops else None),
        "rerank_top": rerank_tops or None,
        "status_by_type": status_by_type,
        "ranks": ranks,
        "responses": {qid: {"type": v["type"], "trace_id": v["trace"], "status": v["body"]["status"],
                            "top_score": v["body"]["top_score"],
                            "results": [{"chunk_id": x["chunk_id"], "score": x["score"], "label": x["citation"]["label"],
                                         "filename": x["filename"]} for x in v["body"]["results"]]}
                      for qid, v in responses.items()},
    }
    md = render_md(out)
    print("\n" + md)
    if not args.no_write:
        (RESULTS / f"{args.name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        (RESULTS / f"{args.name}.md").write_text(md, encoding="utf-8")
        print(f"เขียน {RESULTS / (args.name + '.json')} และ .md")
    return 0


def server_latency(parsed: dict[str, dict]) -> dict:
    lat = {}
    for k in ("embed_ms", "search_ms", "rerank_ms", "map_ms"):
        vals = [p[k] for p in parsed.values() if p[k] is not None]
        if vals and (k != "rerank_ms" or any(v > 0.05 for v in vals)):
            lat[k.removesuffix("_ms")] = latency(vals)
    lat["server_total"] = latency([sum(p[k] or 0.0 for k in ("embed_ms", "search_ms", "rerank_ms", "map_ms"))
                                   for p in parsed.values()])
    return lat


def page_hit_rank(results: list[dict], file: str, pages: list[int]) -> int | None:
    """อันดับแรกที่ filename ตรง และ citation.page อยู่ใน pages (unit ที่ไม่ใช่หน้า = ไม่ hit)"""
    for rank, r in enumerate(results, start=1):
        if r["filename"] == file and r["citation"].get("page") in pages:
            return rank
    return None


def run_questions(args) -> int:
    """ชุดคำถามที่ไม่ได้ใช้ออกแบบระบบ (``eval/datasets/questions.jsonl``) — ใช้วัดโมดูลของเพื่อนได้"""
    from eval.questions import QuestionsError, load

    try:
        rows = load(args.questions)
    except QuestionsError as e:
        print("\n".join(e.errors))
        return 2
    key = service_key()
    headers = {"X-Service-Key": key, "X-User-Id": USER[0], "X-User-Departments": USER[1],
               "X-User-Security-Level": str(USER[2])}
    http = httpx.Client(base_url=args.base_url.rstrip("/"), timeout=120.0)
    stamp = time.strftime("%Y%m%d%H%M%S")
    http.post("/v1/retrieve", headers={**headers, "X-Trace-Id": "eval-warmup"},
              json={"query": "ทดสอบระบบก่อนวัด", "box_id": args.box_id, "top_k": LIMIT})
    responses, client_ms = {}, {}
    for row in rows:
        trace = f"evalq-{stamp}-{row['id']}"
        t0 = time.perf_counter()
        r = http.post("/v1/retrieve", headers={**headers, "X-Trace-Id": trace},
                      json={"query": row["question"], "box_id": args.box_id, "top_k": LIMIT})
        client_ms[trace] = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            raise SystemExit(f"{trace}: {r.status_code} {r.text}")
        responses[row["id"]] = {"trace": trace, "body": r.json(), "row": row}
    ranks = {qid: page_hit_rank(v["body"]["results"], v["row"]["file"], v["row"]["pages"])
             for qid, v in responses.items() if v["row"]["type"] == "question"}
    lat = {"client": latency(list(client_ms.values()))}
    if args.api_log:
        lat.update(server_latency(parse_api_log(args.api_log, set(client_ms))))
    groups = {"question: hit@1": [i for i, r in ranks.items() if r == 1],
              "question: hit อันดับ 2–10": [i for i, r in ranks.items() if r is not None and r > 1],
              "question: ไม่ hit ใน 10": [i for i, r in ranks.items() if r is None]}
    for t in ("unanswerable", "chitchat"):
        groups[t] = [qid for qid, v in responses.items() if v["row"]["type"] == t]
    out = {"base_url": args.base_url, "measured_at": time.strftime("%Y-%m-%d %H:%M"), "questions": str(args.questions),
           "label": args.label, "box_id": args.box_id, "limit": LIMIT, "all": metrics(list(ranks.values())),
           "latency": lat, "ranks": ranks,
           "top_score": {name: describe([responses[i]["body"]["top_score"] for i in ids
                                          if responses[i]["body"]["top_score"] is not None])
                         for name, ids in groups.items()}}
    m = out["all"]
    md = "\n".join([f"# วัด retrieval ชุด {args.questions.name}" + (f" — {args.label}" if args.label else ""), "",
                    f"- {len(rows)} ข้อ (question {len(ranks)}) · กล่อง `{args.box_id}` · top_k {LIMIT} · "
                    "hit = filename ตรง + หน้าอยู่ใน pages", "",
                    "| n | Recall@1 | Recall@5 | MRR |", "|---:|---:|---:|---:|",
                    f"| {m['n']} | {pct(m['recall_at_1'])} | {pct(m['recall_at_5'])} | {m['mrr']:.3f} |", "",
                    "## top_score", "", *score_table(out["top_score"])]) + "\n"
    print(md)
    if not args.no_write:
        (RESULTS / f"{args.name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        (RESULTS / f"{args.name}.md").write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
