"""Evaluation รอบจริงบน ``eval/datasets/questions.jsonl`` ผ่าน ``POST /v1/retrieve`` (กล่อง annual-reports, top_k 10)

ขั้นตอน (API ต้องรันด้วย env ของ config นั้นก่อน — รีสตาร์ต API ทุกครั้งที่เปลี่ยน config):

    python scripts/questions_eval.py run --config baseline --run-id baseline_r1 --api-log api.log
    python scripts/questions_eval.py run --config baseline --run-id baseline_r2 --api-log api.log   # process เดิม (ดูการแกว่ง)
    python scripts/questions_eval.py run --config reranker --api-log api.log          # RERANKER=bge-v2-m3 RERANKER_DEVICE=cuda
    python scripts/questions_eval.py run --config salience_rerank --api-log api.log   # SALIENCE_RERANK=1
    (ผล query_expansion ของ 23 พ.ย. ยังอยู่ใน report — QUERY_EXPANSION ถูกตัดออกจาก API แล้ว จึงรันใหม่ไม่ได้)
    python scripts/questions_eval.py report     # -> eval/results/questions_eval.json + .md (A + B)
    python scripts/questions_eval.py review     # -> หน้าตรวจ eval/datasets/files/questions_review/index.html (C)
    python scripts/questions_eval.py lenient --judgments judgments.json   # -> eval/results/questions_lenient.json + .md

status ที่เกณฑ์ค่าเริ่มต้นของ API (RETRIEVE_MIN_SCORE / RERANK_MIN_SCORE จาก ``app.config.Settings()``):

    python scripts/questions_eval.py run --config baseline --run-id status_baseline --api-log api.log
    python scripts/questions_eval.py run --config reranker --run-id status_reranker --api-log api.log
    python scripts/questions_eval.py status     # -> eval/results/questions_status.json + .md

นิยาม (ไม่ปรับให้ตัวเลขดีขึ้น — ไม่เปลี่ยนค่าตั้งใดๆ ของ API, ไม่เพิ่มฟิลด์ใน API)
- hit = ผลที่ filename ตรง ``file`` และ citation.page อยู่ใน ``pages`` (``scripts/retrieval_api_eval.py::page_hit_rank``)
  · R@1 / R@5 / MRR@10 = ``scripts/retrieval_dev_eval.py::metrics`` (top_k 10 = ``LIMIT``) · เฉพาะ type=question
- ``run`` ตรวจ ``extras`` ใน log ของ API ทุก request ว่าตรง config ที่บอก ไม่ตรง / จับคู่ log ไม่ครบ = หยุด ไม่เขียนผล
  ผลดิบ (response เต็ม รวมข้อความ chunk) ลง ``eval/datasets/files/questions_runs/<run_id>.json`` (ไม่เข้า git — ใหญ่)
  ส่วน ``report`` เขียนผลย่อ (อันดับ + top 10 แบบ file/page/score ไม่มีข้อความ) ลง eval/results ให้คำนวณซ้ำได้
- เทียบรายข้อกับ baseline (``baseline_r1``) ด้วยอันดับแรกที่ hit (ไม่ hit ใน 10 = แย่สุด): ดีขึ้น / แย่ลง / เท่าเดิม
  · ``baseline_r2`` = รันซ้ำใน process เดียวกัน วัดการแกว่ง (อันดับ / top 10 / top_score ต่างกันกี่ข้อ)
- B (การปฏิเสธ): top_score (dense cosine) ของ baseline_r1 + คะแนน reranker อันดับ 1 (``rerank_top`` จาก log ของรัน reranker)
  แยก 3 type · AUC (question = บวก, คะแนนสูง = มีคำตอบ; เสมอ = ครึ่ง) · เกณฑ์แบบไม่วัดวน: แบ่งข้อ stratified ตาม type
  (``SPLIT_SEED`` คงที่) เป็นครึ่ง calibrate / ครึ่ง test -> เลือกเกณฑ์ t จากครึ่ง calibrate = ค่าสูงสุดที่คำถามจริงถูกปฏิเสธ
  (คะแนน < t แบบเดียวกับ RETRIEVE_MIN_SCORE) ไม่เกินงบ (5% เป็นหลัก) -> รายงานผลบนครึ่ง test · **รายงานอย่างเดียว ไม่เปลี่ยนเกณฑ์**
- C (ให้คนตรวจ — ระบบไม่ตัดสินเองว่ามีคำตอบ, กฎเหล็กข้อ 6): pool = ข้อ type=question ที่ baseline หรือ reranker พลาดใน top 5
  · ผู้สมัคร = ผลใน top 5 ของ baseline (ทั้ง 2 รอบ) + reranker ที่ (file, page) ไม่อยู่ใน truth รวมซ้ำระดับ (ข้อ, file, page)
  · ``lenient``: hit@5 แบบผ่อน = hit เดิม หรือ top 5 มี (file, page) ที่คนตัดสินว่า "มีคำตอบ" · **strict ยังเป็นตัวเลขหลัก**
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from scripts.retrieval_api_eval import (  # noqa: E402 — นิยามเดียวกัน
    USER,
    latency,
    page_hit_rank,
    parse_api_log,
    quantile,
    server_latency,
    service_key,
)
from scripts.retrieval_dev_eval import DOCS, LIMIT, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "datasets" / "questions.jsonl"
RESULTS = ROOT / "eval" / "results"
RUNS_DIR = ROOT / "eval" / "datasets" / "files" / "questions_runs"
REVIEW_DIR = ROOT / "eval" / "datasets" / "files" / "questions_review"
REVIEW_ITEMS = RESULTS / "questions_review_items.json"
TEMPLATE = Path(__file__).resolve().parent / "questions_review_template.html"
PDF_DIR = ROOT / "eval" / "datasets"
BOX_ID = "annual-reports"
WARMUP = 3

# config -> env ที่ต้องตั้งตอนเริ่ม API + extras ที่ log ต้องบอก (app/api/routes_retrieve.py::extras_of)
CONFIGS = {
    "baseline": {"env": {}, "extras": [], "label": "baseline (ค่าเริ่มต้น)"},
    "reranker": {"env": {"RERANKER": "bge-v2-m3", "RERANKER_DEVICE": "cuda"}, "extras": ["reranker:bge-v2-m3"],
                 "label": "RERANKER=bge-v2-m3 (cuda)"},
    "salience_rerank": {"env": {"SALIENCE_RERANK": "1"}, "extras": ["salience_rerank"], "label": "SALIENCE_RERANK=1"},
}
BASE_RUN = "baseline_r1"
REPEAT_RUN = "baseline_r2"
DEFAULT_RUNS = (BASE_RUN, REPEAT_RUN, "reranker", "query_expansion", "salience_rerank")
POOL_RUNS = (BASE_RUN, "reranker")  # นิยาม pool ของ C (ข้อที่ baseline หรือ reranker พลาดใน top 5)
CANDIDATE_RUNS = (BASE_RUN, REPEAT_RUN, "reranker")  # ผู้สมัครมาจาก top 5 ของรันเหล่านี้ (r2 ใส่ด้วยให้คิด lenient ได้ครบ)
TOP_N = 5
SPLIT_SEED = 20261123
BUDGETS = (0.05, 0.02, 0.10)  # สัดส่วนคำถามจริงที่ยอมให้ถูกปฏิเสธผิดบนครึ่ง calibrate (5% = หลัก)
SCORE_TYPES = ("question", "unanswerable", "chitchat")
TAGS = {"ต้อง OCR": "ต้อง OCR", "multi_hop": "multi_hop"}
FILE_SHORT = {f: d for d, f in DOCS.items()}


# ---------- คำนวณ (ฟังก์ชันล้วน มี unit test) ----------

def tag_of(row: dict) -> str:
    return TAGS.get((row.get("note") or "").strip(), "ทั่วไป")


def auc(pos: list[float], neg: list[float]) -> float | None:
    """P(คะแนนของบวก > ลบ) + 0.5·P(เสมอ) — Mann-Whitney"""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def stratified_split(rows: list[dict], seed: int = SPLIT_SEED) -> tuple[set[str], set[str]]:
    """แบ่ง id ต่อ type เป็นครึ่ง calibrate / test (เรียง id ก่อนสุ่ม ให้ผลไม่ขึ้นกับลำดับไฟล์) — ครึ่งแรก = floor(n/2)"""
    rng = random.Random(seed)
    cal, test = set(), set()
    for t in sorted({r["type"] for r in rows}):
        ids = sorted(str(r["id"]) for r in rows if r["type"] == t)
        rng.shuffle(ids)
        k = len(ids) // 2
        cal.update(ids[:k])
        test.update(ids[k:])
    return cal, test


def pick_threshold(question_scores: list[float], budget: float) -> float:
    """เกณฑ์ t สูงสุดที่คำถามจริงถูกปฏิเสธ (คะแนน < t) ไม่เกิน floor(budget·n) ข้อ = คะแนนลำดับที่ k+1 จากต่ำสุด"""
    s = sorted(question_scores)
    k = math.floor(budget * len(s))
    return s[min(k, len(s) - 1)]


def reject_rate(scores: list[float], t: float) -> tuple[int, int]:
    return sum(1 for x in scores if x < t), len(scores)


def first_rank_compare(base: dict[str, int | None], other: dict[str, int | None]) -> dict:
    better, worse, same, changed = 0, 0, 0, []
    for qid in base:
        o, n = base[qid], other[qid]
        ov, nv = (o or 10 ** 6), (n or 10 ** 6)
        if nv < ov:
            better += 1
        elif nv > ov:
            worse += 1
        else:
            same += 1
        if o != n:
            changed.append({"id": qid, "baseline": o, "rank": n})
    at5 = {"gained": sorted(q for q in base if not hit5(base[q]) and hit5(other[q])),
           "lost": sorted(q for q in base if hit5(base[q]) and not hit5(other[q]))}
    return {"better": better, "worse": worse, "same": same, "changed": changed, "at5": at5}


def hit5(rank: int | None) -> bool:
    return rank is not None and rank <= TOP_N


def page_of(res: dict) -> int | None:
    return res["citation"].get("page")


def build_candidates(rows: list[dict], runs: dict[str, dict]) -> tuple[list[str], list[dict]]:
    """pool (ข้อที่ baseline หรือ reranker พลาดใน top 5) + ผู้สมัคร (file, page) ที่ไม่อยู่ใน truth จาก top 5 ของ CANDIDATE_RUNS

    รายการละ (ข้อ, file, page) · ``chunks`` = ทุก chunk ของหน้านั้นที่ขึ้นใน top 5 (บอกว่ารันไหนอันดับเท่าไร)"""
    qrows = [r for r in rows if r["type"] == "question"]
    pool = []
    for r in qrows:
        qid = str(r["id"])
        if any(not hit5(page_hit_rank(runs[name][qid]["results"], r["file"], r["pages"])) for name in POOL_RUNS):
            pool.append(qid)
    by_id = {str(r["id"]): r for r in qrows}
    items: dict[tuple, dict] = {}
    for qid in pool:
        r = by_id[qid]
        for name in CANDIDATE_RUNS:
            if name not in runs:
                continue
            for res in runs[name][qid]["results"][:TOP_N]:
                fp = (res["filename"], page_of(res))
                if res["filename"] == r["file"] and fp[1] in r["pages"]:
                    continue
                key = (qid, *fp)
                it = items.setdefault(key, {"id": f"{qid}|{fp[0]}|{fp[1]}", "qid": qid, "file": fp[0], "page": fp[1],
                                            "label": res["citation"]["label"], "seen": [], "chunks": {}})
                it["seen"].append({"run": name, "rank": res["rank"]})
                c = it["chunks"].setdefault(res["chunk_id"], {"chunk_id": res["chunk_id"], "text": res["text"],
                                                              "seen": []})
                c["seen"].append({"run": name, "rank": res["rank"]})
    out = []
    for it in items.values():
        it["chunks"] = sorted(it["chunks"].values(), key=lambda c: min(s["rank"] for s in c["seen"]))
        out.append(it)
    order = {q: i for i, q in enumerate(pool)}
    out.sort(key=lambda it: (order[it["qid"]], min(s["rank"] for s in it["seen"])))
    return pool, out


def lenient_ranks(rows: list[dict], run: dict, judgments: dict[str, str], accept: tuple[str, ...]) -> dict:
    """ต่อข้อ: strict hit@5 / lenient hit@5 (top 5 มีหน้าที่คนตัดสินเป็น accept) / จำนวนหน้าใน top 5 ที่ยังไม่มีคำตัดสิน"""
    out = {}
    for r in rows:
        if r["type"] != "question":
            continue
        qid = str(r["id"])
        top = run[qid]["results"][:TOP_N]
        strict = hit5(page_hit_rank(top, r["file"], r["pages"]))
        lenient, unjudged = strict, 0
        if not strict:
            for res in top:
                v = judgments.get(f"{qid}|{res['filename']}|{page_of(res)}")
                if v is None:
                    unjudged += 1
                elif v in accept:
                    lenient = True
        out[qid] = {"strict": strict, "lenient": lenient, "unjudged": 0 if lenient else unjudged}
    return out


# ---------- รัน ----------

def load_rows(path: Path = QUESTIONS) -> list[dict]:
    from eval.questions import QuestionsError, load

    try:
        rows = load(path)
    except QuestionsError as e:
        raise SystemExit("\n".join(e.errors))
    for i, r in enumerate(rows, start=1):  # id ไม่ใส่ = เลขบรรทัด (schema)
        r.setdefault("id", i)
    return rows


def cmd_run(args) -> int:
    cfg = CONFIGS[args.config]
    run_id = args.run_id or args.config
    rows = load_rows(args.questions)
    headers = {"X-Service-Key": service_key(), "X-User-Id": USER[0], "X-User-Departments": USER[1],
               "X-User-Security-Level": str(USER[2])}
    http = httpx.Client(base_url=args.base_url.rstrip("/"), timeout=120.0)
    ready = http.get("/v1/health/ready", headers=headers)
    if ready.status_code != 200:
        raise SystemExit(f"API ยังไม่พร้อม: {ready.status_code} {ready.text}")
    stamp = time.strftime("%Y%m%d%H%M%S")
    for i in range(WARMUP):  # ไม่นับ (โหลดโมเดล / แคช)
        http.post("/v1/retrieve", headers={**headers, "X-Trace-Id": f"evalq-warmup-{stamp}-{i}"},
                  json={"query": "ทดสอบระบบก่อนวัด", "box_id": BOX_ID, "top_k": LIMIT})
    responses = {}
    for row in rows:
        trace = f"evalq-{run_id}-{stamp}-{row['id']}"
        t0 = time.perf_counter()
        r = http.post("/v1/retrieve", headers={**headers, "X-Trace-Id": trace},
                      json={"query": row["question"], "box_id": BOX_ID, "top_k": LIMIT})
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            raise SystemExit(f"{trace}: {r.status_code} {r.text}")
        responses[str(row["id"])] = {"trace": trace, "client_ms": ms, "body": r.json()}
    print(f"เรียก /v1/retrieve {len(rows)} ครั้ง", flush=True)
    time.sleep(1.0)  # ให้ log ลงไฟล์ครบ
    parsed = parse_api_log(args.api_log, {v["trace"] for v in responses.values()})
    if len(parsed) != len(responses):
        raise SystemExit(f"จับคู่ log ได้ {len(parsed)}/{len(responses)} — ไม่เขียนผล (ตรวจ --api-log)")
    wrong = {v["trace"]: parsed[v["trace"]]["extras"] for v in responses.values()
             if sorted(parsed[v["trace"]]["extras"]) != sorted(cfg["extras"])}
    if wrong:
        sample = next(iter(wrong.items()))
        raise SystemExit(f"extras ใน log ไม่ตรง config {args.config} ({len(wrong)} request เช่น {sample}) — ไม่เขียนผล")
    for v in responses.values():
        v["log"] = parsed[v["trace"]]
    out = {"run_id": run_id, "config": args.config, "label": cfg["label"], "env": cfg["env"], "extras": cfg["extras"],
           "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"), "base_url": args.base_url, "box_id": BOX_ID,
           "top_k": LIMIT, "warmup": WARMUP, "ready": ready.json(), "questions": str(args.questions.relative_to(ROOT)),
           "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(), "responses": responses}
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{run_id}.json"
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    ranks = [page_hit_rank(responses[str(r["id"])]["body"]["results"], r["file"], r["pages"])
             for r in rows if r["type"] == "question"]
    m = metrics(ranks)
    print(f"{run_id}: R@1 {m['recall_at_1']:.3f} R@5 {m['recall_at_5']:.3f} MRR {m['mrr']:.3f} -> {path}")
    return 0


def load_runs(names) -> dict[str, dict]:
    runs = {}
    for name in names:
        p = RUNS_DIR / f"{name}.json"
        if p.is_file():
            runs[name] = json.loads(p.read_text(encoding="utf-8"))
        else:
            print(f"ไม่มีผลรัน {name} ({p}) — ข้าม")
    return runs


def results_of(runs: dict[str, dict]) -> dict[str, dict[str, dict]]:
    """run -> qid -> {"results": [...]}"""
    return {n: {q: {"results": v["body"]["results"]} for q, v in run["responses"].items()} for n, run in runs.items()}


# ---------- รายงาน A + B ----------

def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": min(values), "p10": quantile(values, 0.1), "p25": quantile(values, 0.25),
            "median": quantile(values, 0.5), "p75": quantile(values, 0.75), "p90": quantile(values, 0.9),
            "max": max(values), "mean": statistics.fmean(values)}


def section_a(rows: list[dict], runs: dict[str, dict]) -> dict:
    qrows = [r for r in rows if r["type"] == "question"]
    out = {"configs": {}}
    all_ranks = {}
    for name, run in runs.items():
        resp = run["responses"]
        ranks = {str(r["id"]): page_hit_rank(resp[str(r["id"])]["body"]["results"], r["file"], r["pages"]) for r in qrows}
        all_ranks[name] = ranks
        lat = {"client": latency([v["client_ms"] for v in resp.values()])}
        lat.update(server_latency({v["trace"]: v["log"] for v in resp.values()}))
        out["configs"][name] = {
            "label": run["label"] + (" — รอบ 2 (process เดิม)" if name == REPEAT_RUN else ""),
            "measured_at": run["measured_at"], "all": metrics(list(ranks.values())),
            "by_file": {d: metrics([ranks[str(r["id"])] for r in qrows if r["file"] == f]) for d, f in DOCS.items()},
            "by_tag": {t: metrics([ranks[str(r["id"])] for r in qrows if tag_of(r) == t])
                       for t in ("ทั่วไป", "ต้อง OCR", "multi_hop")},
            "by_truth_pages": {k: metrics([ranks[str(r["id"])] for r in qrows if (len(r["pages"]) > 1) == multi])
                               for k, multi in (("หน้าเดียว", False), ("หลายหน้า", True))},
            "latency": lat, "ranks": ranks,
        }
    if BASE_RUN in all_ranks:
        base = all_ranks[BASE_RUN]
        for name, ranks in all_ranks.items():
            if name != BASE_RUN:
                out["configs"][name]["vs_baseline"] = first_rank_compare(base, ranks)
        if REPEAT_RUN in runs:
            a, b = runs[BASE_RUN]["responses"], runs[REPEAT_RUN]["responses"]
            top_diff = [q for q in a if [x["chunk_id"] for x in a[q]["body"]["results"]]
                        != [x["chunk_id"] for x in b[q]["body"]["results"]]]
            set_diff = [q for q in a if {x["chunk_id"] for x in a[q]["body"]["results"]}
                        != {x["chunk_id"] for x in b[q]["body"]["results"]}]
            score_diff = [abs((a[q]["body"]["top_score"] or 0) - (b[q]["body"]["top_score"] or 0)) for q in a]
            out["repeat"] = {"rank_changed": sorted(q for q in base if base[q] != all_ranks[REPEAT_RUN][q]),
                             "top10_order_changed": len(top_diff), "top10_set_changed": len(set_diff),
                             "top_score_changed": sum(1 for d in score_diff if d > 0),
                             "top_score_max_abs_diff": max(score_diff) if score_diff else None,
                             "n_questions": len(base), "n_requests": len(a)}
    # ข้อที่พลาดทุก config (ไม่รวมรอบซ้ำ)
    names = [n for n in all_ranks if n != REPEAT_RUN]
    by_id = {str(r["id"]): r for r in qrows}
    miss5 = [q for q in by_id if all(not hit5(all_ranks[n][q]) for n in names)]
    miss10 = [q for q in miss5 if all(all_ranks[n][q] is None for n in names)]
    out["missed_by_all"] = {"configs": names, "top5": miss5, "top10": miss10,
                            "top5_by_file": {d: sum(1 for q in miss5 if by_id[q]["file"] == f) for d, f in DOCS.items()},
                            "top5_by_tag": {t: sum(1 for q in miss5 if tag_of(by_id[q]) == t)
                                            for t in ("ทั่วไป", "ต้อง OCR", "multi_hop")}}
    return out


def is_near_miss(row: dict) -> bool:
    """unanswerable ที่ผู้ทำชุดคำถามระบุว่า "ใกล้เคียง" (ถามของบริษัทอื่น / ปีที่ไม่มี ฯลฯ)"""
    return (row.get("note") or "").startswith("ใกล้เคียง")


def section_b(rows: list[dict], runs: dict[str, dict]) -> dict:
    by_type = {t: [str(r["id"]) for r in rows if r["type"] == t] for t in SCORE_TYPES}
    sub = {"unanswerable: ใกล้เคียง": [str(r["id"]) for r in rows if r["type"] == "unanswerable" and is_near_miss(r)],
           "unanswerable: อื่น": [str(r["id"]) for r in rows if r["type"] == "unanswerable" and not is_near_miss(r)]}
    cal, test = stratified_split(rows)
    sources = {}
    if BASE_RUN in runs:
        resp = runs[BASE_RUN]["responses"]
        sources["top_score (dense cosine, baseline)"] = {q: resp[q]["body"]["top_score"] for q in resp}
    if "reranker" in runs:
        resp = runs["reranker"]["responses"]
        sources["reranker score อันดับ 1 (bge-v2-m3)"] = {q: resp[q]["log"]["rerank_top"] for q in resp}
    out = {"split": {"seed": SPLIT_SEED, "calibrate": {t: sum(1 for q in ids if q in cal) for t, ids in by_type.items()},
                     "test": {t: sum(1 for q in ids if q in test) for t, ids in by_type.items()},
                     "calibrate_ids": sorted(cal)},
           "status_at_current_min_score": {}, "scores": {}}
    if BASE_RUN in runs:
        resp = runs[BASE_RUN]["responses"]
        for t, ids in by_type.items():
            c: dict[str, int] = {}
            for q in ids:
                c[resp[q]["body"]["status"]] = c.get(resp[q]["body"]["status"], 0) + 1
            out["status_at_current_min_score"][t] = c
    for name, scores in sources.items():
        missing = [q for q, s in scores.items() if s is None]
        val = {q: (-math.inf if s is None else s) for q, s in scores.items()}  # ไม่มีผล = ถูกปฏิเสธแน่นอน
        dist = {t: describe([val[q] for q in ids if val[q] != -math.inf]) for t, ids in {**by_type, **sub}.items()}
        q_all = [val[q] for q in by_type["question"]]
        aucs = {f"question vs {t}": auc(q_all, [val[q] for q in ids]) for t, ids in
                {"unanswerable": by_type["unanswerable"], **sub, "chitchat": by_type["chitchat"]}.items()}
        aucs["question vs (unanswerable+chitchat)"] = auc(q_all, [val[q] for t in ("unanswerable", "chitchat")
                                                                   for q in by_type[t]])
        thresholds = []
        for budget in BUDGETS:
            t_ = pick_threshold([val[q] for q in by_type["question"] if q in cal], budget)
            row = {"budget": budget, "threshold": t_}
            for half, ids_half in (("calibrate", cal), ("test", test)):
                row[half] = {}
                for t, ids in by_type.items():
                    k, n = reject_rate([val[q] for q in ids if q in ids_half], t_)
                    row[half][t] = {"rejected": k, "n": n}
            thresholds.append(row)
        out["scores"][name] = {"missing": missing, "distribution": dist, "auc": aucs, "thresholds": thresholds}
    return out


def compact_runs(runs: dict[str, dict]) -> dict:
    """ผลย่อต่อรัน (ไม่มีข้อความ chunk) — ไว้คำนวณซ้ำได้จาก eval/results · ``top`` = "ไฟล์:หน้า:score" เรียงตาม rank คั่นช่องว่าง"""
    out = {}
    for name, run in runs.items():
        out[name] = {"label": run["label"], "env": run["env"], "measured_at": run["measured_at"],
                     "questions_sha256": run["questions_sha256"],
                     "responses": {q: {"status": v["body"]["status"], "top_score": v["body"]["top_score"],
                                       "client_ms": round(v["client_ms"], 1),
                                       "rerank_top": v["log"]["rerank_top"],
                                       "top": " ".join(f"{FILE_SHORT.get(x['filename'], x['filename'])}:{page_of(x)}:"
                                                       f"{x['score']:.4f}" for x in v["body"]["results"])}
                                   for q, v in run["responses"].items()}}
    return out


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def frac(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "-"


def render_md(out: dict) -> str:
    a, b = out["A"], out["B"]
    cfgs = a["configs"]
    L = ["# Evaluation รอบจริง — questions.jsonl ผ่าน POST /v1/retrieve", "",
         f"- ชุดคำถาม `{out['questions']}` (sha256 `{out['questions_sha256'][:12]}…`) · {out['counts']}",
         f"- กล่อง `{BOX_ID}` (public: CPALL, PTT, KBANK — demo data เดิม) · top_k {LIMIT} · ผู้เรียก `{USER[0]}` · "
         "ค่าตั้งอื่นของ API = ค่าเริ่มต้น (ไม่ได้เปลี่ยน RETRIEVE_MIN_SCORE / SEARCH_MODE / CHUNK_TARGET_CHARS)",
         "- hit = filename ตรง `file` และ citation.page อยู่ใน `pages` · R@1 / R@5 / MRR@10 เฉพาะ type=question · "
         "แต่ละ config รีสตาร์ต API ด้วย env ของตัวเอง (ตรวจ `extras` ใน log ทุก request) · GPU ไม่มี OCR ระหว่างวัด",
         f"- วัดเมื่อ: " + " · ".join(f"{n} {c['measured_at']}" for n, c in cfgs.items()), "",
         "## A. Retrieval (type=question)", "",
         "| config | n | R@1 | R@5 | MRR@10 | ดีขึ้น / แย่ลง / เท่าเดิม (เทียบ baseline_r1) | R@5 ได้เพิ่ม / เสีย |",
         "|---|---:|---:|---:|---:|---|---|"]
    for n, c in cfgs.items():
        m, v = c["all"], c.get("vs_baseline")
        cmp_ = f"{v['better']} / {v['worse']} / {v['same']}" if v else "(อ้างอิง)"
        at5 = f"+{len(v['at5']['gained'])} / −{len(v['at5']['lost'])}" if v else ""
        L.append(f"| {n} — {c['label']} | {m['n']} | {pct(m['recall_at_1'])} | {pct(m['recall_at_5'])} | "
                 f"{m['mrr']:.3f} | {cmp_} | {at5} |")
    L += ["", "ดีขึ้น/แย่ลง = อันดับแรกที่ hit (ไม่ hit ใน 10 = แย่สุด) · R@5 ได้เพิ่ม/เสีย = ข้อที่ข้ามเส้น top 5", ""]
    if "repeat" in a:
        r = a["repeat"]
        L += [f"**การแกว่งของ baseline (รันซ้ำใน process เดียวกัน):** อันดับแรกที่ hit ต่างกัน {len(r['rank_changed'])}/"
              f"{r['n_questions']} คำถาม" + (f" ({', '.join(r['rank_changed'])})" if r["rank_changed"] else "")
              + f" · ทุก {r['n_requests']} request: ลำดับ top 10 ต่างกัน {r['top10_order_changed']} · ชุด top 10 ต่างกัน "
              f"{r['top10_set_changed']} · top_score ต่างกัน {r['top_score_changed']} (ต่างสูงสุด {r['top_score_max_abs_diff']:.4f})", ""]
    L += ["### แยกไฟล์", "", "| config | " + " | ".join(f"{d} R@1 | {d} R@5 | {d} MRR" for d in DOCS) + " |",
          "|---|" + "---:|" * (3 * len(DOCS))]
    for n, c in cfgs.items():
        L.append(f"| {n} | " + " | ".join(f"{pct(c['by_file'][d]['recall_at_1'])} | {pct(c['by_file'][d]['recall_at_5'])} | "
                                          f"{c['by_file'][d]['mrr']:.3f}" for d in DOCS) + " |")
    L += ["", "n ต่อไฟล์: " + " · ".join(f"{d} {cfgs[next(iter(cfgs))]['by_file'][d]['n']}" for d in DOCS), "",
          "### แยกประเภทคำถาม (จาก `note` ในชุดคำถาม) — R@5", "",
          "| config | " + " | ".join(f"{t} (n={m['n']})" for t, m in next(iter(cfgs.values()))["by_tag"].items())
          + " | " + " | ".join(f"truth {k} (n={m['n']})" for k, m in next(iter(cfgs.values()))["by_truth_pages"].items())
          + " |", "|---|" + "---:|" * 5]
    for n, c in cfgs.items():
        L.append(f"| {n} | " + " | ".join(pct(m["recall_at_5"]) for m in c["by_tag"].values()) + " | "
                 + " | ".join(pct(m["recall_at_5"]) for m in c["by_truth_pages"].values()) + " |")
    L += ["", "### latency ต่อ request (ms)", "",
          "| config | client p50 | client p95 | server p50 | server p95 | embed p50 | search p50 | rerank p50 | rerank p95 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, c in cfgs.items():
        lat = c["latency"]

        def g(k, q):
            return f"{lat[k][q]:.0f}" if k in lat and lat[k].get(q) is not None else "-"
        L.append(f"| {n} | {g('client', 'p50')} | {g('client', 'p95')} | {g('server_total', 'p50')} | "
                 f"{g('server_total', 'p95')} | {g('embed', 'p50')} | {g('search', 'p50')} | {g('rerank', 'p50')} | "
                 f"{g('rerank', 'p95')} |")
    L += ["", "client = รอบ HTTP ทั้งหมด (เครื่องเดียวกับ API, เรียกทีละข้อ) · server = embed + search + rerank + map จาก log · "
              f"warm-up {WARMUP} ครั้งไม่นับ · query embed บน CPU fp32 ทุก config (reranker บน cuda)", ""]
    mba = a["missed_by_all"]
    L += [f"### ข้อที่พลาด top 5 ทุก config ({', '.join(mba['configs'])})", "",
          f"- พลาด top 5 ทุก config: **{len(mba['top5'])}** ข้อ · ไม่ติด top 10 เลยทุก config: {len(mba['top10'])} ข้อ",
          "- แยกไฟล์: " + " · ".join(f"{d} {k}" for d, k in mba["top5_by_file"].items())
          + " · แยกประเภท: " + " · ".join(f"{t} {k}" for t, k in mba["top5_by_tag"].items()),
          f"- รายข้อ (top 5): {', '.join(mba['top5'])}", ""]

    L += ["## B. การปฏิเสธ (unanswerable / chitchat) — รายงานอย่างเดียว ไม่ได้เปลี่ยน RETRIEVE_MIN_SCORE", "",
          f"status ที่ได้จริงที่เกณฑ์ปัจจุบัน ({out['retrieve_min_score']}, baseline_r1): "
          + " · ".join(f"{t}: " + ", ".join(f"{k}={v}" for k, v in c.items()) for t, c in b["status_at_current_min_score"].items()),
          ""]
    for name, s in b["scores"].items():
        L += [f"### {name}", "", "| type | n | min | p10 | p25 | median | p75 | p90 | max |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for t, d in s["distribution"].items():
            if d.get("n"):
                L.append(f"| {t} | {d['n']} | {d['min']:.3f} | {d['p10']:.3f} | {d['p25']:.3f} | {d['median']:.3f} | "
                         f"{d['p75']:.3f} | {d['p90']:.3f} | {d['max']:.3f} |")
        if s["missing"]:
            L.append(f"\nไม่มีคะแนน (ไม่มีผล = นับว่าถูกปฏิเสธ): {', '.join(s['missing'])}")
        L += ["", "AUC: " + " · ".join(f"{k} **{v:.3f}**" for k, v in s["auc"].items() if v is not None), "",
              "| งบคำถามจริงหลุด (calibrate) | เกณฑ์ t | calibrate: question ถูกปฏิเสธผิด | test: question ถูกปฏิเสธผิด | "
              "test: unanswerable ปฏิเสธถูก | test: chitchat ปฏิเสธถูก |", "|---|---:|---:|---:|---:|---:|"]
        for row in s["thresholds"]:
            c, t = row["calibrate"], row["test"]
            L.append(f"| ≤ {100 * row['budget']:.0f}%{' (หลัก)' if row['budget'] == BUDGETS[0] else ''} | "
                     f"{row['threshold']:.4f} | {frac(c['question']['rejected'], c['question']['n'])} | "
                     f"{frac(t['question']['rejected'], t['question']['n'])} | "
                     f"{frac(t['unanswerable']['rejected'], t['unanswerable']['n'])} | "
                     f"{frac(t['chitchat']['rejected'], t['chitchat']['n'])} |")
        L.append("")
    sp = b["split"]
    L += [f"แบ่งข้อ stratified ตาม type (seed {sp['seed']}): calibrate "
          + ", ".join(f"{t} {k}" for t, k in sp["calibrate"].items()) + " · test "
          + ", ".join(f"{t} {k}" for t, k in sp["test"].items())
          + " · เกณฑ์ = ค่าสูงสุดที่คำถามจริงในครึ่ง calibrate ถูกปฏิเสธ (คะแนน < t) ไม่เกินงบ · ตัวเลขฝั่ง test คือผลที่รายงาน", ""]
    if out.get("review"):
        rv = out["review"]
        L += ["## C. รายการให้คนตรวจ (ระบบไม่ตัดสินเอง)", "",
              f"- pool = ข้อที่ {' หรือ '.join(POOL_RUNS)} พลาดใน top 5: **{rv['pool']}** ข้อ · ผู้สมัคร (ข้อ, file, page) จาก top 5 ของ "
              f"{', '.join(CANDIDATE_RUNS)}: **{rv['items']}** รายการ (ภาพหน้า {rv['pages']} หน้าไม่ซ้ำ)",
              f"- หน้าตรวจ: `{rv['html']}` (ไม่เข้า git) · รายการ: `{rv['items_file']}` · คิด R@5 แบบ lenient: "
              "`python scripts/questions_eval.py lenient --judgments <ไฟล์ที่ดาวน์โหลด>`", ""]
    return "\n".join(L) + "\n"


def cmd_report(args) -> int:
    from app.config import build_settings

    rows = load_rows(args.questions)
    runs = load_runs(args.runs)
    if BASE_RUN not in runs:
        raise SystemExit(f"ต้องมี {BASE_RUN} ก่อน")
    sha = hashlib.sha256(args.questions.read_bytes()).hexdigest()
    bad = [n for n, r in runs.items() if r["questions_sha256"] != sha]
    if bad:
        raise SystemExit(f"ผลรัน {bad} วัดกับชุดคำถามคนละเวอร์ชัน")
    counts = {t: sum(1 for r in rows if r["type"] == t) for t in SCORE_TYPES}
    out = {"questions": str(args.questions.relative_to(ROOT)).replace("\\", "/"), "questions_sha256": sha,
           "counts": ", ".join(f"{t} {k}" for t, k in counts.items()),
           "retrieve_min_score": build_settings().retrieve_min_score, "generated_at": time.strftime("%Y-%m-%d %H:%M"),
           "A": section_a(rows, runs), "B": section_b(rows, runs)}
    if REVIEW_ITEMS.is_file():
        rv = json.loads(REVIEW_ITEMS.read_text(encoding="utf-8"))
        out["review"] = {"pool": len(rv["pool"]), "items": len(rv["items"]), "pages": rv["unique_pages"],
                         "html": rv["html"], "items_file": str(REVIEW_ITEMS.relative_to(ROOT)).replace("\\", "/")}
    out["runs"] = compact_runs(runs)
    md = render_md(out)
    (RESULTS / "questions_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS / "questions_eval.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


# ---------- C: หน้าตรวจ ----------

def render_pages(keys: set[tuple[str, int]], out_dir: Path, zoom: float = 1.5) -> dict[str, str]:
    import fitz  # pymupdf

    out_dir.mkdir(parents=True, exist_ok=True)
    paths, docs = {}, {}
    for file, page in sorted(keys):
        name = f"{FILE_SHORT.get(file, Path(file).stem)}_p{page:03d}.jpg"
        dest = out_dir / name
        if not dest.is_file():
            doc = docs.get(file) or docs.setdefault(file, fitz.open(PDF_DIR / file))
            pix = doc[page - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            dest.write_bytes(pix.tobytes("jpeg", jpg_quality=75))
        paths[f"{file}|{page}"] = f"pages/{name}"
    for d in docs.values():
        d.close()
    return paths


def cmd_review(args) -> int:
    rows = load_rows(args.questions)
    runs = load_runs(CANDIDATE_RUNS)
    missing = [n for n in POOL_RUNS if n not in runs]
    if missing:
        raise SystemExit(f"ต้องมีผลรัน {missing} ก่อน")
    pool, items = build_candidates(rows, results_of(runs))
    by_id = {str(r["id"]): r for r in rows}
    page_keys = {(it["file"], it["page"]) for it in items if it["page"] is not None}
    page_keys |= {(by_id[q]["file"], p) for q in pool for p in by_id[q]["pages"]}  # หน้าตาม truth (ไว้เปิดดูประกอบ)
    images = render_pages(page_keys, REVIEW_DIR / "pages")
    ranks = {n: {q: page_hit_rank(results_of(runs)[n][q]["results"], by_id[q]["file"], by_id[q]["pages"]) for q in pool}
             for n in runs}
    dataset_id = hashlib.sha256("\n".join(it["id"] for it in items).encode()).hexdigest()[:16]
    data = {"dataset_id": dataset_id, "runs": list(runs), "pool": [
        {"qid": q, "question": by_id[q]["question"], "file": by_id[q]["file"], "pages": by_id[q]["pages"],
         "tag": tag_of(by_id[q]), "ranks": {n: ranks[n][q] for n in runs},
         "truth_images": [images[f"{by_id[q]['file']}|{p}"] for p in by_id[q]["pages"]]} for q in pool],
        "items": [{**it, "image": images.get(f"{it['file']}|{it['page']}")} for it in items]}
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    (REVIEW_DIR / "index.html").write_text(html, encoding="utf-8")
    html_rel = str((REVIEW_DIR / "index.html").relative_to(ROOT)).replace("\\", "/")
    REVIEW_ITEMS.write_text(json.dumps({
        "dataset_id": dataset_id, "generated_at": time.strftime("%Y-%m-%d %H:%M"), "html": html_rel,
        "pool_runs": list(POOL_RUNS), "candidate_runs": list(runs), "top_n": TOP_N, "pool": pool,
        "unique_pages": len({(it["file"], it["page"]) for it in items}),
        "items": [{"id": it["id"], "qid": it["qid"], "file": it["file"], "page": it["page"], "seen": it["seen"],
                   "chunk_ids": [c["chunk_id"] for c in it["chunks"]]} for it in items]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"pool {len(pool)} ข้อ · ผู้สมัคร {len(items)} รายการ · ภาพ {len(images)} หน้า -> {REVIEW_DIR / 'index.html'}")
    print(f"รายการ -> {REVIEW_ITEMS}")
    return 0


def cmd_lenient(args) -> int:
    rows = load_rows(args.questions)
    spec = json.loads(REVIEW_ITEMS.read_text(encoding="utf-8"))
    jd = json.loads(args.judgments.read_text(encoding="utf-8"))
    if jd.get("dataset_id") != spec["dataset_id"]:
        raise SystemExit(f"judgments เป็นของชุดรายการอื่น ({jd.get('dataset_id')} ≠ {spec['dataset_id']})")
    item_ids = {it["id"] for it in spec["items"]}
    judgments = {k: v["verdict"] for k, v in jd["judgments"].items() if k in item_ids}
    runs = results_of(load_runs(DEFAULT_RUNS))
    qn = sum(1 for r in rows if r["type"] == "question")
    counts = {v: sum(1 for x in judgments.values() if x == v) for v in ("yes", "no", "unsure")}
    out = {"dataset_id": spec["dataset_id"], "judged": len(judgments), "items": len(item_ids), "verdicts": counts,
           "configs": {}}
    for name, run in runs.items():
        strict = lenient_ranks(rows, run, judgments, ("yes",))
        loose = lenient_ranks(rows, run, judgments, ("yes", "unsure"))
        covered = name in spec["candidate_runs"]
        out["configs"][name] = {
            "n": qn, "strict_r5": sum(v["strict"] for v in strict.values()) / qn,
            "lenient_r5": sum(v["lenient"] for v in strict.values()) / qn,
            "lenient_r5_with_unsure": sum(v["lenient"] for v in loose.values()) / qn,
            "questions_with_unjudged": sum(1 for v in strict.values() if v["unjudged"]),
            "fully_covered_by_review": covered,
            "by_file": {d: {"strict": sum(strict[str(r["id"])]["strict"] for r in rows if r.get("file") == f),
                            "lenient": sum(strict[str(r["id"])]["lenient"] for r in rows if r.get("file") == f),
                            "n": sum(1 for r in rows if r.get("file") == f)} for d, f in DOCS.items()}}
    L = ["# R@5 แบบ lenient (จากคำตัดสินของคน) — ตัวเลขหลักยังเป็น strict", "",
         f"- คำตัดสิน {len(judgments)}/{len(item_ids)} รายการ (มีคำตอบ {counts['yes']} · ไม่มี {counts['no']} · ไม่แน่ใจ {counts['unsure']})",
         "- lenient = hit เดิม หรือ top 5 มีหน้าที่คนตัดสินว่ามีคำตอบ · config ที่ไม่ได้อยู่ในชุดผู้สมัคร (ไม่ครบ) = ขอบล่าง", "",
         "| config | strict R@5 | lenient R@5 | lenient + ไม่แน่ใจ | ข้อที่ top 5 ยังมีหน้าไม่ได้ตรวจ | " +
         " | ".join(f"{d} strict→lenient" for d in DOCS) + " |", "|---|---:|---:|---:|---:|" + "---|" * len(DOCS)]
    for n, c in out["configs"].items():
        L.append(f"| {n}{'' if c['fully_covered_by_review'] else ' (ขอบล่าง)'} | {pct(c['strict_r5'])} | "
                 f"{pct(c['lenient_r5'])} | {pct(c['lenient_r5_with_unsure'])} | {c['questions_with_unjudged']} | "
                 + " | ".join(f"{b['strict']}→{b['lenient']}/{b['n']}" for b in c["by_file"].values()) + " |")
    md = "\n".join(L) + "\n"
    (RESULTS / "questions_lenient.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS / "questions_lenient.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


STATUS_RUNS = {"status_baseline": "rerank_off", "status_reranker": "rerank_on"}


def expected_status(body: dict, min_score: float, rerank_min_score: float) -> str:
    """status ที่ควรได้ตามกติกาของ API (ไว้ตรวจว่า API รันด้วยเกณฑ์ที่คิดว่าใช้)"""
    if not body["results"]:
        return "no_relevant_context"
    if body.get("top_relevance") is not None:
        return "ok" if body["top_relevance"] >= rerank_min_score else "no_relevant_context"
    return "ok" if body["top_score"] >= min_score else "no_relevant_context"


def status_table(rows: list[dict], responses: dict[str, dict], ids: set[str] | None = None) -> dict:
    out = {}
    for t in SCORE_TYPES:
        qs = [str(r["id"]) for r in rows if r["type"] == t and (ids is None or str(r["id"]) in ids)]
        k = sum(1 for q in qs if responses[q]["body"]["status"] == "no_relevant_context")
        out[t] = {"no_relevant_context": k, "n": len(qs)}
    return out


def optimism_note(configs: dict) -> str:
    """เทียบชุดเต็มกับครึ่ง test จริง: ชุดเต็มดีกว่าทุกช่องหรือไม่ (คำถามจริง: ปฏิเสธน้อยกว่าดี · อื่น: ปฏิเสธมากกว่าดี)"""
    worse = []
    for name, c in configs.items():
        for t in SCORE_TYPES:
            f, h = c["full"][t], c["test_half"][t]
            rf, rh = f["no_relevant_context"] / f["n"], h["no_relevant_context"] / h["n"]
            if (rf > rh) if t == "question" else (rf < rh):
                worse.append(f"{name}/{t}")
    if not worse:
        return "รอบนี้ชุดเต็มดีกว่าหรือเท่าครึ่ง test ทุกช่อง (สอดคล้องกับที่คาดว่าดีเกินจริง)"
    return ("รอบนี้ชุดเต็ม**ไม่ได้ดีกว่า**ครึ่ง test ในช่อง " + ", ".join(worse)
            + " (ครึ่ง test เล็ก — unanswerable 19, chitchat 6 — สัดส่วนแกว่งได้มาก) ใช้ทั้งสองคอลัมน์ประกอบกัน")


def cmd_status(args) -> int:
    from app.config import Settings

    defaults = Settings()
    thresholds = {"RETRIEVE_MIN_SCORE": defaults.retrieve_min_score, "RERANK_MIN_SCORE": defaults.rerank_min_score}
    rows = load_rows(args.questions)
    runs = load_runs(STATUS_RUNS)
    if len(runs) != len(STATUS_RUNS):
        raise SystemExit(f"ต้องมีผลรัน {sorted(STATUS_RUNS)}")
    _, test = stratified_split(rows)
    out = {"measured_at": {n: r["measured_at"] for n, r in runs.items()}, "thresholds": thresholds,
           "split_seed": SPLIT_SEED, "configs": {}}
    for name, run in runs.items():
        resp = run["responses"]
        mism = [q for q, v in resp.items()
                if v["body"]["status"] != expected_status(v["body"], defaults.retrieve_min_score, defaults.rerank_min_score)]
        rel = [v["body"].get("top_relevance") for v in resp.values() if v["body"]["results"]]
        mode_ok = all(x is not None for x in rel) if STATUS_RUNS[name] == "rerank_on" else all(x is None for x in rel)
        if mism or not mode_ok:
            raise SystemExit(f"{name}: status ไม่ตรงเกณฑ์ค่าเริ่มต้น {len(mism)} ข้อ / top_relevance ไม่ตรงโหมด "
                             f"({'ok' if mode_ok else 'ผิด'}) — API อาจไม่ได้รันด้วยค่าเริ่มต้น")
        out["configs"][name] = {"label": run["label"], "rule": "top_relevance < RERANK_MIN_SCORE"
                                if STATUS_RUNS[name] == "rerank_on" else "top_score < RETRIEVE_MIN_SCORE",
                                "full": status_table(rows, resp), "test_half": status_table(rows, resp, test),
                                "rejected_questions": sorted(q for q in resp if q in {str(r["id"]) for r in rows
                                                             if r["type"] == "question"}
                                                             and resp[q]["body"]["status"] == "no_relevant_context")}
    L = ["# status ที่เกณฑ์ใหม่ (ค่าเริ่มต้นของ API) — questions.jsonl ผ่าน POST /v1/retrieve", "",
         f"- เกณฑ์: RETRIEVE_MIN_SCORE {thresholds['RETRIEVE_MIN_SCORE']} (dense top_score, reranker ปิด) · "
         f"RERANK_MIN_SCORE {thresholds['RERANK_MIN_SCORE']} (top_relevance, RERANKER=bge-v2-m3 บน cuda)",
         "- **เกณฑ์เลือกจากครึ่ง calibrate** (stratified ตาม type, seed "
         f"{SPLIT_SEED}; ดู questions_eval.md ข้อ B) — ตัวเลขบนชุดเต็มรวมครึ่ง calibrate ที่ใช้เลือกเกณฑ์ด้วย "
         "จึง**ดีเกินจริงเล็กน้อย**โดยหลักการ · คอลัมน์ครึ่ง test คือตัวเลขที่ไม่วัดวน · "
         "ค่าเริ่มต้น 0.55 ปัดจากเกณฑ์ที่เลือกได้ (dense 0.5483, reranker 0.5513) จึงต่างจากตาราง B ของ questions_eval.md เล็กน้อย",
         "- " + optimism_note(out["configs"]),
         "- ตรวจแล้ว: status ทุก request ตรงกับกติกาที่เกณฑ์ค่าเริ่มต้น และ top_relevance มี/null ตรงโหมด",
         "- วัดเมื่อ: " + " · ".join(f"{n} {t}" for n, t in out["measured_at"].items()), "",
         "| config (กติกา) | type | ชุดเต็ม: no_relevant_context | ครึ่ง test: no_relevant_context |", "|---|---|---:|---:|"]
    meaning = {"question": "คำถามจริง (ปฏิเสธผิด)", "unanswerable": "unanswerable (ปฏิเสธถูก)",
               "chitchat": "chitchat (ปฏิเสธถูก)"}
    for name, c in out["configs"].items():
        for t in SCORE_TYPES:
            f, h = c["full"][t], c["test_half"][t]
            L.append(f"| {name} ({c['rule']}) | {meaning[t]} | {frac(f['no_relevant_context'], f['n'])} | "
                     f"{frac(h['no_relevant_context'], h['n'])} |")
    L += [""] + [f"- คำถามจริงที่ได้ no_relevant_context ใน {n}: {', '.join(c['rejected_questions']) or '-'}"
                 for n, c in out["configs"].items()]
    md = "\n".join(L) + "\n"
    (RESULTS / "questions_status.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS / "questions_status.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Evaluation บน questions.jsonl ผ่าน /v1/retrieve")
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="วัด 1 config (API ต้องรันด้วย env ของ config นี้)")
    p.add_argument("--config", required=True, choices=sorted(CONFIGS))
    p.add_argument("--run-id", help="ชื่อผลรัน (ค่าเริ่มต้น = ชื่อ config)")
    p.add_argument("--api-log", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p = sub.add_parser("report", help="A + B -> eval/results/questions_eval.*")
    p.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    sub.add_parser("review", help="C: หน้าตรวจ + รายการ")
    sub.add_parser("status", help="status ที่เกณฑ์ค่าเริ่มต้น จากรัน status_baseline / status_reranker")
    p = sub.add_parser("lenient", help="R@5 แบบ lenient จาก judgments ที่ดาวน์โหลดจากหน้าตรวจ")
    p.add_argument("--judgments", type=Path, required=True)
    args = ap.parse_args(argv)
    args.questions = args.questions.resolve()
    return {"run": cmd_run, "report": cmd_report, "review": cmd_review, "lenient": cmd_lenient,
            "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
