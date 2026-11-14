"""รวมผล ``retrieval_api_eval`` หลายรอบ (ไม่ rerank / rerank CPU / rerank GPU) เป็นตารางเทียบเดียว

    python scripts/reranker_compare.py retrieval_api_eval_rerank_off retrieval_api_eval_rerank_bge_cpu \\
        retrieval_api_eval_rerank_bge_cuda --out reranker_eval

ไม่เรียก API ไม่จูนอะไร — อ่าน eval/results/<name>.json อย่างเดียว
แยก unanswerable ออกจากคำถามจริงได้แค่ไหน: AUC (Mann-Whitney) ของคะแนนอันดับ 1 = โอกาสที่คำถามจริงที่สุ่มมาได้คะแนนสูงกว่า
unanswerable ที่สุ่มมา (0.5 = แยกไม่ได้ · 1.0 = แยกได้หมด) — **ไม่ได้เลือกเกณฑ์** แค่บอกว่ามีเกณฑ์ที่แยกได้หรือไม่
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def groups(run: dict, field: str) -> dict[str, list[float]]:
    """คะแนนอันดับ 1 ต่อกลุ่ม: field = top_score (dense cosine จาก response) | rerank (rerank_top จาก log)"""
    ranks = run["ranks"]
    resp = run["responses"]

    def value(qid):
        if field == "top_score":
            return resp[qid]["top_score"]
        return (run.get("rerank_top") or {}).get(qid)

    out: dict[str, list[float]] = {"question": [], "question: hit@1": [], "question: ไม่ hit@1": []}
    for qid, r in ranks.items():
        v = value(qid)
        if v is None:
            continue
        out["question"].append(v)
        out["question: hit@1" if r == 1 else "question: ไม่ hit@1"].append(v)
    for qid, x in resp.items():
        if x["type"] != "question":
            v = value(qid)
            if v is not None:
                out.setdefault(x["type"], []).append(v)
    return out


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def f3(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="+", help="ชื่อไฟล์ใน eval/results (ไม่มี .json) — ตัวแรก = ฐานเทียบ")
    ap.add_argument("--out", default="reranker_eval")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    runs = {n: json.loads((RESULTS / f"{n}.json").read_text(encoding="utf-8")) for n in args.names}
    base_name = args.names[0]
    base = runs[base_name]

    L = ["# cross-encoder reranker (BAAI/bge-reranker-v2-m3) บนชุด dev ผ่าน /v1/retrieve", "",
         "- rerank 20 อันดับแรกจาก RRF แล้วคืน top 10 · hit / ตัวชี้วัด = นิยามเดียวกับ scripts/retrieval_dev_eval.py "
         "(ไม่ปรับพารามิเตอร์/เกณฑ์ใดๆ) · score ใน API ยังเป็น dense cosine (ไม่ได้เปลี่ยน)", ""]
    L += ["## Recall@1 / Recall@5 / MRR", "",
          "| รอบ | (ก) 50 ข้อ R@1 | R@5 | MRR | (ข) 70 ข้อ R@1 | R@5 | MRR | CPALL MRR | PTT MRR | KBANK MRR |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    summary: dict = {"runs": {}}
    for name, run in runs.items():
        a, b, d = run["reachable"]["new"], run["all"], run["by_doc"]
        L.append(f"| {run.get('label') or name} | {pct(a['recall_at_1'])} | {pct(a['recall_at_5'])} | {a['mrr']:.3f} | "
                 f"{pct(b['recall_at_1'])} | {pct(b['recall_at_5'])} | {b['mrr']:.3f} | "
                 + " | ".join(f"{d[k]['mrr']:.3f}" for k in ("CPALL", "PTT", "KBANK")) + " |")
        summary["runs"][name] = {"label": run.get("label"), "reachable": a, "all": b, "by_doc": d,
                                 "latency": run["latency"]}
    L += ["", f"รายข้อเทียบกับ {base.get('label') or base_name} (อันดับแรกที่ hit ใน 70 ข้อ):", ""]
    for name, run in list(runs.items())[1:]:
        better = sum(1 for q, r in run["ranks"].items() if (r or 99) < (base["ranks"][q] or 99))
        worse = sum(1 for q, r in run["ranks"].items() if (r or 99) > (base["ranks"][q] or 99))
        same = len(run["ranks"]) - better - worse
        L.append(f"- {run.get('label') or name}: ดีขึ้น {better} · แย่ลง {worse} · เท่าเดิม {same}")
        summary["runs"][name]["vs_base"] = {"better": better, "worse": worse, "same": same}

    L += ["", "## latency ต่อ request (ms)", "", "| รอบ | ช่วง | p50 | p95 | max |", "|---|---|---:|---:|---:|"]
    for name, run in runs.items():
        for stage in ("client", "embed", "search", "rerank", "server_total"):
            v = run["latency"].get(stage)
            if v and v.get("n"):
                L.append(f"| {run.get('label') or name} | {stage} | {v['p50']:.0f} | {v['p95']:.0f} | {v['max']:.0f} |")

    L += ["", "## คะแนนอันดับ 1: แยก unanswerable / chitchat ออกจากคำถามจริงได้ไหม", "",
          "AUC = โอกาสที่คำถามจริง (type=question) ได้คะแนนอันดับ 1 สูงกว่า unanswerable/chitchat (0.5 = แยกไม่ได้, 1 = แยกได้หมด)",
          "", "| รอบ | คะแนน | question vs unanswerable | question vs chitchat | hit@1 vs ไม่ hit@1 |",
          "|---|---|---:|---:|---:|"]
    summary["separation"] = {}
    for name, run in runs.items():
        for field, label in (("top_score", "dense cosine (top_score)"), ("rerank", "reranker (rerank_top)")):
            g = groups(run, field)
            if not g["question"]:
                continue
            row = {"vs_unanswerable": auc(g["question"], g.get("unanswerable", [])),
                   "vs_chitchat": auc(g["question"], g.get("chitchat", [])),
                   "hit1_vs_rest": auc(g["question: hit@1"], g["question: ไม่ hit@1"])}
            summary["separation"][f"{name}:{field}"] = row
            L.append(f"| {run.get('label') or name} | {label} | {f3(row['vs_unanswerable'])} | {f3(row['vs_chitchat'])} | "
                     f"{f3(row['hit1_vs_rest'])} |")

    for name, run in runs.items():
        if not run.get("rerank_score"):
            continue
        L += ["", f"### การกระจายคะแนน reranker อันดับ 1 — {run.get('label') or name}", "",
              "| กลุ่ม | n | min | p10 | p25 | median | p75 | p90 | max |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for g, d in run["rerank_score"].items():
            if d.get("n"):
                L.append(f"| {g} | {d['n']} | {d['min']:.3f} | {d['p10']:.3f} | {d['p25']:.3f} | {d['median']:.3f} | "
                         f"{d['p75']:.3f} | {d['p90']:.3f} | {d['max']:.3f} |")
        g = groups(run, "rerank")
        q, u = sorted(g["question"]), sorted(g.get("unanswerable", []))
        if q and u:
            above = sum(1 for x in q if x > max(u))
            L += ["", f"คำถามจริงที่คะแนนสูงกว่า unanswerable ทุกข้อ (> {max(u):.3f}): {above}/{len(q)} · "
                      f"unanswerable ที่คะแนนต่ำกว่าคำถามจริงทุกข้อ (< {min(q):.3f}): "
                      f"{sum(1 for x in u if x < min(q))}/{len(u)}"]
    md = "\n".join(L) + "\n"
    print(md)
    (RESULTS / f"{args.out}.md").write_text(md, encoding="utf-8")
    (RESULTS / f"{args.out}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
