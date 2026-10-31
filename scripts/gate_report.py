"""รายงานผลด่านตรวจคุณภาพบน PDF จริง

    python scripts/gate_report.py [FILE.pdf ...] [--labels eval/datasets/ocr_pages.jsonl]

ไม่ใส่ไฟล์ = 3 PDF ใน eval/datasets/ รายไฟล์: จำนวน ok / needs_ocr แยกตามเหตุผล, หน้าที่โดนกฎระดับไฟล์,
การกระจายของ dictionary_ratio (วัดอย่างเดียว เฉพาะเมื่อใส่ --dictionary เพราะช้า) และถ้ามีป้ายจากคนแล้ว (label ไม่ใช่ null)
จะแสดงตาราง ป้ายคน x ผลด่านตรวจ พร้อมรายการหน้าที่ไม่ตรงกัน
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.extraction.pipeline import extract_file  # noqa: E402
from app.extraction.quality import GATE_KEYS, MIN_THAI_CHARS, dictionary_ratio, gate_key  # noqa: E402

DATASETS = ROOT / "eval" / "datasets"
DEFAULT_FILES = ("E_ONE_REPORT_CPALLT.PDF", "ONEREPORTKBANKT.PDF", "ONEREPORTPTTT.PDF")
DEFAULT_LABELS = DATASETS / "ocr_pages.jsonl"
LABELS = ("ok", "broken", "skip")


def load_labels(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    return [r for r in rows if r.get("label") is not None]


def percentiles(values: list[float]) -> str:
    v = sorted(values)
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]  # noqa: E731
    return f"n={len(v)} min={v[0]:.3f} p10={q(.1):.3f} median={q(.5):.3f} p90={q(.9):.3f} max={v[-1]:.3f}"


def report_file(name: str, doc, with_dictionary: bool = False) -> None:
    keys = Counter(gate_key(u) for u in doc.units)
    print(f"\n=== {name}  ({doc.unit_count} หน้า)")
    for k in list(GATE_KEYS) + [k for k in keys if k not in GATE_KEYS]:
        if keys[k]:
            print(f"  {k:<34}{keys[k]:>5}")
    swept = [u.position["page"] for u in doc.units if gate_key(u) == "needs_ocr:document_font_broken"]
    if swept:
        print(f"  หน้าที่โดนกฎระดับไฟล์ ({len(swept)}): {', '.join(map(str, swept[:40]))}{' ...' if len(swept) > 40 else ''}")
    if not with_dictionary:
        return
    groups = defaultdict(list)
    for u in doc.units:
        if u.signals.get("gate:thai_chars", 0) >= MIN_THAI_CHARS:
            groups[gate_key(u)].append(dictionary_ratio(u.text))
    for k, vals in groups.items():
        print(f"  dictionary_ratio [{k}] {percentiles(vals)}")


def report_labels(labels: list[dict], docs: dict) -> None:
    print("\n=== ป้ายจากคน x ผลด่านตรวจ")
    table: dict[str, Counter] = defaultdict(Counter)
    mismatches = []
    for r in labels:
        doc = docs.get(r["file"])
        if doc is None:
            print(f"  (ข้าม {r['file']}: ไม่ได้รันไฟล์นี้)")
            continue
        unit = doc.units[r["page"] - 1]
        key = gate_key(unit)
        result = "ok" if key.startswith("ok") else "needs_ocr"
        table[r["label"]][result] += 1
        if (r["label"] == "ok" and result == "needs_ocr") or (r["label"] == "broken" and result == "ok"):
            mismatches.append((r, key, unit.reason))
    print(f"  {'ป้าย':<10}{'ok':>8}{'needs_ocr':>12}")
    for lab in LABELS:
        print(f"  {lab:<10}{table[lab]['ok']:>8}{table[lab]['needs_ocr']:>12}")
    print(f"\n  หน้าที่ไม่ตรงกัน ({len(mismatches)}): ป้าย ok แต่ gate ส่ง OCR (เสียค่าใช้จ่าย) / ป้าย broken แต่ gate ปล่อยผ่าน (อันตราย)")
    for r, key, reason in mismatches:
        print(f"  - {r['file']} หน้า {r['page']}: ป้าย {r['label']} / gate {key}" + (f" — {reason}" if reason else "")
              + (f" [{r['note']}]" if r.get("note") else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="รายงานผลด่านตรวจคุณภาพ")
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--dictionary", action="store_true", help="คำนวณและแสดงการกระจาย dictionary_ratio (ช้า)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    paths = args.files or [DATASETS / n for n in DEFAULT_FILES]
    docs = {}
    for p in paths:
        doc = extract_file(p)
        if doc.error:
            print(f"{p.name}: error {doc.error.code.value} — {doc.error.message}")
            continue
        docs[p.name] = doc
        report_file(p.name, doc, args.dictionary)
    labels = load_labels(args.labels)
    if labels:
        report_labels(labels, docs)
    else:
        print(f"\n(ยังไม่มีป้ายจากคนใน {args.labels.name} — ดูวิธีติดป้ายใน README)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
