"""เทียบผล normalize กับ truth ที่คนเติมใน eval/datasets/thai_normalization.jsonl

    python scripts/check_truth.py [--file PATH]

พิมพ์ตารางรายคู่ (เฉพาะคู่ที่มีช่อง truth): source, rule, ผล และสรุปจำนวนแต่ละแบบ
exit code 1 ถ้ามีคู่ที่ได้ผล fail
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.truth_check import (  # noqa: E402
    EVIDENCE,
    EXACT,
    FAIL,
    FAIL_MESSAGE,
    NEEDS_REPAIR,
    NO_TRUTH,
    classify_case,
)

DEFAULT_FILE = ROOT / "eval" / "datasets" / "thai_normalization.jsonl"
LABELS = {
    EXACT: "exact",
    EVIDENCE: "evidence",
    NEEDS_REPAIR: "needs_repair",
    FAIL: "fail",
    NO_TRUTH: "ยังไม่มี truth",
}


def load(path: Path) -> list[dict]:
    # แยกบรรทัดด้วย "\n" เท่านั้น (splitlines จะตัดที่ \x85 /   ที่อยู่ในข้อมูล)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="เทียบ normalize(input) กับ truth ที่คนเติม")
    ap.add_argument("--file", type=Path, default=DEFAULT_FILE)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cases = load(args.file)
    rows = [c for c in cases if "truth" in c]
    counts: Counter = Counter()
    print(f"{'#':>3}  {'ผล':<15} {'rule':<32} source")
    for i, c in enumerate(rows, 1):
        status, out = classify_case(c)
        counts[status] += 1
        print(f"{i:>3}  {LABELS[status]:<15} {c['rule']:<32} {c['source']}")
        if status == FAIL:
            print(f"       {FAIL_MESSAGE}")
            print(f"       input: {c['input']!r}\n       out:   {out!r}\n       truth: {c['truth']!r}")

    print()
    print(f"คู่ที่มีช่อง truth: {len(rows)} (จากทั้งหมด {len(cases)}; synthetic ไม่มีช่อง truth)")
    for key in (EXACT, EVIDENCE, NEEDS_REPAIR, FAIL, NO_TRUTH):
        print(f"  {LABELS[key]:<15} {counts[key]}")
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
