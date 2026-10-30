"""วัดผล normalizer บน PDF จริง: จำนวนที่แก้รายกฎ, signals, หน้าที่โดนแก้มากที่สุด, ตัวอย่างก่อน/หลัง

    python scripts/normalize_report.py [--samples 20] [--out report.txt] [FILE.pdf ...]

ไม่ใส่ไฟล์ = 3 PDF ใน eval/datasets/ + mock_corpus.pdf
ข้อความก่อน normalize ได้จาก ``app.extraction.pdf.raw_blocks`` (จุดเดียวกับที่ pipeline ใช้) และกฎจาก
``app.thai.normalizer.RULES`` ไม่มี logic ดึงข้อความ/normalize ซ้ำในสคริปต์นี้
ตัวอย่างเก็บเฉพาะจุดที่แก้เฉพาะที่ (ส่วนที่เปลี่ยนไม่เกิน 24 ตัวอักษร) จากรอบแรกของการไล่กฎ
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402

from app.extraction.pdf import raw_blocks  # noqa: E402
from app.thai.normalizer import (  # noqa: E402
    NORMALIZER_VERSION,
    RULE_NAMES,
    RULES,
    SIGNAL_NAMES,
    normalize_with_report,
)

DEFAULT_FILES = [
    ROOT / "eval" / "datasets" / "E_ONE_REPORT_CPALLT.PDF",
    ROOT / "eval" / "datasets" / "ONEREPORTKBANKT.PDF",
    ROOT / "eval" / "datasets" / "ONEREPORTPTTT.PDF",
    ROOT / "mock_corpus.pdf",
]
CONTEXT = 15
MAX_CHANGED = 24


def vis(s: str) -> str:
    """แสดงอักขระมองไม่เห็นให้อ่านได้"""
    out = []
    for ch in s:
        o = ord(ch)
        if ch == "�":
            out.append("<FFFD>")
        elif ch == "\n":
            out.append("\\n")
        elif o < 0x20 or 0x7F <= o <= 0x9F or ch in "​‌‍﻿­ ":
            out.append(f"<U+{o:04X}>")
        else:
            out.append(ch)
    return "".join(out)


def diff_region(a: str, b: str) -> tuple[int, int, int]:
    """(เริ่ม, จบใน a, จบใน b) ของส่วนที่ต่างกัน โดยตัด prefix/suffix ร่วมออก"""
    i = 0
    n = min(len(a), len(b))
    while i < n and a[i] == b[i]:
        i += 1
    j = 0
    while j < n - i and a[len(a) - 1 - j] == b[len(b) - 1 - j]:
        j += 1
    return i, len(a) - j, len(b) - j


class Sampler:
    """reservoir sampling ต่อกฎ"""

    def __init__(self, n: int, seed: int = 0):
        self.n = n
        self.rng = random.Random(seed)
        self.seen: Counter = Counter()
        self.items: dict[str, list] = defaultdict(list)

    def offer(self, rule: str, item: tuple) -> None:
        self.seen[rule] += 1
        bucket = self.items[rule]
        if len(bucket) < self.n:
            bucket.append(item)
        else:
            k = self.rng.randrange(self.seen[rule])
            if k < self.n:
                bucket[k] = item


def collect_samples(sampler: Sampler, text: str, where: str) -> None:
    for name, rule in RULES:
        new, n = rule(text)
        if n and new != text:
            i, ea, eb = diff_region(text, new)
            if ea - i <= MAX_CHANGED and eb - i <= MAX_CHANGED:
                lo = max(0, i - CONTEXT)
                before = text[lo : ea + CONTEXT]
                after = new[lo : eb + CONTEXT]
                sampler.offer(name, (where, before, after))
        text = new


def run(files: list[Path], n_samples: int) -> str:
    lines: list[str] = [f"# normalizer {NORMALIZER_VERSION}", ""]
    sampler = Sampler(n_samples)
    all_fixes: Counter = Counter()
    all_sigs: Counter = Counter()
    per_file: dict[str, tuple[Counter, Counter, int]] = {}
    page_scores: list[tuple[int, str, int]] = []

    for path in files:
        fixes: Counter = Counter()
        sigs: Counter = Counter()
        with pymupdf.open(str(path)) as doc:
            pages = doc.page_count
            for page in doc:
                page_total = 0
                for rb in raw_blocks(page):
                    res = normalize_with_report(rb.text)
                    fixes.update(res.fixes)
                    sigs.update(res.signals)
                    page_total += res.total_fixes
                    collect_samples(sampler, rb.text, f"{path.name} หน้า {page.number + 1}")
                page_scores.append((page_total, path.name, page.number + 1))
        per_file[path.name] = (fixes, sigs, pages)
        all_fixes.update(fixes)
        all_sigs.update(sigs)

    names = list(per_file)
    width = max(len(n) for n in names + ["รวม"]) + 2
    lines.append("## จำนวนที่แก้รายกฎ")
    lines.append("rule".ljust(20) + "".join(n.rjust(width) for n in names + ["รวม"]))
    for r in RULE_NAMES:
        row = [per_file[n][0][r] for n in names] + [all_fixes[r]]
        lines.append(r.ljust(20) + "".join(f"{v:,}".rjust(width) for v in row))
    lines.append("")
    lines.append("## signals (นับ ไม่แก้)")
    lines.append("signal".ljust(24) + "".join(n.rjust(width) for n in names + ["รวม"]))
    for s in SIGNAL_NAMES:
        row = [per_file[n][1][s] for n in names] + [all_sigs[s]]
        lines.append(s.ljust(24) + "".join(f"{v:,}".rjust(width) for v in row))
    lines.append("")
    lines.append("## 10 หน้าที่โดนแก้มากที่สุด (จำนวน fixes รวมทุกกฎ)")
    for total, fname, pno in sorted(page_scores, reverse=True)[:10]:
        lines.append(f"{total:>8,}  {fname} หน้า {pno}")
    lines.append("")
    lines.append(f"## ตัวอย่างก่อน/หลัง (กฎละไม่เกิน {n_samples} จุด, บริบท ±{CONTEXT} ตัวอักษร)")
    for r in RULE_NAMES:
        items = sampler.items.get(r, [])
        lines.append("")
        lines.append(f"### {r}  (จุดที่แก้เฉพาะที่ที่เจอ: {sampler.seen[r]:,}; แสดง {len(items)})")
        for where, before, after in items:
            lines.append(f"- {where}")
            lines.append(f"    ก่อน: {vis(before)}")
            lines.append(f"    หลัง: {vis(after)}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="รายงานผล Thai normalizer บน PDF จริง")
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--samples", type=int, default=20, help="ตัวอย่างต่อกฎ (ค่าเริ่มต้น 20)")
    ap.add_argument("--out", type=Path, help="เขียนลงไฟล์ (ไม่ใส่ = พิมพ์ออกจอ)")
    args = ap.parse_args(argv)

    report = run(args.files or DEFAULT_FILES, args.samples)
    if args.out:
        args.out.write_text(report, encoding="utf-8", newline="\n")
        print(f"เขียนรายงานที่ {args.out}")
    else:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
