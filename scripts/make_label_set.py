"""สร้างชุดหน้าให้คนติดป้าย (ไว้ตรวจเกณฑ์ด่านตรวจ ไม่วัดวน)

    python scripts/make_label_set.py [--force]

เลือกราว 40 หน้า (seed ตายตัว ได้ชุดเดิมทุกครั้ง) เขียน ``eval/datasets/ocr_pages.jsonl`` เป็น
``{"file", "page", "label": null, "note": ""}`` และสร้าง HTML ไฟล์เดียวที่ ``eval/datasets/files/label_pages.html``
แสดงทีละหน้า: ภาพหน้า PDF คู่กับข้อความที่ดึงได้หลัง normalize ส่วนผลของด่านตรวจซ่อนไว้ (กดดูได้)
เพื่อไม่ให้คนติดป้ายตามโปรแกรม

ป้าย: "ok" = ข้อความตรงกับ PDF พอจะใช้ค้นและอ้างอิงได้ (ช่องว่าง/ลำดับไม่ต้องสน)
      "broken" = เจอคำที่ผิดจาก PDF ทั้งที่ต้นฉบับไม่ได้ผิด · "skip" = ไม่แน่ใจ
ไม่เขียนทับ ocr_pages.jsonl ที่มีข้อมูลอยู่แล้ว (ป้ายที่คนติดไว้) ยกเว้นสั่ง --force
"""

import argparse
import base64
import html
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402

from app.extraction.pipeline import extract_file  # noqa: E402
from app.extraction.quality import gate_key  # noqa: E402

DATASETS = ROOT / "eval" / "datasets"
JSONL = DATASETS / "ocr_pages.jsonl"
HTML = DATASETS / "files" / "label_pages.html"
FILES = ("E_ONE_REPORT_CPALLT.PDF", "ONEREPORTKBANKT.PDF", "ONEREPORTPTTT.PDF")
CPALL, KBANK, PTT = FILES
SEED = 20260131
DPI = 80

BORDERLINE = {PTT: [43, 152, 293], CPALL: [298, 302]}
KBANK_LOW_BROKEN = 10
RANDOM_PICKS = {KBANK: 5, PTT: 5, CPALL: 5}
SCAN_LIKE_PICKS = 3
LOW_TEXT_PICKS = 3


def _thai(u) -> float:
    return u.signals.get("gate:thai_chars", 0)


def select(docs: dict) -> list[tuple[str, int]]:
    rng = random.Random(SEED)
    chosen: list[tuple[str, int]] = []

    def add(name, page):
        if (name, page) not in chosen:
            chosen.append((name, page))

    kb = sorted(
        (u for u in docs[KBANK].units if _thai(u) >= 100),
        key=lambda u: (u.signals["gate:broken_events"], u.position["page"]),
    )
    for u in kb[:KBANK_LOW_BROKEN]:
        add(KBANK, u.position["page"])
    for name, pages in BORDERLINE.items():
        for p in pages:
            add(name, p)

    def pick(pool, n):
        pool = sorted((n_, p) for n_, p in pool if (n_, p) not in chosen)
        for item in rng.sample(pool, min(n, len(pool))):
            add(*item)

    for name, n in RANDOM_PICKS.items():
        pool = [(name, u.position["page"]) for u in docs[name].units
                if _thai(u) >= 100 and gate_key(u) in ("ok", "needs_ocr:broken", "needs_ocr:document_font_broken")]
        pick(pool, n)
    pick([(CPALL, u.position["page"]) for u in docs[CPALL].units if gate_key(u) == "needs_ocr:scan_like"],
         SCAN_LIKE_PICKS)
    pick([(name, u.position["page"]) for name, d in docs.items() for u in d.units
          if gate_key(u) == "ok:low_text"], LOW_TEXT_PICKS)
    rng.shuffle(chosen)  # ไม่ให้ลำดับในหน้าเว็บบอกกลุ่ม
    return chosen


def render_html(rows, docs) -> str:
    parts = [
        "<!doctype html><html lang='th'><meta charset='utf-8'><title>ติดป้ายหน้า PDF</title>",
        "<style>body{font-family:sans-serif;max-width:1400px;margin:auto;padding:12px}"
        ".page{border:1px solid #888;margin:24px 0;padding:12px}"
        ".cols{display:flex;gap:16px}.cols>div{flex:1;min-width:0}img{max-width:100%}"
        "pre{white-space:pre-wrap;background:#f4f4f4;padding:8px;max-height:900px;overflow:auto}"
        "@media(prefers-color-scheme:dark){body{background:#111;color:#eee}pre{background:#222}}</style>",
        f"<h1>ติดป้าย {len(rows)} หน้า</h1>"
        "<p>ดูภาพเทียบกับข้อความ แล้วใส่ label ใน ocr_pages.jsonl: ok / broken / skip "
        "(ช่องว่างและลำดับไม่ต้องสน) ผลของด่านตรวจซ่อนไว้ กดดูหลังติดป้ายแล้วเท่านั้น</p>",
    ]
    open_docs: dict[str, pymupdf.Document] = {}
    try:
        for i, (name, page_no) in enumerate(rows, 1):
            if name not in open_docs:
                open_docs[name] = pymupdf.open(str(DATASETS / name))
            pix = open_docs[name][page_no - 1].get_pixmap(dpi=DPI)
            img = base64.b64encode(pix.tobytes("jpg", jpg_quality=70)).decode()
            unit = docs[name].units[page_no - 1]
            gate = f"{gate_key(unit)}" + (f" — {unit.reason}" if unit.reason else "")
            parts.append(
                f"<div class='page'><h2>#{i} {html.escape(name)} หน้า {page_no}</h2><div class='cols'>"
                f"<div><img alt='หน้า {page_no}' src='data:image/jpeg;base64,{img}'></div>"
                f"<div><pre>{html.escape(unit.text) or '(ไม่มีข้อความ)'}</pre>"
                f"<details><summary>ผลของด่านตรวจ (กดดูหลังติดป้าย)</summary>{html.escape(gate)}</details>"
                "</div></div></div>"
            )
    finally:
        for d in open_docs.values():
            d.close()
    parts.append("</html>")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="สร้างชุดหน้าให้คนติดป้าย")
    ap.add_argument("--force", action="store_true", help="เขียนทับ ocr_pages.jsonl ที่มีข้อมูลอยู่")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if JSONL.exists() and JSONL.read_text(encoding="utf-8").strip() and not args.force:
        print(f"{JSONL} มีข้อมูลอยู่แล้ว (อาจมีป้ายที่คนติดไว้) ไม่เขียนทับ — ใช้ --force ถ้าต้องการ")
        return 1

    docs = {}
    for name in FILES:
        print(f"อ่าน {name} ...")
        docs[name] = extract_file(DATASETS / name)
        if docs[name].error:
            print(f"error: {docs[name].error.message}")
            return 1
    rows = select(docs)
    JSONL.write_text(
        "".join(json.dumps({"file": n, "page": p, "label": None, "note": ""}, ensure_ascii=False) + "\n"
                for n, p in rows),
        encoding="utf-8", newline="\n",
    )
    HTML.parent.mkdir(parents=True, exist_ok=True)
    HTML.write_text(render_html(rows, docs), encoding="utf-8", newline="\n")
    print(f"เขียน {len(rows)} หน้า -> {JSONL}\nHTML -> {HTML} ({HTML.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
