"""ทดลอง OCR กับหน้าที่ระบุ: extract_file + run_ocr แล้วพิมพ์ข้อความ เวลาต่อหน้า VRAM สูงสุด agreement

    python scripts/ocr_try.py eval/datasets/ONEREPORTKBANKT.PDF 2
    python scripts/ocr_try.py eval/datasets/ONEREPORTKBANKT.PDF 81-82 --out-dir eval/datasets/files/ocr_try

หน้าที่ด่านตรวจไม่ได้ส่ง OCR (status = ok) จะถูกบังคับเป็น needs_ocr เฉพาะในสคริปต์นี้เพื่อทดลอง (--no-force เพื่อปิด)
ข้อความเต็มของแต่ละหน้าเขียนลงไฟล์ <out-dir>/<ชื่อไฟล์>_p<หน้า>.md (ค่าเริ่มต้น eval/datasets/files/ocr_try ซึ่งไม่เข้า git)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.extraction.ocr import get_backend, render_page_images, run_ocr  # noqa: E402
from app.extraction.pipeline import extract_file  # noqa: E402
from app.models import UnitStatus  # noqa: E402

PREVIEW_LINES = 15


def parse_pages(value: str) -> tuple[int, int]:
    start, _, end = value.partition("-")
    pages = (int(start), int(end or start))
    if pages[0] < 1 or pages[1] < pages[0]:
        raise argparse.ArgumentTypeError("ช่วงหน้าต้องเป็น N หรือ N-M และเริ่มที่ 1 ขึ้นไป")
    return pages


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ทดลอง OCR รายหน้า")
    ap.add_argument("path")
    ap.add_argument("pages", type=parse_pages, nargs="?", default=None, help="N หรือ N-M (PDF); ภาพไม่ต้องระบุ")
    ap.add_argument("--backend", default=None, help="ค่าเริ่มต้นจาก config (OCR_BACKEND)")
    ap.add_argument("--out-dir", default="eval/datasets/files/ocr_try")
    ap.add_argument("--no-force", action="store_true", help="OCR เฉพาะหน้าที่ด่านตรวจส่งมาเอง")
    ap.add_argument("--lines", type=int, default=PREVIEW_LINES)
    ap.add_argument("--save-images", action="store_true", help="เขียนภาพที่ส่งให้โมเดล (PNG) ไว้ตรวจด้วยตา")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    path = Path(args.path)
    doc = extract_file(path, pages=args.pages)
    if doc.error:
        print(f"extract ล้มเหลว: {doc.error.code.value} — {doc.error.message}")
        return 1
    before = {u.index: u for u in doc.units}
    print(f"ไฟล์: {doc.filename}  ชนิด: {doc.detected_type.value}  unit: {doc.unit_count}")
    for u in doc.units:
        print(f"  ก่อน OCR {u.label}: {u.status.value}" + (f" — {u.reason}" if u.reason else ""))
    if not args.no_force:
        forced = [
            u.model_copy(update={"status": UnitStatus.NEEDS_OCR, "reason": "บังคับ OCR (ทดลองด้วย ocr_try.py)"})
            if u.status == UnitStatus.OK else u
            for u in doc.units
        ]
        doc = doc.model_copy(update={"units": forced})

    backend = get_backend(args.backend)
    peak = None
    t0 = time.perf_counter()
    out = run_ocr(doc, path, backend)
    wall = time.perf_counter() - t0
    try:
        from app.extraction.ocr_typhoon import peak_vram_bytes

        peak = peak_vram_bytes()
    except Exception:
        pass

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for u in out.units:
        page = u.position["page"]
        print(f"\n===== {u.label} =====")
        if before[u.index].status == UnitStatus.OK and args.no_force:
            print("(ไม่ได้ OCR — ด่านตรวจไม่ได้ส่งมา)")
            continue
        print(f"status: {u.status.value}  method: {u.method.value}  เวลา OCR: {u.signals.get('ocr:seconds', 0):.1f} s"
              f"  ทั้งหน้า: {u.elapsed_ms / 1000:.1f} s")
        if u.reason:
            print(f"reason: {u.reason}")
        recall = u.signals.get("ocr:text_layer_recall")
        precision = u.signals.get("ocr:text_layer_precision")
        layer = before[u.index].text.strip()
        fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
        print(f"ชั้นข้อความเดิม: {len(layer)} ตัว  recall: {fmt(recall)}  precision: {fmt(precision)}")
        print(f"figure: {int(u.signals.get('ocr:figure_count', 0))}  page_number: {u.signals.get('ocr:page_number')}"
              f" / {u.signals.get('ocr:page_number_2')}")
        images = render_page_images(path, doc.detected_type, page)
        print(f"ภาพที่ส่งโมเดล: {len(images)} ภาพ")
        if args.save_images:
            for i, img in enumerate(images, 1):
                (out_dir / f"{path.stem}_p{page}_part{i}.png").write_bytes(img)
        print(f"flags: {[f.value for f in u.quality.flags]}  letters: {u.signals.get('gate:letters')}  "
              f"broken_events: {u.signals.get('gate:broken_events')}")
        dest = out_dir / f"{path.stem}_p{page}.md"
        dest.write_text(u.text, encoding="utf-8")
        print(f"ข้อความเต็ม: {dest}")
        print(f"--- {args.lines} บรรทัดแรก ---")
        for line in u.text.splitlines()[: args.lines]:
            print(line[:300])
    print(f"\nเวลารวม run_ocr: {wall:.1f} s")
    if peak is not None:
        print(f"VRAM สูงสุด (torch allocated): {peak / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
