"""รัน extraction กับไฟล์เดียวแล้วพิมพ์สรุป

    python scripts/extract_cli.py path/to/file.pdf [--out result.json] [--pages 1-20]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.extraction.pipeline import extract_file  # noqa: E402
from app.extraction.quality import gate_summary  # noqa: E402


def parse_pages(value: str) -> tuple[int, int]:
    try:
        start, _, end = value.partition("-")
        pages = (int(start), int(end or start))
    except ValueError:
        raise argparse.ArgumentTypeError("รูปแบบต้องเป็น N หรือ N-M เช่น 1-20")
    if pages[0] < 1 or pages[1] < pages[0]:
        raise argparse.ArgumentTypeError("ช่วงหน้าต้องเริ่มที่ 1 ขึ้นไปและจบไม่น้อยกว่าหน้าเริ่ม")
    return pages


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract ไฟล์เป็น Document (JSON)")
    ap.add_argument("path")
    ap.add_argument("--out", help="ไฟล์ JSON ปลายทาง")
    ap.add_argument("--pages", type=parse_pages, help="ช่วงหน้า เช่น 1-20 (PDF)")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    t0 = time.perf_counter()
    doc = extract_file(args.path, pages=args.pages)
    total = time.perf_counter() - t0

    print(f"ไฟล์: {doc.filename}")
    print(f"ชนิด: {doc.detected_type.value if doc.detected_type else '-'}")
    print(f"สถานะ: {doc.status.value}")
    if doc.error:
        print(f"error: {doc.error.code.value} — {doc.error.message} (retriable={doc.error.retriable})")
    else:
        times = [u.elapsed_ms for u in doc.units]
        print(f"จำนวนหน้าในไฟล์: {doc.page_equivalent}  unit ที่ได้: {doc.unit_count}")
        print("unit ต่อสถานะ: " + "  ".join(f"{k}={v}" for k, v in doc.unit_summary.items()))
        print(
            f"เวลารวม: {total:.2f} s  เฉลี่ยต่อหน้า: {sum(times) / len(times):.1f} ms"
            f"  สูงสุดต่อหน้า: {max(times):.1f} ms"
        )
        print("ด่านตรวจคุณภาพ: " + "  ".join(f"{k}={v}" for k, v in gate_summary(doc.units).items()))
        print("  (ok = ใช้ข้อความที่ดึงได้ต่อ · needs_ocr = ต้องส่ง OCR · ดูเหตุผลรายหน้าด้านล่าง)")
        for u in doc.units:
            if u.status.value != "ok":
                print(f"  {u.label}: {u.status.value} — {u.reason}")

    if args.out:
        Path(args.out).write_text(doc.model_dump_json(indent=2), encoding="utf-8", newline="\n")
        print(f"เขียน JSON ที่ {args.out}")
    return 1 if doc.error else 0


if __name__ == "__main__":
    sys.exit(main())
