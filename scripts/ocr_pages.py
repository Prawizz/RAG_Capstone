"""รัน OCR ของหน้าที่ใช้วัดความแม่น (T06b) ทีละหน้า ผ่านแคช

    python scripts/ocr_pages.py typhoon-local        # ~12–15 นาทีบน RTX 3070 laptop (greedy, ตัดตรงร่อง, 200 dpi)
    python scripts/ocr_pages.py tesseract-local      # CPU ไม่กี่วินาทีต่อหน้า
    python scripts/ocr_pages.py typhoon-local --refresh   # ข้ามแคช (เขียนทับ)

รันทีละหน้า ห้ามรันขนาน (GPU หนัก) ผลเข้าแคช (data/ocr_cache หรือ OCR_CACHE_DIR) ใช้ต่อโดย ocr_truth_prepare.py
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.extraction.ocr import get_backend, recognize_page_cached  # noqa: E402
from app.extraction.pipeline import extract_file  # noqa: E402
from app.models import DetectedType  # noqa: E402

DATASETS = Path(__file__).resolve().parent.parent / "eval" / "datasets"
PAGES = [
    ("E_ONE_REPORT_CPALLT.PDF", 442), ("E_ONE_REPORT_CPALLT.PDF", 462),
    ("ONEREPORTKBANKT.PDF", 2), ("ONEREPORTKBANKT.PDF", 81), ("ONEREPORTKBANKT.PDF", 87),
    ("ONEREPORTPTTT.PDF", 43), ("ONEREPORTPTTT.PDF", 293),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OCR 7 หน้าที่ใช้วัดความแม่น (ผ่านแคช)")
    ap.add_argument("backend", help="typhoon-local | tesseract-local")
    ap.add_argument("--refresh", action="store_true", help="ไม่อ่านแคช แต่เขียนทับ")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    backend = get_backend(args.backend)
    hashes: dict[str, str] = {}
    t_all = time.perf_counter()
    for name, page_no in PAGES:
        path = DATASETS / name
        if name not in hashes:
            from app.extraction.pipeline import _hash_file

            hashes[name] = _hash_file(path)[0]
        t0 = time.perf_counter()
        res, page, hit = recognize_page_cached(
            backend, path, DetectedType.PDF, hashes[name], page_no, refresh_cache=args.refresh
        )
        print(f"{name} p{page_no}: {'แคช' if hit else 'รันจริง'}  {time.perf_counter() - t0:.1f} s"
              f" (เวลาที่เก็บไว้ {res.seconds:.1f} s)  ภาพ {len(page.images)}  ตัด {page.split_method}"
              f" x={page.split_x if page.split_x is None else round(page.split_x, 4)}  tokens={res.tokens}"
              f"  ตัวอักษร {len(res.text)}", flush=True)
    print(f"รวม {time.perf_counter() - t_all:.1f} s")
    try:
        from app.extraction.ocr_typhoon import peak_vram_bytes

        if args.backend == "typhoon-local":
            print(f"VRAM สูงสุด {peak_vram_bytes() / 2**30:.2f} GiB")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
