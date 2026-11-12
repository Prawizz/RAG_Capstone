"""export สัญญา API v1: ``contract/openapi.json`` + ``contract/examples/<endpoint>_<case>.json``

    python scripts/export_openapi.py           # เขียนไฟล์
    python scripts/export_openapi.py --check   # ตรวจว่าไฟล์ตรงกับโค้ด (ไม่ตรง = exit 1) ใช้ใน test/CI

ตัวอย่างมาจาก ``app.api.schemas.EXAMPLES`` ที่เดียว (ตัวเดียวกับที่ฝังใน OpenAPI) — ทุกตัวถูก validate กับ model ก่อนเขียน
ไฟล์ตัวอย่างที่ไม่มีใน EXAMPLES แล้วถูกลบ (โฟลเดอร์ examples เป็นของสคริปต์นี้ทั้งหมด)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.api.main import create_app  # noqa: E402
from app.api.schemas import EXAMPLES  # noqa: E402

CONTRACT_DIR = ROOT / "contract"


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def render() -> dict[str, str]:
    """path (relative to contract/) -> เนื้อไฟล์"""
    out = {"openapi.json": dump(create_app().openapi())}
    for name, (model, payload) in sorted(EXAMPLES.items()):
        model.model_validate(payload)  # ตัวอย่างที่ผิด schema ห้ามออกไป
        out[f"examples/{name}.json"] = dump(payload)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="ตรวจอย่างเดียว ไม่เขียน")
    ap.add_argument("--out", type=Path, default=CONTRACT_DIR)
    args = ap.parse_args(argv)

    files = render()
    existing = {p.relative_to(args.out).as_posix() for p in (args.out / "examples").glob("*.json")} \
        if (args.out / "examples").is_dir() else set()
    stale = sorted(existing - {k for k in files if k.startswith("examples/")})
    if args.check:
        bad = [k for k, v in files.items()
               if not (args.out / k).is_file() or (args.out / k).read_text(encoding="utf-8") != v]
        for k in bad + stale:
            print(f"ไม่ตรงกับโค้ด: contract/{k}")
        if bad or stale:
            print("รัน python scripts/export_openapi.py แล้ว commit ไฟล์ใน contract/")
            return 1
        print(f"contract/ ตรงกับโค้ด ({len(files)} ไฟล์)")
        return 0
    for k, v in files.items():
        path = args.out / k
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(v, encoding="utf-8", newline="\n")
    for k in stale:
        (args.out / k).unlink()
    print(f"เขียน {len(files)} ไฟล์ใน {args.out} (ลบตัวอย่างเก่า {len(stale)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
