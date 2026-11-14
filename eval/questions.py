"""ตัวตรวจ ``eval/datasets/questions.jsonl`` ตาม ``eval/datasets/questions.schema.json`` (เขียนตรวจเอง ไม่เพิ่ม dependency)

    python -m eval.questions                                   # ตรวจ eval/datasets/questions.jsonl
    python -m eval.questions path/to/questions.jsonl

ชุดนี้ใช้วัดส่วนเสริมที่ออกแบบจากข้อที่พลาดในชุด dev (query_expansion / salience rerank ฯลฯ) — ห้ามวัดบนชุด dev เพราะวัดวน
แถวละ object: ``question`` + ``type`` (question | unanswerable | chitchat) · type = question ต้องมี ``file`` (filename ตามที่ API
แสดง) และ ``pages`` (เลขหน้า PDF เริ่มที่ 1, หลายหน้าได้, ไม่ซ้ำ) · type อื่นห้ามมี file/pages · ``id`` / ``note`` ไม่บังคับ
ไฟล์ว่าง = error (ยังไม่มีคำถาม) — ไม่ถือว่าผ่าน
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "eval" / "datasets" / "questions.jsonl"
SCHEMA_PATH = ROOT / "eval" / "datasets" / "questions.schema.json"
TYPES = ("question", "unanswerable", "chitchat")
KEYS = {"id", "question", "type", "file", "pages", "note"}


class QuestionsError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("\n".join(errors))
        self.errors = errors


def validate_row(obj: Any, where: str) -> list[str]:
    if not isinstance(obj, dict):
        return [f"{where}: ต้องเป็น JSON object"]
    errs = [f"{where}: ไม่รู้จักฟิลด์ {k!r}" for k in sorted(set(obj) - KEYS)]
    q = obj.get("question")
    if not isinstance(q, str) or not q.strip():
        errs.append(f"{where}: question ต้องเป็นข้อความไม่ว่าง")
    t = obj.get("type")
    if t not in TYPES:
        errs.append(f"{where}: type ต้องเป็น {' | '.join(TYPES)} (ได้ {t!r})")
    if "id" in obj and (isinstance(obj["id"], bool) or not isinstance(obj["id"], (str, int))):
        errs.append(f"{where}: id ต้องเป็นข้อความหรือจำนวนเต็ม")
    if "note" in obj and not isinstance(obj["note"], str):
        errs.append(f"{where}: note ต้องเป็นข้อความ")
    file, pages = obj.get("file"), obj.get("pages")
    if t == "question":
        if not isinstance(file, str) or not file.strip():
            errs.append(f"{where}: type question ต้องมี file (ชื่อไฟล์ตามที่ API แสดง)")
        if not isinstance(pages, list) or not pages:
            errs.append(f"{where}: type question ต้องมี pages (รายการเลขหน้า อย่างน้อย 1)")
        else:
            if any(isinstance(p, bool) or not isinstance(p, int) or p < 1 for p in pages):
                errs.append(f"{where}: pages ต้องเป็นจำนวนเต็ม >= 1")
            elif len(set(pages)) != len(pages):
                errs.append(f"{where}: pages ซ้ำกัน")
    elif t in TYPES:
        if file is not None or pages not in (None, []):
            errs.append(f"{where}: type {t} ห้ามมี file / pages")
    return errs


def load(path: Path = DEFAULT_PATH) -> list[dict]:
    """อ่าน + ตรวจทั้งไฟล์ (รวบรวม error ทุกบรรทัด) — ผ่าน = รายการแถว (มี ``id`` เสมอ: ไม่ใส่ = ``line<N>``)"""
    if not path.is_file():
        raise QuestionsError([f"ไม่พบ {path}"])
    rows, errs, seen = [], [], {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path.name}:{n}"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            errs.append(f"{where}: JSON ไม่ถูกต้อง ({e.msg})")
            continue
        row_errs = validate_row(obj, where)
        errs += row_errs
        if row_errs:
            continue
        rid = str(obj.get("id", f"line{n}"))
        if rid in seen:
            errs.append(f"{where}: id {rid!r} ซ้ำกับบรรทัด {seen[rid]}")
            continue
        seen[rid] = n
        rows.append({**obj, "id": rid, "pages": obj.get("pages") or []})
    if not rows and not errs:
        errs.append(f"{path}: ยังไม่มีคำถาม (ไฟล์ว่าง) — เติมตาม {SCHEMA_PATH.name} ก่อนวัด")
    if errs:
        raise QuestionsError(errs)
    return rows


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else DEFAULT_PATH
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        rows = load(path)
    except QuestionsError as e:
        print("\n".join(e.errors))
        return 1
    counts = {t: sum(1 for r in rows if r["type"] == t) for t in TYPES}
    print(f"{path}: ผ่าน {len(rows)} ข้อ ({', '.join(f'{k} {v}' for k, v in counts.items())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
