"""แคชผล OCR ราย "หน้า" (T06b)

เหตุผล: การคำนวณบน GPU ให้ผลต่างกันเล็กน้อยระหว่างรอบ (เช่นการแบ่งบรรทัด) และ OCR ช้า (นาทีต่อหน้า)
แคชทำให้ไฟล์เดิมได้ข้อความเดิมเป๊ะและไม่ต้องรอ OCR ใหม่

key = SHA-256 ของ JSON (เรียง key) ที่ประกอบด้วย
  content_hash ของไฟล์, เลขหน้า, ชื่อ backend, revision ของโมเดล, dpi (PDF; ภาพ = None),
  วิธีตัดหน้า (none / gutter / center) + ตำแหน่งตัด (ปัด 4 ตำแหน่ง), และพารามิเตอร์ของ backend
  (``backend.cache_params()`` เช่น do_sample / max_new_tokens / ด้านยาวของภาพ — ค่าใดเปลี่ยน key เปลี่ยน)
  ชิ้นภาพของการ OCR ซ้ำแบบแบ่งบน/ล่าง (``ocr.py`` resplit) ใส่ ``region`` = (x0, y0, x1, y1) สัดส่วนของหน้า ปัด 4 ตำแหน่ง
  และ split_method ใหม่ ("<วิธีตัดเดิม>+rows") — ไม่มี region = ไม่มี key นี้ใน fields (key ของผลเดิมไม่เปลี่ยน)
  ชิ้นที่หมุนภาพให้ตั้งตรงก่อน OCR ใส่ ``rotation`` = องศาที่หมุนตามเข็มนาฬิกา (90 / 180 / 270); ไม่หมุน = ไม่มี key นี้
ค่าที่เก็บ: ผลดิบของ backend ทั้งหน้า (markdown / ข้อความ, ข้อมูลเพิ่มเติม เช่นบรรทัด+bbox ของ Tesseract),
  โทเคน, เวลา (วินาทีที่ใช้จริงตอนรันครั้งแรก) พร้อม key fields ไว้ตรวจย้อนหลัง

โฟลเดอร์: env ``OCR_CACHE_DIR`` หรือค่าเริ่มต้น ``<repo>/data/ocr_cache`` (``data/`` อยู่ใน .gitignore)
ข้ามแคช: ``run_ocr(..., use_cache=False)`` (ไม่อ่านและไม่เขียน) หรือ ``refresh_cache=True`` (ไม่อ่าน แต่เขียนทับ)
ไฟล์แคชเสีย/อ่านไม่ได้ = เหมือนไม่มีแคช (คำนวณใหม่)
"""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

CACHE_VERSION = 1
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ocr_cache"


def cache_dir() -> Path:
    env = os.environ.get("OCR_CACHE_DIR")
    return Path(env) if env else DEFAULT_CACHE_DIR


def make_key_fields(*, content_hash: str, page: int, backend: str, revision: str, dpi: Optional[int],
                    split_method: str, split_x: Optional[float], params: dict[str, Any],
                    region: Optional[tuple[float, float, float, float]] = None,
                    rotation: int = 0) -> dict[str, Any]:
    fields = {
        "v": CACHE_VERSION,
        "content_hash": content_hash,
        "page": page,
        "backend": backend,
        "revision": revision,
        "dpi": dpi,
        "split_method": split_method,
        "split_x": None if split_x is None else round(split_x, 4),
        "params": dict(sorted(params.items())),
    }
    if region is not None:
        fields["region"] = [round(float(v), 4) for v in region]
    if rotation:
        fields["rotation"] = int(rotation)
    return fields


def make_key(fields: dict[str, Any]) -> str:
    blob = json.dumps(fields, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class OcrCache:
    def __init__(self, directory: Optional[Path] = None):
        self.dir = Path(directory) if directory else cache_dir()

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Optional[dict[str, Any]]:
        try:
            data = json.loads(self._path(key).read_text(encoding="utf-8"))
            if data.get("key") == key and isinstance(data.get("text"), str):
                return data
        except Exception:
            pass
        return None

    def put(self, key: str, fields: dict[str, Any], result) -> None:
        """``result`` = OcrResult; เขียนแบบ atomic (ไฟล์ชั่วคราวแล้ว rename) ล้มเหลวเงียบ (แคชเป็นแค่ตัวช่วย)"""
        data = {"key": key, "fields": fields, **asdict(result)}
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self._path(key))
        except OSError:
            pass
