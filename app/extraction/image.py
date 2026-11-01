"""ไฟล์ภาพ (jpg / png): เอกสาร 1 หน้า ไม่มีชั้นข้อความ จึงเป็น needs_ocr ตั้งแต่ต้น

unit เดียว type=page, position {"page": 1}; method = ocr (วิธีที่ต้องใช้ — ยังไม่ได้รัน สถานะบอกว่ารออยู่)
ข้อความจริงมาจาก ``app/extraction/ocr.py::run_ocr`` เท่านั้น extract_file ไม่เรียก OCR
"""

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from app.extraction.errors import ExtractionError
from app.models import ErrorCode, ExtractionMethod, QualityReport, Unit, UnitStatus, UnitType

IMAGE_REASON = "ไฟล์ภาพไม่มีชั้นข้อความ — ต้อง OCR"


def load_oriented_image(path: str | Path) -> "Image.Image":
    """เปิดภาพ หมุนตาม EXIF Orientation (กล้อง/มือถือเก็บภาพตะแคงพร้อมแท็ก) แล้วแปลงเป็น RGB (โปร่งใส -> พื้นขาว)
    ถอดรหัสทั้งภาพ: ไฟล์ที่ตัดท้ายหรือเสียจะโยนที่นี่"""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im.load()
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            base = Image.new("RGB", rgba.size, (255, 255, 255))
            base.paste(rgba, mask=rgba.getchannel("A"))
            return base
        return im.convert("RGB")


@dataclass
class ImageResult:
    units: list[Unit] = field(default_factory=list)
    has_outline: bool = False
    producer: Optional[str] = None
    page_count: int = 1


def extract_image(path: str | Path) -> ImageResult:
    t0 = time.perf_counter()
    try:
        width, height = load_oriented_image(path).size
    except Exception as e:
        raise ExtractionError(ErrorCode.CORRUPT, f"เปิดไฟล์ภาพไม่ได้: {e}") from e
    unit = Unit(
        index=0,
        type=UnitType.PAGE,
        position={"page": 1},
        label="หน้า 1",
        status=UnitStatus.NEEDS_OCR,
        reason=IMAGE_REASON,
        method=ExtractionMethod.OCR,
        text="",
        text_hash=hashlib.sha256(b"").hexdigest(),
        quality=QualityReport(),
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        signals={"char_count": 0.0, "block_count": 0.0, "image_coverage": 1.0,
                 "image:width": float(width), "image:height": float(height)},
    )
    return ImageResult(units=[unit])
