"""รูปมาตรฐานสำหรับเทียบข้อความ OCR (ฟังก์ชันเดียว ใช้ทั้ง scripts/ocr_truth_prepare.py และ scripts/ocr_truth_eval.py)

ใช้ normalizer กลางของแอปเป็นขั้นแรก (ไม่ใช่ normalizer ตัวที่สอง): ขั้นถัดไปเป็นแค่การลดรูปเพื่อเทียบ ไม่ใช้กับข้อความที่เก็บเข้าดัชนี
"""

import re

from app.thai.normalizer import normalize

_SCRIPT_DIGITS = str.maketrans({**{chr(0x2080 + i): str(i) for i in range(10)},
                                **{"⁰": "0", "¹": "1", "²": "2", "³": "3", **{chr(0x2074 + i): str(4 + i) for i in range(6)}}})
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_BULLET = re.compile(r"(?m)^[ \t]*(?:•[ \t]*|[+*\-][ \t]+)")  # • ตามด้วยตัวอักษรได้เลย; + * - ต้องมีช่องว่างตาม (กัน "-5" ถูกตัดเครื่องหมาย)
_MD = re.compile(r"[#*|]")
_WS = re.compile(r"\s+")


def canon(text: str) -> str:
    """normalizer กลาง -> ํา เป็น ำ -> ตัวห้อย/ตัวยก ₀-₉ ⁰-⁹ เป็นเลขปกติ -> เครื่องหมายคำพูดโค้งเป็นตรง
    -> ตัด bullet นำหน้าบรรทัด (• + * -) -> ตัด markdown (# * |) -> ตัดช่องว่างทั้งหมด"""
    t = normalize(text).replace("ํา", "ำ").translate(_SCRIPT_DIGITS).translate(_QUOTES)
    return _WS.sub("", _MD.sub("", _BULLET.sub("", t)))
