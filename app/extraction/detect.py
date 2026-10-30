"""ตรวจชนิดไฟล์จาก magic bytes (ห้ามดูนามสกุล — CLAUDE.md ข้อ 2)

- ``%PDF-`` ภายใน 1024 ไบต์แรก -> pdf (ยอมให้มีขยะนำหน้า)
- ZIP -> เปิดดูข้างใน: ``word/`` docx, ``ppt/`` pptx, ``xl/`` xlsx, อย่างอื่น unsupported_type
- OLE2 -> ดูชื่อ stream: WordDocument doc, PowerPoint Document ppt, Workbook/Book xls
  ถ้ามี EncryptionInfo / EncryptedPackage -> encrypted (docx/xlsx/pptx ที่ตั้งรหัสผ่านถูกห่อเป็น OLE2)
- อย่างอื่น -> unsupported_type
- ZIP/OLE2 ที่โครงสร้างเสีย -> corrupt
"""

import zipfile
from pathlib import Path

import olefile

from app.extraction.errors import ExtractionError
from app.models import DetectedType, ErrorCode

PDF_MAGIC = b"%PDF-"
PDF_SCAN_BYTES = 1024
ZIP_MAGIC = b"PK\x03\x04"
OLE_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")

_ZIP_PREFIXES = (
    ("word/", DetectedType.DOCX),
    ("ppt/", DetectedType.PPTX),
    ("xl/", DetectedType.XLSX),
)
_OLE_STREAMS = (
    ("WordDocument", DetectedType.DOC),
    ("PowerPoint Document", DetectedType.PPT),
    ("Workbook", DetectedType.XLS),
    ("Book", DetectedType.XLS),
)
_OLE_ENCRYPTED = ("EncryptionInfo", "EncryptedPackage")


def detect(path: str | Path) -> DetectedType:
    """คืนชนิดไฟล์ หรือโยน ``ExtractionError`` (unsupported_type / encrypted / corrupt)"""
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(PDF_SCAN_BYTES)

    # ZIP/OLE ดูที่ต้นไฟล์เท่านั้น; PDF ยอมให้มีขยะนำหน้าได้ใน 1024 ไบต์แรก (ตามมาตรฐาน)
    if head.startswith(ZIP_MAGIC):
        return _detect_zip(path)
    if head.startswith(OLE_MAGIC):
        return _detect_ole(path)
    if PDF_MAGIC in head:
        return DetectedType.PDF
    raise ExtractionError(
        ErrorCode.UNSUPPORTED_TYPE, "ไม่รู้จักชนิดไฟล์ (magic bytes ไม่ตรงกับชนิดที่รองรับ)"
    )


def _detect_zip(path: Path) -> DetectedType:
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
    except zipfile.BadZipFile as e:
        raise ExtractionError(ErrorCode.CORRUPT, f"ไฟล์ ZIP เสียหาย: {e}") from e
    for prefix, detected in _ZIP_PREFIXES:
        if any(n.startswith(prefix) for n in names):
            return detected
    raise ExtractionError(ErrorCode.UNSUPPORTED_TYPE, "เป็นไฟล์ ZIP ที่ไม่ใช่ docx/pptx/xlsx")


def _detect_ole(path: Path) -> DetectedType:
    try:
        ole = olefile.OleFileIO(str(path))
    except Exception as e:  # olefile โยนได้หลายแบบเมื่อโครงสร้างเสีย
        raise ExtractionError(ErrorCode.CORRUPT, f"ไฟล์ OLE2 เสียหาย: {e}") from e
    try:
        streams = {entry[0] for entry in ole.listdir(streams=True, storages=True)}
    finally:
        ole.close()
    if any(name in streams for name in _OLE_ENCRYPTED):
        raise ExtractionError(ErrorCode.ENCRYPTED, "ไฟล์ Office ตั้งรหัสผ่านไว้")
    for stream, detected in _OLE_STREAMS:
        if stream in streams:
            return detected
    raise ExtractionError(ErrorCode.UNSUPPORTED_TYPE, "เป็นไฟล์ OLE2 ที่ไม่ใช่ doc/ppt/xls")
