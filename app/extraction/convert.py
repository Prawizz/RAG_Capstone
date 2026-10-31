"""แปลงไฟล์รุ่นเก่า (.doc / .xls / .ppt) เป็น .docx / .xlsx / .pptx ด้วย LibreOffice แล้วส่งเข้า extractor เดิม

- แปลงในโฟลเดอร์ชั่วคราวใหม่ทุกครั้ง (``TemporaryDirectory``) ลบทิ้งหลังใช้ทั้งกรณีสำเร็จและล้มเหลว
  สำเนาไฟล์ต้นทางอยู่ในโฟลเดอร์นั้นด้วย เพราะ LibreOffice เขียน .~lock / .tmp ไว้ข้างไฟล์ที่เปิด
- profile แยกทุกครั้ง (``-env:UserInstallation``) กันล็อกชนกันเมื่อแปลงพร้อมกัน; ``--headless --norestore``;
  ไม่เปิดให้ macro ทำงาน (profile ใหม่ใช้ค่าเริ่มต้นซึ่งปิด macro)
- timeout ``CONVERT_TIMEOUT_SECONDS`` -> conversion_timeout (retriable) และฆ่าทั้ง process tree
  (soffice.exe สร้าง soffice.bin เป็นลูก: Windows ``taskkill /F /T``, POSIX process group)
- แปลงไม่สำเร็จ / ไม่ได้ไฟล์ผลลัพธ์ -> corrupt; ไม่มี soffice -> unsupported_type
- ตรวจไฟล์ใส่รหัสก่อนแปลงเท่าที่ตรวจได้ (xls: record FILEPASS, doc: bit fEncrypted ใน FIB) -> encrypted
- ``_run`` คือจุดที่ฉีดคำสั่งแปลงแทนของจริงใน test (monkeypatch)
"""

import contextlib
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Optional

import olefile

from app.extraction.errors import ExtractionError
from app.models import DetectedType, ErrorCode

CONVERT_TIMEOUT_SECONDS = 60

# ชนิดเดิม -> (นามสกุลผลลัพธ์, ชื่อ filter ของ LibreOffice, ชนิดที่ extractor รับ)
TARGETS = {
    DetectedType.DOC: ("docx", "MS Word 2007 XML", DetectedType.DOCX),
    DetectedType.XLS: ("xlsx", "Calc MS Excel 2007 XML", DetectedType.XLSX),
    DetectedType.PPT: ("pptx", "Impress MS PowerPoint 2007 XML", DetectedType.PPTX),
}

NO_CONVERTER = "ระบบยังไม่ได้ติดตั้งตัวแปลงไฟล์รุ่นเก่า กรุณาบันทึกเป็น .docx / .xlsx / .pptx แล้วอัปโหลดใหม่"
CONVERT_FAILED = "แปลงไฟล์ไม่สำเร็จ ไฟล์อาจเสียหายหรือมีรหัสผ่าน"


def find_soffice() -> Optional[str]:
    env = os.environ.get("SOFFICE_PATH")
    if env:
        return env if Path(env).is_file() else None
    if sys.platform == "win32":
        roots = [os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")]
        candidates = [Path(r) / "LibreOffice" / "program" / "soffice.exe" for r in roots]
    elif sys.platform == "darwin":
        candidates = [Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")]
    else:
        candidates = [Path("/usr/bin/soffice"), Path("/usr/local/bin/soffice"), Path("/opt/libreoffice/program/soffice")]
    for c in candidates:
        if c.is_file():
            return str(c)
    return shutil.which("soffice") or shutil.which("libreoffice")


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=15)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=15)
    except Exception:
        pass


def _run(cmd: list[str], timeout: float) -> int:
    """รันคำสั่งแปลง คืน return code; เกินเวลาให้ฆ่าทั้ง process tree แล้วโยน ``subprocess.TimeoutExpired``"""
    kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "stdin": subprocess.DEVNULL}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        raise


def legacy_page_count(path: Path, detected: DetectedType) -> Optional[int]:
    """.doc: จำนวนหน้าที่ Word บันทึกไว้ใน SummaryInformation ของไฟล์ต้นฉบับ (ไม่มี / 0 -> None)"""
    if detected != DetectedType.DOC:
        return None
    try:
        ole = olefile.OleFileIO(str(path))
    except Exception:
        return None
    try:
        pages = ole.get_metadata().num_pages
        return int(pages) if pages else None
    except Exception:
        return None
    finally:
        ole.close()


def check_not_encrypted(path: Path, detected: DetectedType) -> None:
    """ตรวจได้เท่าไหร่ตรวจ ตรวจไม่ได้ก็ปล่อยให้ timeout / corrupt จัดการ"""
    try:
        ole = olefile.OleFileIO(str(path))
    except Exception:
        return
    try:
        if detected == DetectedType.XLS:
            name = "Workbook" if ole.exists("Workbook") else "Book" if ole.exists("Book") else None
            if name is None:
                return
            data = ole.openstream(name).read(1 << 20)
            pos = 0
            while pos + 4 <= len(data):
                rid, size = struct.unpack_from("<HH", data, pos)
                if rid == 0x002F:  # FILEPASS
                    raise ExtractionError(ErrorCode.ENCRYPTED, "ไฟล์ถูกตั้งรหัสผ่าน")
                if rid == 0x000A:  # EOF ของ globals substream
                    break
                pos += 4 + size
        elif detected == DetectedType.DOC and ole.exists("WordDocument"):
            head = ole.openstream("WordDocument").read(16)
            if len(head) >= 12 and struct.unpack_from("<H", head, 0x0A)[0] & 0x0100:  # fEncrypted
                raise ExtractionError(ErrorCode.ENCRYPTED, "ไฟล์ถูกตั้งรหัสผ่าน")
    except ExtractionError:
        raise
    except Exception:
        return
    finally:
        ole.close()


@contextlib.contextmanager
def converted(path: str | Path, detected: DetectedType) -> Iterator[Path]:
    """แปลงไฟล์รุ่นเก่า ให้ path ของไฟล์ใหม่ที่อยู่ในโฟลเดอร์ชั่วคราว (ลบทิ้งเมื่อออกจาก with)"""
    path = Path(path)
    ext, flt, _ = TARGETS[detected]
    soffice = find_soffice()
    if soffice is None:
        raise ExtractionError(ErrorCode.UNSUPPORTED_TYPE, NO_CONVERTER)
    check_not_encrypted(path, detected)
    with tempfile.TemporaryDirectory(prefix="rag-convert-", ignore_cleanup_errors=True) as tmp:
        tmp_dir = Path(tmp)
        src_dir, out_dir, profile = tmp_dir / "in", tmp_dir / "out", tmp_dir / "profile"
        src_dir.mkdir()
        out_dir.mkdir()
        src = src_dir / f"input.{detected.value}"
        shutil.copyfile(path, src)
        cmd = [
            soffice, "--headless", "--norestore", "--nolockcheck",
            f"-env:UserInstallation={profile.as_uri()}",
            "--convert-to", f"{ext}:{flt}", "--outdir", str(out_dir), str(src),
        ]
        try:
            code = _run(cmd, CONVERT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as e:
            raise ExtractionError(
                ErrorCode.CONVERSION_TIMEOUT, f"แปลงไฟล์เกินเวลา {CONVERT_TIMEOUT_SECONDS} วินาที"
            ) from e
        except OSError as e:
            raise ExtractionError(ErrorCode.UNSUPPORTED_TYPE, NO_CONVERTER) from e
        out = out_dir / f"input.{ext}"
        if code != 0 or not out.is_file() or out.stat().st_size == 0:
            raise ExtractionError(ErrorCode.CORRUPT, CONVERT_FAILED)
        yield out
