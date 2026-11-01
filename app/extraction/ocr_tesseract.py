"""backend OCR: Tesseract 5 (tha + eng) รันบนเครื่องด้วย CPU — ไว้เทียบความแม่นกับ Typhoon (T06b)

เรียกโปรแกรม ``tesseract`` เป็น subprocess (ไม่ใช้ pytesseract; ขอ TSV ด้วย ``-c tessedit_create_tsv=1`` ไม่ใช้ config ชื่อ "tsv"
เพราะ ``--tessdata-dir`` ที่เราตั้งไม่มีโฟลเดอร์ configs) ด้วยภาพ PNG ชุดเดียวกับที่ส่งให้ Typhoon
(dpi เดียวกัน ตัดหน้าคู่ตรงร่องเหมือนกัน) ผลลัพธ์รูปแบบ TSV (``image_to_data``) -> บรรทัดพร้อม bbox
- ``OcrResult.text`` = ข้อความล้วน (บรรทัดละบรรทัด ย่อหน้าคั่นด้วยบรรทัดว่าง)
- ``OcrResult.extra`` = {"size": [กว้าง, สูง], "lines": [{"text", "bbox": [x0, y0, x1, y1], "conf", "block", "par"}]}
  (พิกัดเป็นพิกเซลของภาพที่ส่ง; หน้าตัดครึ่งจะได้ extra ต่อภาพใน ``{"parts": [...]}``)
- Tesseract ภาษาไทยคั่นช่องว่างระหว่างคำ ตัดช่องว่างระหว่างตัวอักษรไทยกับไทยออก (ภาษาไทยไม่เว้นวรรคระหว่างคำ);
  ช่องว่างที่คั่นตัวอักษรไทยกับละติน/ตัวเลขคงไว้

การตั้งค่า: env ``TESSERACT_CMD`` (path ของ tesseract), ``OCR_TESSDATA_DIR`` (โฟลเดอร์ traineddata ชนิด best ถ้ามี;
ไม่ตั้ง = ใช้ tessdata ข้างโปรแกรม) ไม่มีโปรแกรม / ไม่มี tha.traineddata หรือ eng.traineddata -> ``OcrUnavailable``
revision = รุ่นโปรแกรม + SHA-256 ย่อของไฟล์ภาษา (key ของแคชจึงเปลี่ยนเมื่อเปลี่ยน traineddata)
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from app.extraction.ocr import OCR_RENDER_DPI, OcrResult, OcrUnavailable

LANGS = "tha+eng"
PSM = 3  # จัดหน้าอัตโนมัติ (เต็มหน้า ไม่มี OSD)
RUN_TIMEOUT_SECONDS = 300
_THAI = re.compile("[฀-๿]")

DEFAULT_LOCATIONS = (
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
    Path.home() / "tools" / "Tesseract-OCR" / "tesseract.exe",
    Path("/usr/bin/tesseract"), Path("/usr/local/bin/tesseract"), Path("/opt/homebrew/bin/tesseract"),
)


def find_tesseract() -> Optional[str]:
    env = os.environ.get("TESSERACT_CMD")
    if env:
        return env if Path(env).is_file() else None
    return shutil.which("tesseract") or next((str(p) for p in DEFAULT_LOCATIONS if p.is_file()), None)


def _tessdata_dir(cmd: str) -> Path:
    env = os.environ.get("OCR_TESSDATA_DIR")
    return Path(env) if env else Path(cmd).resolve().parent / "tessdata"


def _run(cmd: list[str], timeout: float) -> tuple[int, str, str]:
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    p = subprocess.run(cmd, capture_output=True, timeout=timeout, **kwargs)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def join_words(words: list[str]) -> str:
    """ต่อคำในบรรทัด: ไทย+ไทย ติดกัน นอกนั้นคั่นช่องว่าง"""
    out = ""
    for w in words:
        if out and not (_THAI.match(out[-1]) and _THAI.match(w[0])):
            out += " "
        out += w
    return out


def parse_tsv(tsv: str) -> list[dict]:
    """TSV ของ Tesseract -> บรรทัด [{text, bbox, conf, block, par}] ตามลำดับที่โปรแกรมให้"""
    lines: dict[tuple[int, int, int], dict] = {}
    order: list[tuple[int, int, int]] = []
    for row in tsv.splitlines()[1:]:
        c = row.split("\t", 11)
        if len(c) < 12 or c[0] != "5":  # level 5 = คำ
            continue
        text = c[11].strip()
        if not text:
            continue
        try:
            block, par, line = int(c[2]), int(c[3]), int(c[4])
            x, y, w, h, conf = int(c[6]), int(c[7]), int(c[8]), int(c[9]), float(c[10])
        except ValueError:
            continue
        key = (block, par, line)
        if key not in lines:
            lines[key] = {"words": [], "x0": x, "y0": y, "x1": x + w, "y1": y + h, "confs": [], "block": block, "par": par}
            order.append(key)
        d = lines[key]
        d["words"].append(text)
        d["x0"], d["y0"] = min(d["x0"], x), min(d["y0"], y)
        d["x1"], d["y1"] = max(d["x1"], x + w), max(d["y1"], y + h)
        if conf >= 0:
            d["confs"].append(conf)
    out = []
    for key in order:
        d = lines[key]
        out.append({
            "text": join_words(d["words"]),
            "bbox": [d["x0"], d["y0"], d["x1"], d["y1"]],
            "conf": round(sum(d["confs"]) / len(d["confs"]), 1) if d["confs"] else None,
            "block": d["block"], "par": d["par"],
        })
    return out


def lines_to_text(lines: list[dict]) -> str:
    parts: list[str] = []
    prev = None
    for ln in lines:
        key = (ln["block"], ln["par"])
        if prev is not None and key != prev:
            parts.append("")
        parts.append(ln["text"])
        prev = key
    return "\n".join(parts)


class TesseractLocalBackend:
    name = "tesseract-local"
    sends_data_offsite = False

    def __init__(self, psm: int = PSM, langs: str = LANGS):
        cmd = find_tesseract()
        if cmd is None:
            raise OcrUnavailable("ไม่พบโปรแกรม tesseract (ติดตั้ง Tesseract 5 หรือตั้ง TESSERACT_CMD)")
        self.cmd, self.psm, self.langs = cmd, psm, langs
        self.tessdata = _tessdata_dir(cmd)
        files = [self.tessdata / f"{lang}.traineddata" for lang in langs.split("+")]
        missing = [f.name for f in files if not f.is_file()]
        if missing:
            raise OcrUnavailable(f"ไม่พบไฟล์ภาษาของ Tesseract {missing} ใน {self.tessdata} (ตั้ง OCR_TESSDATA_DIR)")
        code, out, err = _run([cmd, "--version"], 30)
        m = re.search(r"tesseract\s+v?([\w.\-]+)", out + err, re.IGNORECASE)
        self.model = f"tesseract-{m.group(1) if m else 'unknown'}"
        h = hashlib.sha256()
        for f in files:
            h.update(f.read_bytes())
        self.revision = h.hexdigest()[:12]

    def cache_params(self) -> dict:
        return {"langs": self.langs, "psm": self.psm, "dpi": OCR_RENDER_DPI}

    def recognize(self, image_png: bytes) -> OcrResult:
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="rag-tess-") as tmp:
            img = Path(tmp) / "page.png"
            img.write_bytes(image_png)
            code, out, err = _run(
                [self.cmd, str(img), "stdout", "-l", self.langs, "--psm", str(self.psm), "--dpi", str(OCR_RENDER_DPI),
                 "--tessdata-dir", str(self.tessdata), "-c", "tessedit_create_tsv=1", "-c", "tessedit_create_txt=0"],
                RUN_TIMEOUT_SECONDS,
            )
        if code != 0:
            raise RuntimeError(f"tesseract ล้มเหลว (exit {code}): {err.strip()[:300]}")
        if not out.lstrip().startswith("level"):  # ไม่ใช่ TSV (เช่น config ไม่ถูกอ่าน) ห้ามเงียบแล้วได้ผลว่าง
            raise RuntimeError("tesseract ไม่ได้คืนผลแบบ TSV")
        lines = parse_tsv(out)
        from PIL import Image
        import io

        size = list(Image.open(io.BytesIO(image_png)).size)
        return OcrResult(text=lines_to_text(lines), backend=self.name, model=self.model, revision=self.revision,
                         seconds=time.perf_counter() - t0, extra={"size": size, "lines": lines})
