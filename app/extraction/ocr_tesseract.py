"""backend OCR: Tesseract 5 (tha + eng) รันบนเครื่องด้วย CPU — ไว้เทียบความแม่นกับ Typhoon (T06b)

เรียกโปรแกรม ``tesseract`` เป็น subprocess (ไม่ใช้ pytesseract; ขอ TSV ด้วย ``-c tessedit_create_tsv=1`` ไม่ใช้ config ชื่อ "tsv"
เพราะ ``--tessdata-dir`` ที่เราตั้งไม่มีโฟลเดอร์ configs) ด้วยภาพ PNG ชุดเดียวกับที่ส่งให้ Typhoon
(dpi เดียวกัน ตัดหน้าคู่ตรงร่องเหมือนกัน) ผลลัพธ์รูปแบบ TSV (``image_to_data``) -> บรรทัดพร้อม bbox
- ``OcrResult.text`` = ข้อความล้วน (บรรทัดละบรรทัด ย่อหน้าคั่นด้วยบรรทัดว่าง)
- ``OcrResult.extra`` = {"size": [กว้าง, สูง], "column_split": ..., "lines": [{"text", "bbox": [x0, y0, x1, y1], "conf", "block", "par"(, "col")}]}
  (พิกัดเป็นพิกเซลของภาพที่ส่ง; หน้าตัดครึ่งจะได้ extra ต่อภาพใน ``{"parts": [...]}``)
- Tesseract ภาษาไทยคั่นช่องว่างระหว่างคำ ตัดช่องว่างระหว่างตัวอักษรไทยกับไทยออก (ภาษาไทยไม่เว้นวรรคระหว่างคำ);
  ช่องว่างที่คั่นตัวอักษรไทยกับละติน/ตัวเลขคงไว้

แยกแถบข้างสีพื้น (side band): บางหน้ามีแถบสารบัญสีพื้นอยู่ซ้ายของเนื้อหา (KBANK) ซึ่ง Tesseract จัดหน้าอัตโนมัติ (psm 3) อ่านแถบกับ
แถวตารางเป็นบรรทัดเดียวกัน หลังรันเต็มภาพแล้ว ถ้า (ก) หาขอบแถบสีพื้นได้ชัด (``find_side_band_edge``: สีกลาง (median) ของแต่ละ
คอลัมน์พิกเซลในช่วง ``SIDE_BAND_WINDOW`` ของความกว้างเปลี่ยนจาก "มีสี" ไปเป็น "ขาว" อย่างฉับพลัน — ไม่ใช้โปรไฟล์หมึกแบบ ``find_gutter``
เพราะแถบเป็นพื้นสีไม่ใช่ช่องว่าง) และ (ข) มีบรรทัดคร่อมเส้นนั้นอย่างน้อย ``SPLIT_MIN_CROSSING`` บรรทัด จะรัน Tesseract แยกทีละ region
(แถบข้าง / เนื้อหา) แล้วบวก offset ให้ bbox กลับเป็นพิกัดภาพเดิม แต่ละบรรทัดมี ``col`` = 0 (แถบ) / 1 (เนื้อหา) และ
``extra["column_split"]`` = {"x", "method", "crossing_lines_before"} (ไม่แยก = None) ค่า/เวอร์ชันของวิธีนี้อยู่ใน ``cache_params``

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

SIDE_BAND_WINDOW = (0.05, 0.40)  # ช่วงความกว้างของภาพที่หาขอบแถบข้าง
SIDE_BAND_MIN_TINT = 25.0  # ระยะสี RGB จากสีขาวของฝั่งแถบ (ค่ากลางของคอลัมน์) ขั้นต่ำ — น้อยกว่านี้ไม่ถือว่ามีสีพื้น
SIDE_BAND_MAX_WHITE = 8.0  # ระยะสีจากสีขาวของฝั่งเนื้อหา สูงสุดที่ยังถือว่าขาว
SIDE_BAND_EDGE_SPAN = 40  # px: ความกว้างของช่วงที่เฉลี่ยทั้งสองข้างของขอบ
SPLIT_CROSS_MARGIN = 20  # px: บรรทัดต้องยื่นพ้นเส้นแบ่งเกินนี้ทั้งสองข้างถึงนับว่า "คร่อม"
SPLIT_MIN_CROSSING = 5  # จำนวนบรรทัดคร่อมขั้นต่ำถึงจะแยก region
COLUMN_SPLIT_METHOD = "side-band-v1"
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


def find_side_band_edge(image) -> Optional[int]:
    """x (พิกเซล) ที่แถบพื้นสีทางซ้ายสิ้นสุด (คอลัมน์แรกที่เป็นพื้นขาว) หรือ None ถ้าไม่มีขอบที่ชัด

    สีกลางของแต่ละคอลัมน์ (median ตามแนวตั้ง ทนต่อตัวหนังสือ/ภาพประกอบ) -> ระยะ RGB จากสีขาว; ขอบ = จุดที่ **ทุก** ``SIDE_BAND_EDGE_SPAN``
    คอลัมน์ทางซ้ายมีระยะ >= ``SIDE_BAND_MIN_TINT`` และ **ทุก** คอลัมน์ทางขวา <= ``SIDE_BAND_MAX_WHITE`` ในช่วง ``SIDE_BAND_WINDOW``
    (ใช้ min/max ไม่ใช่ค่าเฉลี่ย กันแถบที่แคบกว่าช่วงเฉลี่ยถูกจับเป็นขอบเพี้ยน); หลายจุดเลือกที่ต่างกันมากสุด
    """
    import numpy as np

    a = np.asarray(image.convert("RGB"), dtype=np.float32)
    dist = np.sqrt(((255.0 - np.median(a, axis=0)) ** 2).sum(axis=1))
    width, span = len(dist), SIDE_BAND_EDGE_SPAN
    best: Optional[tuple[float, int]] = None
    for x in range(max(int(width * SIDE_BAND_WINDOW[0]), span), min(int(width * SIDE_BAND_WINDOW[1]), width - span)):
        left, right = float(dist[x - span:x].min()), float(dist[x:x + span].max())
        if left >= SIDE_BAND_MIN_TINT and right <= SIDE_BAND_MAX_WHITE and (best is None or left - right > best[0]):
            best = (left - right, x)
    return best[1] if best else None


def crossing_lines(lines: list[dict], x: int) -> list[dict]:
    return [ln for ln in lines if ln["bbox"][0] < x - SPLIT_CROSS_MARGIN and ln["bbox"][2] > x + SPLIT_CROSS_MARGIN]


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
        return {"langs": self.langs, "psm": self.psm, "dpi": OCR_RENDER_DPI,
                "column_split": [COLUMN_SPLIT_METHOD, SIDE_BAND_WINDOW, SIDE_BAND_MIN_TINT, SIDE_BAND_MAX_WHITE,
                                 SIDE_BAND_EDGE_SPAN, SPLIT_CROSS_MARGIN, SPLIT_MIN_CROSSING]}

    def _tsv_lines(self, image_png: bytes) -> list[dict]:
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
        return parse_tsv(out)

    def recognize(self, image_png: bytes) -> OcrResult:
        import io

        from PIL import Image

        t0 = time.perf_counter()
        lines = self._tsv_lines(image_png)
        im = Image.open(io.BytesIO(image_png))
        column_split = None
        edge = find_side_band_edge(im)
        crossing = crossing_lines(lines, edge) if edge else []
        if edge and len(crossing) >= SPLIT_MIN_CROSSING:
            regions = []
            for col, box in enumerate(((0, 0, edge, im.height), (edge, 0, im.width, im.height))):
                buf = io.BytesIO()
                im.crop(box).save(buf, "PNG")
                for ln in self._tsv_lines(buf.getvalue()):
                    x0, y0, x1, y1 = ln["bbox"]
                    regions.append({**ln, "bbox": [x0 + box[0], y0, x1 + box[0], y1], "col": col, "block": ln["block"] + 1000 * col})
            lines = regions
            column_split = {"x": edge, "method": COLUMN_SPLIT_METHOD, "crossing_lines_before": len(crossing)}
        return OcrResult(text=lines_to_text(lines), backend=self.name, model=self.model, revision=self.revision,
                         seconds=time.perf_counter() - t0, extra={"size": list(im.size), "lines": lines, "column_split": column_split})
