"""เตรียมรายการให้คนตัดสินข้อความที่ถูกของ 7 หน้า (T06b ช่วงที่ 1)

    python scripts/ocr_truth_prepare.py            # อ่านผล OCR จากแคช (ต้องรัน scripts/ocr_pages.py ทั้งสอง backend ก่อน)
    python scripts/ocr_truth_prepare.py --no-html  # สร้างแค่ items.json

ผลลัพธ์
- eval/datasets/ocr_truth/<file>_p<page>.items.json   (ข้อความจาก PDF สาธารณะ เข้า git ได้)
- eval/datasets/files/ocr_truth_review.html           (ไฟล์เดียว มีภาพ ไม่เข้า git) เปิดด้วยเบราว์เซอร์

ขั้นตอนต่อหน้า
1. normalize (normalizer กลาง) ทั้งสองฝั่ง แตกเป็นบรรทัด (ตาราง = แถวละบรรทัด) ``<figure>`` ของ Typhoon เก็บแยก (``figures``)
   ไม่เอามาเป็นข้อความที่ถูก; เลขหน้าที่ Typhoon ตัดออก (``<page_number>``) ไม่ถือเป็น "มีแต่ใน Tesseract"
2. canonical = ตัดเครื่องหมาย markdown (# * | และ "- " หัวบรรทัด), ช่องว่างทุกชนิด, ปรับเครื่องหมายคำพูดโค้งเป็นตรง
3. หน่วยที่ให้คนตัดสิน = บรรทัดที่มองเห็นในภาพ (บรรทัดของ Tesseract) เพราะ Typhoon เขียนย่อหน้ายาวเป็นบรรทัดเดียวและลำดับอ่านข้ามคอลัมน์ได้
   แต่ละบรรทัด Tesseract ถูกนำไปหาตำแหน่งที่คล้ายที่สุดในข้อความ Typhoon ทั้งหน้า (canonical ต่อกัน, rapidfuzz partial_ratio_alignment
   แล้วขยับขอบ ± ``REFINE_SHIFT`` ตัวให้ความคล้ายแบบ Levenshtein สูงสุด) — **ไม่ตามลำดับ ไม่ต้องติดกัน** จึงรับมือหน้า 2 คอลัมน์ได้
   ข้อความ Typhoon ช่วงนั้น (ตัดมาจากบรรทัดของ Typhoon ตามที่ Typhoon เขียน) กลายเป็นอีกฝั่งของคู่
4. แบ่งเป็น 4 แบบ
   - agree: canonical ตรงกันทุกตัวอักษร -> รับอัตโนมัติ แต่สุ่ม ``AUDIT_FRACTION`` (seed ตายตัว) ให้คนตรวจซ้ำ (``audit``)
   - diff: คล้าย (ความคล้าย >= ``MATCH_SCORE``) แต่ไม่ตรง
   - tesseract_only: บรรทัด Tesseract ที่หาช่วงคล้ายใน Typhoon ไม่ได้ (สงสัย Typhoon อ่านตก)
   - typhoon_only: ข้อความ Typhoon ตั้งแต่ ``MIN_GAP_CANON`` ตัวที่ไม่มีบรรทัด Tesseract ใดคลุม (สงสัยโมเดลแต่ง; ภาพที่ให้ดู = แถบ
     ระหว่างบรรทัดข้างเคียงที่จับคู่ได้ ถ้าไม่ได้/สูงเกิน ``MAX_REGION_HEIGHT`` = ภาพทั้งหน้า)
   บรรทัด Tesseract ที่เป็นขยะ (canonical สั้นกว่า ``MIN_TESS_CANON`` หรือ conf ต่ำกว่า ``MIN_TESS_CONF``) ถูกข้ามและบันทึกไว้
   ใน ``skipped_tesseract`` ให้ตรวจย้อนหลังได้
   ``pos`` ของแต่ละรายการ = ตำแหน่งในข้อความ Typhoon (ไว้เรียงลำดับ / ประกอบข้อความที่ถูกในช่วงที่ 2)
"""

import argparse
import base64
import hashlib
import html as html_lib
import io
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.extraction.ocr import OcrResult  # noqa: E402
from app.extraction.ocr_markdown import parse_ocr_markdown  # noqa: E402
from app.models import BlockType  # noqa: E402
from app.thai.normalizer import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "eval" / "datasets"
TRUTH_DIR = DATASETS / "ocr_truth"
REVIEW_HTML = DATASETS / "files" / "ocr_truth_review.html"
TEMPLATE = Path(__file__).resolve().parent / "ocr_review_template.html"

MATCH_SCORE = 70.0  # ความคล้าย (0–100; partial_ratio ตอนหาตำแหน่ง, Levenshtein หลังขยับขอบ) ขั้นต่ำที่ถือว่า "คล้าย"
MIN_FUZZY_LEN = 4  # บรรทัด Typhoon ที่ canonical สั้นกว่านี้ ไม่ใช้ fuzzy (ต้องตรงทั้งบรรทัด Tesseract เท่านั้น)
REFINE_SHIFT = 4  # ขยับขอบช่วงข้อความ Typhoon ได้กี่ตัวอักษรเพื่อให้ตรงกับบรรทัด Tesseract
MIN_GAP_CANON = 6  # ข้อความ Typhoon ที่ไม่มีบรรทัด Tesseract คลุม ยาวเท่านี้ขึ้นไปถึงเป็นรายการ typhoon_only
MAX_REGION_HEIGHT = 700  # px สูงสุดของแถบภาพระหว่างบรรทัดข้างเคียง (เกินนี้ = ลำดับอ่านข้ามคอลัมน์ ใช้ภาพทั้งหน้า)
MIN_TESS_CANON = 2
MIN_TESS_CONF = 25.0
AUDIT_FRACTION = 0.10
AUDIT_SEED = 20260101
CROP_PAD = 14  # px รอบ bbox
PAGE_THUMB_LONG_SIDE = 1500

_MD_CHARS = re.compile(r"[#*|]")
_LEADING_DASH = re.compile(r"(?m)^\s*-\s+")
_WS = re.compile(r"\s+")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def canon(text: str) -> str:
    """รูปที่ใช้เทียบ: normalize -> ตัดเครื่องหมาย markdown -> ตัดช่องว่าง"""
    t = normalize(text).translate(_QUOTES)
    t = _LEADING_DASH.sub("", t)
    return _WS.sub("", _MD_CHARS.sub("", t))


# ---------- แตกบรรทัด ----------

def typhoon_lines(raw: str) -> tuple[list[str], list[str], list[int]]:
    """(บรรทัดข้อความ, ข้อความ figure, เลขหน้า) จากผลดิบของ Typhoon; ข้อความผ่าน normalizer แล้วแต่ยังไม่ตัด markdown ออก"""
    parsed = parse_ocr_markdown(raw)
    lines: list[str] = []
    figures: list[str] = []
    for b in parsed.blocks:
        if b.type == BlockType.IMAGE:
            figures.append(normalize(b.text))
        elif b.type == BlockType.TABLE:
            for row in b.table.rows:
                t = normalize(" | ".join(c for c in row if c))
                if t.strip():
                    lines.append(t)
        else:
            for ln in b.text.splitlines():
                t = normalize(ln)
                t = re.sub(r"^\s*[*\-]\s+", "", t)  # จุดนำหน้ารายการ (เลขลำดับ "1." คงไว้ เพราะมีในภาพ)
                if t.strip():
                    lines.append(t.strip())
    return [ln for ln in lines if canon(ln)], figures, parsed.page_numbers


def tesseract_parts(res: OcrResult) -> list[dict]:
    extra = res.extra or {}
    return extra["parts"] if "parts" in extra else [extra]


# ---------- จับคู่ ----------

@dataclass
class TLine:
    text: str
    canon: str
    image: int
    bbox: list[int]
    conf: Optional[float]
    start: int = 0
    end: int = 0
    covered: int = 0


def _ratio(a: str, b: str) -> float:
    """ความคล้าย 0–100 แบบ Levenshtein (แทนที่ตัวอักษรได้ ไม่ใช่แค่ใส่/ลบ) — ใช้ตัดสินขอบช่วงข้อความ Typhoon:
    ratio แบบ indel จะชอบ "ตัดตัวอักษรที่ Tesseract อ่านผิดทิ้ง" มากกว่า "ยอมให้ผิดหนึ่งตัว" ทำให้ขอบช่วงเลื่อนเข้าไปกลางคำ"""
    from rapidfuzz.distance import Levenshtein

    return 100.0 * Levenshtein.normalized_similarity(a, b)


def _align(needle: str, haystack: str):
    from rapidfuzz import fuzz

    return fuzz.partial_ratio_alignment(needle, haystack)


def _span_display(segs: list[dict], s: int, e: int) -> str:
    """ข้อความแสดงผล (มีช่องว่างตามที่ Typhoon เขียน) ของช่วง canonical [s, e) ของข้อความ Typhoon ทั้งหน้า"""
    parts = []
    for sg in segs:
        lo, hi = max(s, sg["start"]), min(e, sg["end"])
        if lo >= hi:
            continue
        if sg["idx"] is None:
            parts.append(sg["display"])
        else:
            a, b = lo - sg["start"], hi - sg["start"]
            parts.append(sg["display"][sg["idx"][a]: sg["idx"][b - 1] + 1])
    return " ".join(parts).strip()


def _refine(c: str, hay: str, s: int, e: int) -> tuple[float, int, int]:
    """ขยับขอบช่วง [s, e) ทีละไม่เกิน ``REFINE_SHIFT`` ตัวอักษร ให้ fuzz.ratio กับ c สูงสุด (เท่ากัน = ใกล้ขอบเดิมกว่า)"""
    best = (_ratio(c, hay[s:e]), s, e, 0)
    for ds in range(-REFINE_SHIFT, REFINE_SHIFT + 1):
        for de in range(-REFINE_SHIFT, REFINE_SHIFT + 1):
            s2, e2 = s + ds, e + de
            if s2 < 0 or e2 > len(hay) or e2 <= s2:
                continue
            r = _ratio(c, hay[s2:e2])
            cost = abs(ds) + abs(de)
            if r > best[0] + 1e-9 or (abs(r - best[0]) <= 1e-9 and cost < best[3]):
                best = (r, s2, e2, cost)
    return best[0], best[1], best[2]


def build_items(file: str, page: int, typhoon: OcrResult, tesseract: OcrResult) -> dict:
    """หน่วยที่ให้คนตัดสิน = บรรทัดที่มองเห็นในภาพ (บรรทัดของ Tesseract) คู่กับช่วงข้อความ Typhoon ที่ตรงกัน

    เหตุผล: Typhoon เขียนย่อหน้ายาวเป็นบรรทัดเดียว (หลายร้อยตัวอักษร) และลำดับอ่านข้ามคอลัมน์ได้ ส่วน Tesseract ให้บรรทัดสั้นพร้อม bbox
    จึงเอาแต่ละบรรทัด Tesseract ไปหาตำแหน่งในข้อความ Typhoon ทั้งหน้า (ไม่ตามลำดับ ไม่ต้องติดกัน) แล้วตัดข้อความ Typhoon ช่วงนั้นมาเทียบ
    """
    ty_lines, figures, page_numbers = typhoon_lines(typhoon.text)

    segs: list[dict] = []
    hay = ""
    for line in ty_lines:
        disp = _display(line)
        c = canon(disp)
        idx = [i for i, ch in enumerate(disp) if not ch.isspace()]
        segs.append({"display": disp, "canon": c, "idx": idx if len(idx) == len(c) else None, "start": len(hay), "end": len(hay) + len(c)})
        hay += c

    kept: list[TLine] = []
    skipped: list[dict] = []
    page_no_canon = {str(n) for n in page_numbers}
    for img, part in enumerate(tesseract_parts(tesseract)):
        for ln in part.get("lines", []):
            c = canon(ln["text"])
            reason = None
            if len(c) < MIN_TESS_CANON:
                reason = "สั้นเกินไป"
            elif ln.get("conf") is not None and ln["conf"] < MIN_TESS_CONF:
                reason = f"conf ต่ำ ({ln['conf']})"
            elif c.isdigit() and c.lstrip("0") in page_no_canon:
                reason = "เลขหน้า (Typhoon ตัดออกโดยตั้งใจ)"
            if reason:
                skipped.append({"text": ln["text"], "reason": reason, "image": img, "bbox": ln["bbox"]})
                continue
            kept.append(TLine(ln["text"], c, img, ln["bbox"], ln.get("conf")))

    items: list[dict] = []
    covered = [False] * len(hay)
    for k, t in enumerate(kept):
        c = t.canon
        span = None
        if hay and len(c) < MIN_FUZZY_LEN:
            p = hay.find(c)  # สั้นมาก: ต้องเจอตรงตัว
            if p >= 0:
                span = (100.0, p, p + len(c))
        elif hay and len(c) <= len(hay):
            al = _align(c, hay)
            if al.score >= MATCH_SCORE:
                r, s, e = _refine(c, hay, al.dest_start, al.dest_end)
                if r >= MATCH_SCORE:
                    span = (r, s, e)
        base = {"typhoon": None, "tesseract": {"text": t.text, "image": t.image, "bbox": t.bbox, "conf": t.conf},
                "region": None, "auto_accept": False, "audit": False}
        if span is None:
            items.append({**base, "kind": "tesseract_only", "score": 0.0, "pos": None, "k": k})
            continue
        r, s, e = span
        for i in range(s, e):
            covered[i] = True
        kind = "agree" if hay[s:e] == c else "diff"
        items.append({**base, "kind": kind, "score": round(r, 1), "pos": s, "end": e, "k": k,
                      "typhoon": {"text": _span_display(segs, s, e)}, "auto_accept": kind == "agree"})

    # ช่วงข้อความ Typhoon ที่ไม่มีบรรทัด Tesseract ใดคลุม (ต่อบรรทัดของ Typhoon) = typhoon_only
    matched = sorted((it for it in items if it["pos"] is not None), key=lambda it: it["pos"])
    for sg in segs:
        i = sg["start"]
        while i < sg["end"]:
            if covered[i]:
                i += 1
                continue
            j = i
            while j < sg["end"] and not covered[j]:
                j += 1
            if j - i >= MIN_GAP_CANON:
                prev = next((m for m in reversed(matched) if m["end"] <= i), None)
                nxt = next((m for m in matched if m["pos"] >= j), None)
                region = None
                if prev and nxt and prev["tesseract"]["image"] == nxt["tesseract"]["image"]:
                    lo, hi = sorted((prev["tesseract"]["bbox"][1], nxt["tesseract"]["bbox"][3]))
                    if hi - lo <= MAX_REGION_HEIGHT:
                        region = {"image": prev["tesseract"]["image"], "bbox": [0, max(lo - 2 * CROP_PAD, 0), 10 ** 6, hi + 2 * CROP_PAD]}
                items.append({"typhoon": {"text": _span_display(segs, i, j)}, "tesseract": None, "region": region,
                              "auto_accept": False, "audit": False, "kind": "typhoon_only", "score": 0.0, "pos": i, "end": j})
            i = j

    # ตำแหน่งของ tesseract_only ในลำดับของ Typhoon: ต่อท้ายบรรทัด Tesseract ก่อนหน้าที่จับคู่ได้
    last_end = 0
    by_k = sorted(items, key=lambda it: it.get("k", 10 ** 9))
    for it in by_k:
        if it["kind"] == "tesseract_only":
            it["pos"] = last_end - 0.5
        elif it["kind"] in ("agree", "diff"):
            last_end = it["end"]
    for it in items:
        it.pop("k", None)
    items.sort(key=lambda it: (it["pos"], 0 if it["kind"] != "tesseract_only" else 1))

    agree = [it for it in items if it["kind"] == "agree"]
    if agree:
        rng = random.Random(f"{AUDIT_SEED}:{file}:{page}")
        for it in rng.sample(agree, max(1, round(AUDIT_FRACTION * len(agree)))):
            it["audit"] = True

    stem = Path(file).stem
    for n, it in enumerate(items, 1):
        it["id"] = f"{stem}_p{page}_{n:03d}"
        it["suggest"] = _suggest(it)
        it["order"] = n - 1
        it.pop("end", None)
    counts = {"agree": 0, "agree_audit": 0, "diff": 0, "typhoon_only": 0, "tesseract_only": 0}
    for it in items:
        if it["kind"] == "agree":
            counts["agree_audit" if it["audit"] else "agree"] += 1
        else:
            counts[it["kind"]] += 1
    counts["to_judge"] = counts["agree_audit"] + counts["diff"] + counts["typhoon_only"] + counts["tesseract_only"]
    return {
        "file": file, "page": page,
        "typhoon": {"backend": typhoon.backend, "model": typhoon.model, "revision": typhoon.revision,
                    "seconds": typhoon.seconds, "tokens": typhoon.tokens},
        "tesseract": {"backend": tesseract.backend, "model": tesseract.model, "revision": tesseract.revision,
                      "seconds": tesseract.seconds},
        "typhoon_page_numbers": page_numbers,
        "figures": figures, "skipped_tesseract": skipped, "counts": counts, "items": items,
    }


def _display(line: str) -> str:
    """ข้อความที่แสดงให้คน/เก็บเป็น "ข้อความที่ Typhoon อ่านได้": ตัดเครื่องหมาย markdown (# * |) แต่คงช่องว่าง"""
    return _WS.sub(" ", _MD_CHARS.sub(" ", line)).strip()


def _suggest(it: dict) -> str:
    """ตัวที่ "ใกล้กว่า" สำหรับใส่ให้คนแก้ต่อ: เทียบสัดส่วนคำไทยที่อยู่ในพจนานุกรม (วัดอย่างเดียว ไม่ใช่ข้อความที่ถูก)
    ตัวใดไม่มีให้ใช้อีกตัว; เสมอ/ไม่มีอักษรไทย -> Typhoon"""
    ty, te = it.get("typhoon"), it.get("tesseract")
    if not ty:
        return "tesseract"
    if not te:
        return "typhoon"
    from app.extraction.quality import dictionary_ratio

    a, b = dictionary_ratio(ty["text"]), dictionary_ratio(te["text"])
    return "tesseract" if (a is not None and b is not None and b > a) else "typhoon"


# ---------- ภาพ + HTML ----------

def _jpeg_b64(im, quality: int = 80) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def attach_images(page_data: dict, images_png: list[bytes]) -> dict:
    """ภาพครอปต่อรายการที่ต้องตัดสิน + ภาพหน้า (ย่อ) -> ใส่ใน page_data (เฉพาะใช้ทำ HTML ไม่เขียนลง items.json)"""
    from PIL import Image

    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in images_png]
    out_items = []
    for it in page_data["items"]:
        if it["auto_accept"] and not it["audit"]:
            continue
        box = (it.get("tesseract") or {}).get("bbox") if it["kind"] != "typhoon_only" else None
        img_i = (it.get("tesseract") or {}).get("image", 0) if box else None
        if it["kind"] == "typhoon_only" and it["region"]:
            img_i, box = it["region"]["image"], it["region"]["bbox"]
        crop = None
        if box is not None and img_i is not None and img_i < len(imgs):
            im = imgs[img_i]
            x0, y0, x1, y1 = box
            c = im.crop((max(x0 - CROP_PAD, 0), max(y0 - CROP_PAD, 0), min(x1 + CROP_PAD, im.width), min(y1 + CROP_PAD, im.height)))
            if c.width > 0 and c.height > 0:
                crop = _jpeg_b64(c)
        out_items.append({**it, "crop": crop})
    thumbs = []
    for im in imgs:
        s = PAGE_THUMB_LONG_SIDE / max(im.size)
        thumbs.append(_jpeg_b64(im.resize((int(im.width * s), int(im.height * s))), 70) if s < 1 else _jpeg_b64(im, 70))
    return {"file": page_data["file"], "page": page_data["page"], "counts": page_data["counts"],
            "items": out_items, "page_images": thumbs}


def build_html(pages: list[dict]) -> str:
    total = sum(len(p["items"]) for p in pages)
    dataset_id = hashlib.sha256(json.dumps([[it["id"] for it in p["items"]] for p in pages]).encode()).hexdigest()[:16]
    payload = json.dumps({"dataset_id": dataset_id, "total": total, "pages": pages}, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")  # กัน </script> ในข้อความ
    return TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload)


# ---------- main ----------

def main(argv: list[str] | None = None) -> int:
    import importlib.util

    from app.extraction.ocr import get_backend, recognize_page_cached
    from app.extraction.pipeline import _hash_file
    from app.models import DetectedType

    spec = importlib.util.spec_from_file_location("ocr_pages", Path(__file__).with_name("ocr_pages.py"))
    ocr_pages = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ocr_pages)

    ap = argparse.ArgumentParser(description="สร้างรายการให้คนตัดสินข้อความที่ถูก")
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ty, te = get_backend("typhoon-local"), get_backend("tesseract-local")
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    html_pages = []
    total_to_judge = 0
    for name, page_no in ocr_pages.PAGES:
        path = DATASETS / name
        h = _hash_file(path)[0]
        # ต้องมีในแคชแล้ว (ไม่เรียกโมเดลที่นี่): ใช้ cache อย่างเดียว
        r_ty, pg, hit1 = recognize_page_cached(ty, path, DetectedType.PDF, h, page_no)
        r_te, _, hit2 = recognize_page_cached(te, path, DetectedType.PDF, h, page_no)
        if not (hit1 and hit2):
            print(f"[คำเตือน] {name} p{page_no}: ไม่พบในแคชครบ ({'typhoon' if not hit1 else ''} {'tesseract' if not hit2 else ''}) "
                  "ผลถูกรันใหม่ตอนนี้ — ควรรัน scripts/ocr_pages.py ก่อน", file=sys.stderr)
        data = build_items(name, page_no, r_ty, r_te)
        data["split"] = {"method": pg.split_method, "x": pg.split_x, "dpi": pg.dpi}
        out = TRUTH_DIR / f"{Path(name).stem}_p{page_no}.items.json"
        out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        c = data["counts"]
        total_to_judge += c["to_judge"]
        print(f"{name} p{page_no}: agree(รับอัตโนมัติ)={c['agree']} agree(สุ่มตรวจ)={c['agree_audit']} diff={c['diff']} "
              f"typhoon_only={c['typhoon_only']} tesseract_only={c['tesseract_only']} -> ต้องตัดสิน {c['to_judge']}  "
              f"(ข้ามบรรทัดขยะของ Tesseract {len(data['skipped_tesseract'])}, figure {len(data['figures'])})  -> {out.name}")
        if not args.no_html:
            html_pages.append(attach_images(data, pg.images))
    print(f"รวมรายการที่คนต้องตัดสิน: {total_to_judge}")
    if not args.no_html:
        REVIEW_HTML.parent.mkdir(parents=True, exist_ok=True)
        REVIEW_HTML.write_text(build_html(html_pages), encoding="utf-8")
        print(f"หน้าตรวจ: {REVIEW_HTML}  ({REVIEW_HTML.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
