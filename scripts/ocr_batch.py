"""OCR แบบ batch สำหรับรันข้ามคืน: OCR หน้า needs_ocr ของ One Report ทั้ง 3 เล่มด้วย typhoon-local ลงแคช

    python scripts/ocr_batch.py                                    # KBANK -> CPALL -> PTT ทั้งหมด
    python scripts/ocr_batch.py --files ONEREPORTKBANKT.PDF --limit 2     # ลอง 2 หน้า
    python scripts/ocr_batch.py --files ONEREPORTKBANKT.PDF --pages 1-20,50

หลักการ
- หา "หน้าที่ต้อง OCR" ด้วย extract_file + ด่านตรวจตัวเดิม (ทั้งไฟล์ — กฎระดับไฟล์ต้องเห็นทุกหน้า) แล้วเรียก
  ``recognize_page_auto`` ตัวเดิม (ตัวเดียวกับ run_ocr) โดยไม่ส่งพารามิเตอร์ OCR ใดๆ -> cache key ตรงกับที่ pipeline ใช้
  หน้าที่รอบแรกชนเพดานโทเคน -> OCR ซ้ำแบบแบ่งบน/ล่างอัตโนมัติ (แคชรายชิ้น) log บรรทัด "# ... piece" ต่อชิ้น
  และคอลัมน์ error ของหน้าได้ "resplit: ..." (โทเคนรอบแรก, จำนวนชิ้น, ชิ้นที่ยังชนเพดาน, ไม่มีช่องว่าง, ครึ่งหน้าที่หมุนภาพให้ตั้งตรง)
- หน้าที่อยู่ในแคชแล้ว = ข้าม (ไม่โหลดโมเดล); หยุดกลางทางแล้วรันซ้ำต่อได้เพราะอ่านจากแคช
- รันทีละหน้า ไม่ขนาน · หน้า error -> log แล้วไปหน้าถัดไป · CUDA OOM -> ล้าง cache ของ torch แล้วลองหน้านั้นใหม่ 1 ครั้ง
- CUDA error ที่ทำให้ context ใช้ต่อไม่ได้ (AcceleratorError, illegal memory access, CUDA error อื่นที่ไม่ใช่ OOM): บันทึกหน้านั้นเป็น
  error แล้วหยุด batch ทันที (exit 3) ไม่ข้ามไปหน้าถัดไป (ทุกหน้าที่เหลือจะพังด้วยเหตุเดียวกัน) เขียน summary ก่อนออก
  — รันใหม่ใน process ใหม่เพื่อทำต่อจากแคช
- ที่หยุดทั้ง batch (exit 2): backend ใช้ไม่ได้ (ไม่มี GPU / โหลดโมเดลไม่ได้ — ทุกหน้าจะพังเหมือนกัน) หรือเขียนแคชไม่ได้
  (ทำต่อไปก็ไม่มีอะไรเก็บ) · Ctrl+C = หยุดเรียบร้อย เขียน summary (interrupted) แล้ว exit 130
- ``--pages`` ใช้กับทุกไฟล์ใน ``--files`` · ``--limit N`` = จำนวนหน้าที่ "ต้องรันจริง" สูงสุด (หน้าที่มีแคชไม่นับ)
- ``--cooldown S`` = พัก S วินาทีหลังหน้าที่รันจริง (ok / error) ก่อนหน้าถัดไป ให้ GPU คลายร้อน · หน้า hit ไม่พัก และไม่พัก
  หลังหน้าสุดท้ายของงาน (ไม่กระทบพารามิเตอร์ OCR / cache key — เป็นแค่การรอระหว่างหน้า)
- ETA = หน้าที่เหลือ x ค่าเฉลี่ยเวลาของหน้า miss ที่รันจริงในรอบนี้ (+ cooldown) — คำนวณใหม่ทุกหน้า ไม่มีค่าตายตัว
- log: eval/results/ocr_batch.log (ต่อท้าย) · summary: eval/results/ocr_batch_summary.json (ทับทุกรอบ — นับเฉพาะรอบนั้น)
- Windows: กัน sleep ด้วย SetThreadExecutionState ระหว่างรัน แล้วคืนค่าเดิม (กันปิดฝาเครื่อง/ปิดเครื่องไม่ได้)
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.runtime import is_fatal_cuda, is_oom  # noqa: E402,F401
from app.extraction.ocr import (  # noqa: E402
    OcrBackend, OcrUnavailable, get_backend, hits_ceiling, max_tokens_of, recognize_page_auto,
)
from app.extraction.ocr_cache import OcrCache  # noqa: E402
from app.extraction.pipeline import extract_file  # noqa: E402
from app.models import DetectedType, UnitStatus  # noqa: E402

DATASETS = ROOT / "eval" / "datasets"
RESULTS = ROOT / "eval" / "results"
DEFAULT_FILES = ["ONEREPORTKBANKT.PDF", "E_ONE_REPORT_CPALLT.PDF", "ONEREPORTPTTT.PDF"]  # KBANK -> CPALL -> PTT
DEFAULT_LOG = RESULTS / "ocr_batch.log"
DEFAULT_SUMMARY = RESULTS / "ocr_batch_summary.json"

# สถานะของหน้าในผลลัพธ์
OK, CACHED, ERROR = "ok", "cached", "error"


class BatchAbort(Exception):
    """หยุดทั้ง batch (ไม่ใช่ความล้มเหลวรายหน้า)"""


class CudaFatal(BatchAbort):
    """CUDA error ที่ทำให้ context ใช้ต่อไม่ได้ — ต้องเริ่ม process ใหม่ (exit 3)"""


EXIT_ABORTED, EXIT_CUDA_FATAL, EXIT_INTERRUPTED = 2, 3, 130


# ---------- เลือกหน้า ----------

def parse_pages(spec: Optional[str]) -> Optional[set[int]]:
    """'1-10,15,20-30' -> {1..10, 15, 20..30}; None/ว่าง = ทุกหน้า; รูปแบบผิด -> ValueError"""
    if spec is None or not spec.strip():
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo_s, sep, hi_s = part.partition("-")
        try:
            lo = int(lo_s)
            hi = int(hi_s) if sep else lo
        except ValueError:
            raise ValueError(f"ช่วงหน้าไม่ถูกต้อง: {part!r}") from None
        if lo < 1 or hi < lo:
            raise ValueError(f"ช่วงหน้าไม่ถูกต้อง: {part!r}")
        out.update(range(lo, hi + 1))
    return out or None


def select_pages(needs_ocr_pages: list[int], wanted: Optional[set[int]]) -> list[int]:
    """หน้าที่ต้อง OCR ตามลำดับหน้า (ตัดซ้ำ) กรองด้วย ``wanted`` (None = ทั้งหมด)"""
    pages = sorted(set(needs_ocr_pages))
    return pages if wanted is None else [p for p in pages if p in wanted]


def resolve_file(name: str) -> Path:
    p = Path(name)
    return p if p.exists() else DATASETS / name


# ---------- log / เวลา ----------

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def fmt_dur(seconds: float) -> str:
    s = int(max(seconds, 0))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


class BatchLog:
    """log ทีละหน้า (ต่อท้ายไฟล์ แยกด้วย tab): เวลา · ไฟล์ · หน้า · วินาที(จริง) · วินาทีที่ backend รายงาน · tokens ·
    hit/miss · สถานะ · error — flush ทุกบรรทัด (ปิดเครื่องกลางทางก็ไม่เสีย)"""

    COLUMNS = "time\tfile\tpage\tseconds\tocr_seconds\ttokens\tcache\tstatus\terror"

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, line: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def header(self, note: str) -> None:
        self._write(f"# {now_iso()} {note}")
        self._write(f"# {self.COLUMNS}")

    def page(self, rec: dict) -> None:
        def f(v, fmt="{}"):
            return "-" if v is None else fmt.format(v)

        err = (rec.get("error") or "").replace("\t", " ").replace("\r", " ").replace("\n", " ")
        self._write("\t".join([
            rec["time"], rec["file"], str(rec["page"]), f(rec["seconds"], "{:.1f}"), f(rec.get("ocr_seconds"), "{:.1f}"),
            f(rec.get("tokens")), {True: "hit", False: "miss", None: "-"}[rec.get("hit")], rec["status"], err,
        ]))

    def note(self, text: str) -> None:
        self._write(f"# {now_iso()} {text}")


# ---------- กัน Windows sleep ----------

class KeepAwake:
    """ระหว่างรัน กัน Windows เข้า sleep (SetThreadExecutionState ES_SYSTEM_REQUIRED) แล้วคืนสถานะเดิมตอนออก
    ต้องเข้า/ออกในเธรดเดียวกัน (สถานะผูกกับเธรด) ไม่ใช่ Windows = ไม่ทำอะไร; ไม่กันจอดับและไม่กันการปิดฝาเครื่องที่ตั้งไว้ให้ sleep"""

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and sys.platform == "win32"
        self.previous: Optional[int] = None
        self.active = False

    def _set(self, flags: int) -> int:
        import ctypes

        fn = ctypes.windll.kernel32.SetThreadExecutionState
        fn.argtypes = [ctypes.c_uint]
        fn.restype = ctypes.c_uint
        return int(fn(flags))

    def __enter__(self):
        if self.enabled:
            prev = self._set(self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED)  # คืน 0 = ล้มเหลว
            if prev:
                self.previous, self.active = prev, True
        return self

    def __exit__(self, *exc):
        if self.active:
            # คืนค่าเดิม: ES_CONTINUOUS + แฟล็กเดิม (ถ้าเดิมไม่มีอะไร = ES_CONTINUOUS เปล่า = ล้างที่เราตั้งไว้)
            self._set(self.ES_CONTINUOUS | (self.previous & ~self.ES_CONTINUOUS))
            self.active = False
        return False


# ---------- แคช (ตรวจว่าเขียนได้จริง) ----------

class CheckedCache(OcrCache):
    """OcrCache ที่ตรวจหลังเขียนว่าอ่านกลับได้ — ``OcrCache.put`` ล้มเหลวเงียบ (ดิสก์เต็ม ฯลฯ) ซึ่งเป็นอันตรายกับงานข้ามคืน"""

    def __init__(self, directory: Optional[Path] = None):
        super().__init__(directory)
        self.last_put_ok: Optional[bool] = None

    def put(self, key, fields, result) -> None:
        super().put(key, fields, result)
        self.last_put_ok = self.get(key) is not None


# ---------- OCR ทีละหน้า ----------

# is_oom / is_fatal_cuda ย้ายไป app/runtime.py (ใช้ร่วมกับ ingest worker) — import ไว้ที่หัวไฟล์ ชื่อเดิมยังใช้ได้


def free_gpu_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")  # ไม่ import เอง: ไม่มี torch = ไม่มีอะไรต้องล้าง
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def ocr_one_page(backend: OcrBackend, path: Path, content_hash: str, page_no: int, cache: CheckedCache,
                 *, free_memory: Callable[[], None] = free_gpu_memory,
                 on_piece: Optional[Callable] = None) -> dict:
    """ผ่านแคชตัวเดิมด้วยพารามิเตอร์ปริยาย (ห้ามส่ง dpi / split / params เอง) คืน record ของหน้า
    OOM -> ล้างหน่วยความจำ GPU แล้วลองใหม่ 1 ครั้ง; error อื่นไม่ลองใหม่ (คืน status=error)
    ``OcrUnavailable`` โยนต่อ (ทั้ง batch ใช้ไม่ได้) · CUDA error ที่ context เสีย -> ``rec["fatal_cuda"] = True`` (ผู้เรียกต้องหยุด)"""
    t0 = time.perf_counter()
    rec: dict = {"time": now_iso(), "page": page_no, "hit": None, "tokens": None, "ocr_seconds": None, "error": None}
    attempts = 0
    while True:
        attempts += 1
        cache.last_put_ok = None
        try:
            res, _page, hit, info = recognize_page_auto(backend, path, DetectedType.PDF, content_hash, page_no,
                                                        cache=cache, on_piece=on_piece)
        except OcrUnavailable:
            raise
        except Exception as e:
            if is_oom(e) and attempts == 1:
                free_memory()
                continue
            rec.update(status=ERROR, error=f"{type(e).__name__}: {e}" + (" (หลังลองใหม่หลัง OOM)" if attempts > 1 else ""))
            if is_fatal_cuda(e):  # ไม่ลองใหม่ และให้ run_batch หยุดหลังบันทึกหน้านี้
                rec["fatal_cuda"] = True
            break
        rec.update(hit=hit, tokens=res.tokens, ocr_seconds=res.seconds)
        if hit:
            rec["status"] = CACHED
        elif cache.last_put_ok is False:
            raise BatchAbort(f"เขียนแคชไม่ได้/อ่านกลับไม่ได้ (หน้า {page_no}) ที่ {cache.dir} — ผลที่ OCR ไม่ถูกเก็บ")
        else:
            rec["status"] = OK
            limit = max_tokens_of(backend)
            if info is None and hits_ceiling(res, limit):  # ชนเพดาน = ผลน่าจะถูกตัด/วนลูป
                rec["error"] = f"warn: tokens ถึงเพดาน max_new_tokens={limit}"
        if info is not None:
            rec["resplit"] = {"first_pass_tokens": info.first_pass_tokens, "pieces": info.pieces,
                              "hit_pieces": hits_ceiling(res, max_tokens_of(backend)), "no_gap": info.no_gap,
                              "rotated": info.rotated}
            note = "resplit: " + " ".join(f"{k}={v}" for k, v in rec["resplit"].items())
            rec["error"] = note if not rec.get("error") else f"{rec['error']} · {note}"
        break
    rec["seconds"] = time.perf_counter() - t0
    return rec


# ---------- batch ----------

def estimate_eta(miss_seconds: list[float], left: int, cooldown: float = 0.0) -> Optional[float]:
    """วินาทีที่คาดว่าจะเหลือ = left x ค่าเฉลี่ยของหน้า miss ที่รันจริงในรอบนี้ + cooldown ระหว่างหน้า (ไม่พักหลังหน้าสุดท้าย)
    ยังไม่มีหน้า miss เลย -> None (ไม่มีข้อมูลพอ — ไม่เดาด้วยค่าตายตัว) · ``left`` นับว่าทุกหน้าที่เหลือเป็น miss (ขอบบน)
    หมายเหตุ: หน้า miss หน้าแรกรวมเวลาโหลดโมเดล ทำให้ค่าเฉลี่ยช่วงแรกสูงกว่าจริงเล็กน้อย แล้วค่อยๆ ลดลง"""
    if not miss_seconds or left <= 0:
        return None if not miss_seconds else 0.0
    return left * sum(miss_seconds) / len(miss_seconds) + max(left - 1, 0) * cooldown


def short_name(path: Path) -> str:
    return path.name


def plan_files(files: list[Path], wanted: Optional[set[int]], extractor=extract_file) -> list[dict]:
    """extract ทุกไฟล์ (ด่านตรวจเต็มไฟล์) แล้วเลือกหน้า; ไฟล์ที่ extract ไม่ได้ได้ record ที่มี ``error``"""
    plans = []
    for path in files:
        entry: dict = {"path": path, "name": short_name(path), "pages": [], "needs_ocr_total": 0, "total_pages": 0,
                       "content_hash": None, "error": None}
        try:
            if not path.exists():
                raise FileNotFoundError(f"ไม่พบไฟล์ {path}")
            doc = extractor(path)
            if doc.error is not None:
                raise RuntimeError(f"extract ล้มเหลว: {doc.error.code.value}: {doc.error.message}")
            if doc.detected_type != DetectedType.PDF:
                raise RuntimeError(f"รองรับเฉพาะ PDF (ไฟล์นี้เป็น {doc.detected_type.value if doc.detected_type else '?'})")
            todo = [int(u.position["page"]) for u in doc.units if u.status == UnitStatus.NEEDS_OCR]
            entry.update(content_hash=doc.content_hash, total_pages=len(doc.units), needs_ocr_total=len(todo),
                         pages=select_pages(todo, wanted))
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
        plans.append(entry)
    return plans


def run_batch(files: list[Path], backend: OcrBackend, *, cache: Optional[CheckedCache] = None,
              wanted: Optional[set[int]] = None, limit: Optional[int] = None, log: BatchLog,
              extractor=extract_file, out=print, summary_path: Optional[Path] = None,
              args_info: Optional[dict] = None, free_memory: Callable[[], None] = free_gpu_memory,
              cooldown: float = 0.0, sleep: Callable[[float], None] = time.sleep) -> dict:
    """คืน summary (dict); เขียนลง ``summary_path`` ถ้าให้มา — เขียนเสมอแม้ถูกขัดจังหวะ/หยุดกลางทาง"""
    cache = cache or CheckedCache()
    started = time.perf_counter()
    started_at = now_iso()
    log.header(f"เริ่ม batch backend={backend.name} files={[f.name for f in files]} pages="
               f"{'all' if wanted is None else len(wanted)} limit={limit} cooldown={cooldown}s")
    summary: dict = {
        "started_at": started_at, "finished_at": None, "interrupted": False, "aborted": None, "fatal_cuda": False, "args": args_info or {},
        "backend": backend.name, "cache_dir": str(cache.dir), "total_seconds": 0.0,
        "totals": {"planned": 0, "ok": 0, "cached": 0, "error": 0, "not_reached": 0}, "files": {},
    }

    def finalize() -> dict:
        summary["finished_at"] = now_iso()
        summary["total_seconds"] = round(time.perf_counter() - started, 1)
        t = summary["totals"]
        t.update(planned=0, ok=0, cached=0, error=0, not_reached=0)
        for f in summary["files"].values():
            f["not_reached"] = max(f["selected"] - f["ok"] - f["cached"] - f["error"], 0)
            for k in t:
                if k != "planned":
                    t[k] += f[k]
            t["planned"] += f["selected"]
        if summary_path is not None:
            target = Path(summary_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    try:
        out("วางแผน: extract + ด่านตรวจทั้งไฟล์ (ใช้เวลาสักพัก) ...")
        plans = plan_files(files, wanted, extractor)
        for p in plans:
            summary["files"][p["name"]] = {
                "content_hash": p["content_hash"], "total_pages": p["total_pages"], "needs_ocr_pages": p["needs_ocr_total"],
                "selected": len(p["pages"]), "ok": 0, "cached": 0, "error": 0, "not_reached": 0, "seconds": 0.0,
                "error_pages": [], "file_error": p["error"],
            }
            if p["error"]:
                log.note(f"{p['name']}: ข้ามทั้งไฟล์ — {p['error']}")
                out(f"  {p['name']}: ข้ามทั้งไฟล์ — {p['error']}")
            else:
                out(f"  {p['name']}: needs_ocr {p['needs_ocr_total']}/{p['total_pages']} หน้า · เลือก {len(p['pages'])} หน้า")
        remaining = sum(len(p["pages"]) for p in plans)
        done = attempts = 0
        miss_seconds: list[float] = []
        total_planned = remaining

        def eta() -> str:
            left = total_planned - done
            if limit is not None:
                left = min(left, limit - attempts)
            sec = estimate_eta(miss_seconds, left, cooldown)
            return "ETA -" if sec is None else f"ETA ≤{fmt_dur(sec)}"  # ≤ : ถือว่าหน้าที่เหลือยังไม่มีแคช

        jobs = [(p, page_no) for p in plans if not p["error"] for page_no in p["pages"]]
        for i, (p, page_no) in enumerate(jobs):
            if limit is not None and attempts >= limit:
                break
            fsum = summary["files"][p["name"]]
            def on_piece(piece, res, from_cache, _p=p, _n=page_no):
                log.note(f"piece {_p['name']} p{_n} region={[round(v, 4) for v in piece.region]} depth={piece.depth} "
                         f"tokens={res.tokens} seconds={res.seconds:.1f} {'hit' if from_cache else 'miss'}")
                out(f"    piece p{_n} depth={piece.depth} tokens={res.tokens} {res.seconds:.0f}s {'hit' if from_cache else 'miss'}")

            rec = ocr_one_page(backend, p["path"], p["content_hash"], page_no, cache, free_memory=free_memory,
                               on_piece=on_piece)
            rec["file"] = p["name"]
            log.page(rec)
            done += 1
            fsum["seconds"] = round(fsum["seconds"] + rec["seconds"], 1)
            fsum[rec["status"]] += 1
            real_run = rec["status"] != CACHED
            if real_run:
                attempts += 1
                miss_seconds.append(rec["seconds"])
            if rec["status"] == ERROR:
                fsum["error_pages"].append({"page": page_no, "error": rec["error"]})
            tok = "-" if rec["tokens"] is None else rec["tokens"]
            out(f"[{done}/{total_planned}] {p['name']} p{page_no} {'hit ' if rec['hit'] else 'miss' if rec['hit'] is False else '    '}"
                f" {rec['seconds']:.1f}s tokens={tok} {rec['status']}"
                f"{' — ' + rec['error'] if rec['error'] else ''} | ผ่านไป {fmt_dur(time.perf_counter() - started)} {eta()}")
            if rec.get("fatal_cuda"):  # หน้านี้บันทึกเป็น error แล้ว (log / summary) — ห้ามไปหน้าถัดไป
                raise CudaFatal(f"CUDA error ที่ context ใช้ต่อไม่ได้ ({p['name']} หน้า {page_no}): {rec['error']} — "
                                f"หยุด batch; เริ่ม process ใหม่แล้วรันคำสั่งเดิมเพื่อทำต่อจากแคช")
            more = i + 1 < len(jobs) and not (limit is not None and attempts >= limit)
            if real_run and cooldown > 0 and more:  # พักเฉพาะหลังหน้าที่รันจริง; hit ไม่พัก; ไม่พักท้ายงาน
                sleep(cooldown)
    except KeyboardInterrupt:
        summary["interrupted"] = True
        log.note("ถูกขัดจังหวะ (Ctrl+C) — รันซ้ำเพื่อทำต่อ")
        out("\nขัดจังหวะ (Ctrl+C): รันคำสั่งเดิมซ้ำเพื่อทำต่อจากแคช")
    except (BatchAbort, OcrUnavailable) as e:
        summary["aborted"] = f"{type(e).__name__}: {e}"
        summary["fatal_cuda"] = isinstance(e, CudaFatal)
        log.note(f"หยุด batch — {summary['aborted']}")
        out(f"หยุด batch: {summary['aborted']}")
    return finalize()


def print_summary(s: dict, out=print) -> None:
    out(f"\nสรุป: ใช้เวลา {fmt_dur(s['total_seconds'])} · {s['totals']}")
    for name, f in s["files"].items():
        out(f"  {name}: เลือก {f['selected']} · ok {f['ok']} · แคช {f['cached']} · error {f['error']} · ยังไม่ถึง {f['not_reached']}"
            + (f" · ไฟล์ล้มเหลว: {f['file_error']}" if f["file_error"] else ""))
        for e in f["error_pages"]:
            out(f"    หน้า {e['page']}: {e['error']}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="OCR หน้า needs_ocr แบบ batch ลงแคช (typhoon-local)")
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES, metavar="FILE",
                    help="ชื่อไฟล์ใน eval/datasets หรือ path (ค่าเริ่มต้น: KBANK CPALL PTT ตามลำดับ)")
    ap.add_argument("--pages", help="เลือกช่วงหน้า เช่น 1-20,50 (ใช้กับทุกไฟล์ใน --files)")
    ap.add_argument("--limit", type=int, help="รันจริงไม่เกิน N หน้า (หน้าที่มีแคชไม่นับ) — สำหรับลอง")
    ap.add_argument("--cooldown", type=float, default=0.0, metavar="SEC",
                    help="พัก SEC วินาทีหลังหน้าที่รันจริงก่อนหน้าถัดไป (ค่าเริ่มต้น 0; หน้าที่มีแคชไม่พัก)")
    ap.add_argument("--no-keep-awake", action="store_true", help="ไม่กัน Windows sleep")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        wanted = parse_pages(args.pages)
    except ValueError as e:
        ap.error(str(e))
    if args.limit is not None and args.limit < 1:
        ap.error("--limit ต้อง >= 1")
    if not (0 <= args.cooldown < float("inf")):
        ap.error("--cooldown ต้อง >= 0")

    backend = get_backend("typhoon-local")  # สร้างเฉย ๆ ยังไม่โหลดโมเดล (โหลดตอนมีหน้า miss หน้าแรก)
    log = BatchLog(DEFAULT_LOG)
    with KeepAwake(enabled=not args.no_keep_awake) as ka:
        print(f"กัน Windows sleep: {'เปิด' if ka.active else 'ไม่ได้ใช้'}")
        summary = run_batch(
            [resolve_file(f) for f in args.files], backend, wanted=wanted, limit=args.limit, log=log,
            summary_path=DEFAULT_SUMMARY, cooldown=args.cooldown,
            args_info={"files": args.files, "pages": args.pages, "limit": args.limit, "cooldown": args.cooldown},
        )
    print_summary(summary)
    print(f"log: {DEFAULT_LOG}\nsummary: {DEFAULT_SUMMARY}")
    if summary["interrupted"]:
        return EXIT_INTERRUPTED
    if summary["fatal_cuda"]:
        return EXIT_CUDA_FATAL
    return EXIT_ABORTED if summary["aborted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
