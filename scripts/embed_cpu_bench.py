"""วัด BGE-M3 บน CPU แบบสั้น: โหมด CPU ใช้งานได้จริงไหม (ไม่วัด OCR)

    python scripts/embed_cpu_bench.py                # ซ่อน GPU (CUDA_VISIBLE_DEVICES=-1) แล้วใช้ EMBEDDING_DEVICE=auto
    python scripts/embed_cpu_bench.py --questions 20 --chunks 50

- ซ่อน GPU ก่อน import torch -> ``auto`` ต้องตัดสินเป็น cpu (fp32) ผ่าน ``app.runtime`` ตัวเดียวกับระบบจริง (ถ้าไม่ใช่ cpu = หยุด)
- คำถาม = ``query`` ของ eval/datasets/rag_evaluation_100.jsonl (type=question) N ข้อแรก: embed ทีละข้อ วัด latency ต่อข้อ
  (p50/p95, ไม่นับรอบอุ่นเครื่อง 1 ครั้ง)
- chunk = chunk จริงของ CPALL (unit ok, ``CHUNK_TARGET_CHARS``) N ตัวแรก ผ่าน ``vectorstore.embed_chunks`` (ทางเดียวกับตอน index:
  มี context นำหน้า, batch ตาม ``EMBEDDING_BATCH_SIZE``) วัดเวลารวม -> chunks/วินาที
- ผล: eval/results/embed_cpu_bench.json
"""

from __future__ import annotations

import os

# ซ่อน GPU ก่อน import torch (ทางอ้อมผ่าน app.embedding) — ใช้ "-1" ไม่ใช่ "": บน Windows การตั้งค่าว่างคือการลบตัวแปร
# (CRT _putenv("X=")) CUDA runtime จึงยังเห็น GPU; "-1" = ไม่มี device index นี้ = ไม่เห็น GPU ทั้ง Windows และ Linux
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop("EMBEDDING_TEST_MODE", None)

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.chunking import make_document_chunks  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.embedding import get_embedder  # noqa: E402
from app.extraction.pipeline import extract_file  # noqa: E402
from app.models import UnitStatus  # noqa: E402
from app.runtime import configure_logging, cuda_available  # noqa: E402
from app.vectorstore import embed_chunks  # noqa: E402

DATASETS = ROOT / "eval" / "datasets"
QUESTIONS = DATASETS / "rag_evaluation_100.jsonl"
CHUNK_SOURCE = DATASETS / "E_ONE_REPORT_CPALLT.PDF"
OUT = ROOT / "eval" / "results" / "embed_cpu_bench.json"


def percentile(values: list[float], q: float) -> float:
    """nearest-rank"""
    s = sorted(values)
    k = max(1, min(len(s), round(q / 100 * len(s) + 0.5)))
    return s[k - 1]


def main() -> int:
    ap = argparse.ArgumentParser(description="วัด BGE-M3 บน CPU")
    ap.add_argument("--questions", type=int, default=20)
    ap.add_argument("--chunks", type=int, default=50)
    args = ap.parse_args()
    configure_logging()

    import torch

    settings = get_settings()
    if settings.embedding_test_mode:
        print("EMBEDDING_TEST_MODE เปิดอยู่ — ต้องวัดโมเดลจริง")
        return 2
    print(f"cuda_available={cuda_available()} · torch threads={torch.get_num_threads()} · {platform.processor()}")

    t0 = time.perf_counter()
    embedder = get_embedder(settings)
    load_s = time.perf_counter() - t0
    if embedder.device != "cpu":
        print(f"ได้ device {embedder.device} ไม่ใช่ cpu — หยุด")
        return 2
    print(f"โหลดโมเดล {load_s:.1f}s · device={embedder.device} dtype={embedder.dtype}")

    rows = [json.loads(x) for x in QUESTIONS.read_text(encoding="utf-8").splitlines() if x.strip()]
    queries = [r["query"] for r in rows if r.get("type") == "question"][: args.questions]
    embedder.embed_query("อุ่นเครื่อง")
    lat = []
    for q in queries:
        t = time.perf_counter()
        embedder.embed_query(q)
        lat.append((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    doc = extract_file(CHUNK_SOURCE)
    extract_s = time.perf_counter() - t
    ok_units = [u for u in doc.units if u.status == UnitStatus.OK]
    chunks = make_document_chunks(ok_units, settings.chunk_target_chars)[: args.chunks]
    embed_chunks(chunks[:2], embedder)  # อุ่นเครื่องเส้นทาง batch
    t = time.perf_counter()
    embed_chunks(chunks, embedder)
    chunk_s = time.perf_counter() - t

    lens = [len(c.embed_text) for c in chunks]
    out = {
        "device": embedder.device, "dtype": str(embedder.dtype), "torch_threads": torch.get_num_threads(),
        "cpu": platform.processor(), "model_load_s": round(load_s, 1),
        "queries": len(lat), "query_chars_median": statistics.median(len(q) for q in queries),
        "query_ms_p50": round(percentile(lat, 50), 1), "query_ms_p95": round(percentile(lat, 95), 1),
        "query_ms_max": round(max(lat), 1),
        "chunks": len(chunks), "chunk_chars_median": statistics.median(lens), "chunk_chars_max": max(lens),
        "batch_size": settings.embedding_batch_size, "chunks_total_s": round(chunk_s, 1),
        "chunks_per_s": round(len(chunks) / chunk_s, 2),
        "retries": embedder.retries, "fp32_fallbacks": embedder.fp32_fallbacks,
        "extract_s_not_counted": round(extract_s, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for k, v in out.items():
        print(f"  {k}: {v}")
    print(f"เขียน {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
