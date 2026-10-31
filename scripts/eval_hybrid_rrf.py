"""การประเมินผลระบบ Hybrid Search (BM25 + BGE-M3 Dense + RRF) เทียบกับ Dataset จริง
รัน: python scripts/eval_hybrid_rrf.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.chunking import chunk_document, chunk_unit
from app.citations import (
    build_citations_from_chunks,
    evaluate_citation_quality,
    compute_nli_entailment_score,
)
from app.extraction.pipeline import extract_file
from app.models import Unit, UnitStatus, UnitType, Block, BlockType, QualityReport, ExtractionMethod
from app.retrieval import HybridRetriever, FALLBACK_ABSTAIN_RESPONSE
from app.thai.normalizer import normalize

EVAL_PATH = ROOT / "eval" / "datasets" / "rag_evaluation_1000.jsonl"
PDF_PATH = ROOT / "eval" / "datasets" / "E_ONE_REPORT_CPALLT.PDF"


def make_unit_from_text(text: str, page_num: int = 1) -> Unit:
    import hashlib
    return Unit(
        index=page_num - 1,
        type=UnitType.PAGE,
        position={"page": page_num},
        label=f"หน้า {page_num}",
        status=UnitStatus.OK,
        method=ExtractionMethod.TEXT_LAYER,
        blocks=[Block(type=BlockType.PARAGRAPH, text=text, order=0)],
        text=text,
        text_hash=hashlib.sha256(text.encode()).hexdigest(),
        quality=QualityReport(),
        signals={"char_count": float(len(text))},
    )


def run_evaluation(num_samples: int = 100):
    print("=" * 85)
    print(" 🚀 การประเมิน Hybrid Search (BM25 + Dense + RRF) บนชุดข้อมูลจริง")
    print("=" * 85)

    if not EVAL_PATH.exists():
        print(f"[ERROR] ไม่พบ {EVAL_PATH}")
        return

    # 1. โหลดตัวอย่างจาก dataset
    samples = []
    with EVAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("expected_citation"):
                samples.append(item)
            if len(samples) >= num_samples:
                break

    print(f"[1] โหลดชุดข้อมูลประเมินจำนวน {len(samples)} ข้อ")

    # 2. ทำ Chunking ผ่าน app/chunking.py เพื่อสร้างดรรชนีค้นหา
    units = []
    for i, s in enumerate(samples):
        cite_text = s["expected_citation"]
        units.append(make_unit_from_text(cite_text, page_num=i + 1))

    t0 = time.time()
    children, parents = chunk_document(units, document_id="E_ONE_REPORT_CPALLT.PDF")
    chunk_time = (time.time() - t0) * 1000

    print(f"[2] สร้าง Chunks สำเร็จ: {len(children)} Child Chunks, {len(parents)} Parent Chunks (ใช้เวลา {chunk_time:.1f} ms)")

    # 3. เตรียม Hybrid Retriever
    retriever = HybridRetriever(rrf_k=60)
    t0 = time.time()
    retriever.index(children, parents)
    index_time = (time.time() - t0) * 1000
    print(f"[3] สร้าง Hybrid Index (BM25 newmm + Dense 1024 dim) ใช้เวลา {index_time:.1f} ms")

    # 4. ทดสอบ Query แต่ละข้อ
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    abstain_count = 0
    latencies = []

    print(f"\n[4] เริ่มทำการทดสอบ Retrieval (RRF k=60)...")
    for s in samples:
        query = s["query"]
        expected_ans = normalize(s.get("expected_answer", ""))

        t_start = time.time()
        results, decision = retriever.retrieve(query, top_k=5, fusion_method="rrf")
        latencies.append((time.time() - t_start) * 1000)

        if decision.should_abstain:
            abstain_count += 1
            continue

        # ตรวจสอบความถูกต้องว่า expected_answer ปรากฏใน Top-K Child หรือ Parent
        top1_found = False
        top3_found = False
        top5_found = False

        for rank, r in enumerate(results, start=1):
            child_content = normalize(r.child.content)
            parent_content = normalize(r.parent.content) if r.parent else ""
            found = (expected_ans in child_content) or (expected_ans in parent_content)

            if found:
                if rank == 1:
                    top1_found = True
                if rank <= 3:
                    top3_found = True
                if rank <= 5:
                    top5_found = True
                break

        if top1_found:
            top1_hits += 1
        if top3_found:
            top3_hits += 1
        if top5_found:
            top5_hits += 1

    total = len(samples)
    print("\n" + "=" * 85)
    print(f" {'ตัวชี้วัด (Metric)':<40} | {'ผลลัพธ์':>15}")
    print("=" * 85)
    print(f" {'จำนวนคำถามที่ทดสอบ':<40} | {total:>15}")
    print(f" {'Top-1 Recall (เจอในอันดับที่ 1)':<40} | {top1_hits/total:>14.1%}")
    print(f" {'Top-3 Recall (เจอใน Top-3)':<40} | {top3_hits/total:>14.1%}")
    print(f" {'Top-5 Recall (เจอใน Top-5)':<40} | {top5_hits/total:>14.1%}")
    print(f" {'อัตราการ Abstain':<40} | {abstain_count/total:>14.1%}")
    print(f" {'Latency เฉลี่ยต่อคำถาม (ms)':<40} | {sum(latencies)/len(latencies):>14.2f}")
    print("=" * 85)

    # 5. แสดงตัวอย่าง Citation Pack และ ALCE Validation
    sample_q = samples[2]  # คำถามที่ 3: รายได้รวม
    print(f"\n--- ตัวอย่างการแพ็กเกจส่ง AI Gateway (ALCE / LongCite Format) ---")
    print(f"คำถาม: {sample_q['query']}")
    res, _ = retriever.retrieve(sample_q["query"], top_k=2)
    payload = retriever.prepare_llm_payload(res)

    print(f"\n📦 Parent Context สำหรับ LLM Generation (บริบทเต็ม):")
    for i, pc in enumerate(payload["parent_contexts"][:1], 1):
        print(f"   [{i}] {pc[:140]}...")

    print(f"\n🏷️ Sentence-level Citation Format ที่ส่งให้ LLM:")
    for c in payload["citation_context"][:2]:
        print(f"   [{c['sentence_id']}] \"{c['text'][:60]}...\" (Offset: {c['metadata']['char_start']}-{c['metadata']['char_end']})")


if __name__ == "__main__":
    run_evaluation(num_samples=100)

