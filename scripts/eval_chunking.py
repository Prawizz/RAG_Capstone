"""Script ทดสอบ chunking กับ dataset จริง (rag_evaluation_1000.jsonl)

รัน: python scripts/eval_chunking.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# เพิ่ม root ใน path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import hashlib
from app.chunking import (
    chunk_unit,
    format_citation_context,
    build_system_prompt,
    PARENT_WINDOW_MAX,
)
from app.models import (
    Block, BlockType, ExtractionMethod, QualityReport,
    Unit, UnitStatus, UnitType,
)

EVAL_PATH = ROOT / "eval" / "datasets" / "rag_evaluation_1000.jsonl"


def make_unit(text: str, index: int = 0) -> Unit:
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    return Unit(
        index=index,
        type=UnitType.PAGE,
        position={"page": index + 1},
        label=f"หน้า {index + 1}",
        status=UnitStatus.OK,
        method=ExtractionMethod.TEXT_LAYER,
        blocks=[Block(type=BlockType.PARAGRAPH, text=text, order=0)],
        text=text,
        text_hash=text_hash,
        quality=QualityReport(),
        signals={"char_count": float(len(text))},
    )


def main():
    if not EVAL_PATH.exists():
        print(f"[ERROR] ไม่พบ {EVAL_PATH}")
        sys.exit(1)

    samples = []
    with EVAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"โหลด {len(samples)} ตัวอย่างจาก {EVAL_PATH.name}\n")
    print("=" * 70)

    total = 0
    success = 0
    total_children = 0
    total_parents = 0
    citation_hits = 0
    errors = []

    start_time = time.time()

    for sample in samples:
        qid = sample["question_id"]
        citation_text = sample.get("expected_citation") or ""
        expected_answer = sample.get("expected_answer") or ""

        if not citation_text.strip():
            continue

        total += 1
        try:
            unit = make_unit(citation_text, index=0)
            result = chunk_unit(unit, f"doc-{qid}")

            if result.children:
                success += 1
                total_children += len(result.children)
                total_parents += len(result.parents)

                # ตรวจว่า expected_answer อยู่ใน content
                all_content = " ".join(c.content for c in result.children)
                if expected_answer.strip() in all_content or \
                   expected_answer[:10] in all_content:
                    citation_hits += 1

        except Exception as e:
            errors.append(f"Q{qid}: {e}")

    elapsed = time.time() - start_time

    print(f"{'Metric':<35} {'Value':>15}")
    print("-" * 52)
    print(f"{'ตัวอย่างทั้งหมด':<35} {total:>15}")
    print(f"{'chunk สำเร็จ':<35} {success:>15} ({success/total:.1%})")
    print(f"{'ล้มเหลว':<35} {len(errors):>15}")
    print(f"{'เฉลี่ย children/citation':<35} {total_children/max(success,1):>14.1f}")
    print(f"{'เฉลี่ย parents/citation':<35} {total_parents/max(success,1):>14.1f}")
    print(f"{'Citation text ติดใน chunks':<35} {citation_hits:>12} ({citation_hits/max(success,1):.1%})")
    print(f"{'เวลา (วินาที)':<35} {elapsed:>14.2f}")
    print(f"{'เร็ว (ms/sample)':<35} {elapsed*1000/max(total,1):>13.1f}")
    print("=" * 70)

    if errors:
        print(f"\n[ERRORS] ({len(errors)} รายการ):")
        for e in errors[:5]:
            print(f"  {e}")

    # แสดงตัวอย่าง citation context format
    print("\n--- ตัวอย่าง Citation Context (Q1) ---")
    sample0 = samples[0]
    unit = make_unit(sample0.get("expected_citation", ""), index=0)
    result = chunk_unit(unit, "doc-demo")
    ctx = format_citation_context(result.children)
    for item in ctx:
        print(f"  [{item['sentence_id']}] {item['text'][:60]}...")
        print(f"         offset: [{item['metadata']['char_start']}, {item['metadata']['char_end']}]")

    print("\n--- System Prompt (ย่อ) ---")
    prompt = build_system_prompt()
    print(prompt[:200] + "...")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
