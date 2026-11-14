# cross-encoder reranker (BAAI/bge-reranker-v2-m3) บนชุด dev ผ่าน /v1/retrieve

- rerank 20 อันดับแรกจาก RRF แล้วคืน top 10 · hit / ตัวชี้วัด = นิยามเดียวกับ scripts/retrieval_dev_eval.py (ไม่ปรับพารามิเตอร์/เกณฑ์ใดๆ) · score ใน API ยังเป็น dense cosine (ไม่ได้เปลี่ยน)

## Recall@1 / Recall@5 / MRR

| รอบ | (ก) 50 ข้อ R@1 | R@5 | MRR | (ข) 70 ข้อ R@1 | R@5 | MRR | CPALL MRR | PTT MRR | KBANK MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RERANKER=off | 42.0% | 66.0% | 0.533 | 34.3% | 52.9% | 0.432 | 0.636 | 0.437 | 0.189 |
| RERANKER=bge-v2-m3 (cpu) | 46.0% | 74.0% | 0.588 | 38.6% | 60.0% | 0.485 | 0.672 | 0.503 | 0.241 |
| RERANKER=bge-v2-m3 (cuda bf16) | 46.0% | 74.0% | 0.588 | 38.6% | 60.0% | 0.485 | 0.672 | 0.503 | 0.241 |

รายข้อเทียบกับ RERANKER=off (อันดับแรกที่ hit ใน 70 ข้อ):

- RERANKER=bge-v2-m3 (cpu): ดีขึ้น 18 · แย่ลง 8 · เท่าเดิม 44
- RERANKER=bge-v2-m3 (cuda bf16): ดีขึ้น 18 · แย่ลง 8 · เท่าเดิม 44

## latency ต่อ request (ms)

| รอบ | ช่วง | p50 | p95 | max |
|---|---|---:|---:|---:|
| RERANKER=off | client | 135 | 161 | 169 |
| RERANKER=off | embed | 105 | 129 | 136 |
| RERANKER=off | search | 20 | 36 | 48 |
| RERANKER=off | server_total | 130 | 156 | 164 |
| RERANKER=bge-v2-m3 (cpu) | client | 14847 | 22104 | 22593 |
| RERANKER=bge-v2-m3 (cpu) | embed | 110 | 135 | 903 |
| RERANKER=bge-v2-m3 (cpu) | search | 31 | 50 | 63 |
| RERANKER=bge-v2-m3 (cpu) | rerank | 14579 | 21925 | 22437 |
| RERANKER=bge-v2-m3 (cpu) | server_total | 14839 | 22099 | 22588 |
| RERANKER=bge-v2-m3 (cuda bf16) | client | 328 | 418 | 462 |
| RERANKER=bge-v2-m3 (cuda bf16) | embed | 78 | 135 | 188 |
| RERANKER=bge-v2-m3 (cuda bf16) | search | 44 | 64 | 81 |
| RERANKER=bge-v2-m3 (cuda bf16) | rerank | 191 | 246 | 265 |
| RERANKER=bge-v2-m3 (cuda bf16) | server_total | 324 | 411 | 455 |

## คะแนนอันดับ 1: แยก unanswerable / chitchat ออกจากคำถามจริงได้ไหม

AUC = โอกาสที่คำถามจริง (type=question) ได้คะแนนอันดับ 1 สูงกว่า unanswerable/chitchat (0.5 = แยกไม่ได้, 1 = แยกได้หมด)

| รอบ | คะแนน | question vs unanswerable | question vs chitchat | hit@1 vs ไม่ hit@1 |
|---|---|---:|---:|---:|
| RERANKER=off | dense cosine (top_score) | 0.885 | 1.000 | 0.380 |
| RERANKER=bge-v2-m3 (cpu) | dense cosine (top_score) | 0.886 | 1.000 | 0.402 |
| RERANKER=bge-v2-m3 (cpu) | reranker (rerank_top) | 0.963 | 1.000 | 0.449 |
| RERANKER=bge-v2-m3 (cuda bf16) | dense cosine (top_score) | 0.886 | 1.000 | 0.402 |
| RERANKER=bge-v2-m3 (cuda bf16) | reranker (rerank_top) | 0.964 | 1.000 | 0.447 |

### การกระจายคะแนน reranker อันดับ 1 — RERANKER=bge-v2-m3 (cpu)

| กลุ่ม | n | min | p10 | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| question: hit@1 | 27 | 0.175 | 0.732 | 0.940 | 0.990 | 0.997 | 0.998 | 1.000 |
| question: hit อันดับ 2–10 | 21 | 0.227 | 0.636 | 0.962 | 0.993 | 0.999 | 0.999 | 1.000 |
| question: ไม่ hit ใน 10 | 22 | 0.617 | 0.904 | 0.969 | 0.988 | 0.995 | 0.999 | 1.000 |
| chitchat | 10 | 0.000 | 0.000 | 0.001 | 0.003 | 0.005 | 0.019 | 0.023 |
| unanswerable | 20 | 0.008 | 0.036 | 0.046 | 0.138 | 0.455 | 0.863 | 0.911 |

คำถามจริงที่คะแนนสูงกว่า unanswerable ทุกข้อ (> 0.911): 57/70 · unanswerable ที่คะแนนต่ำกว่าคำถามจริงทุกข้อ (< 0.175): 12/20

### การกระจายคะแนน reranker อันดับ 1 — RERANKER=bge-v2-m3 (cuda bf16)

| กลุ่ม | n | min | p10 | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| question: hit@1 | 27 | 0.179 | 0.732 | 0.939 | 0.989 | 0.997 | 0.998 | 1.000 |
| question: hit อันดับ 2–10 | 21 | 0.224 | 0.633 | 0.962 | 0.993 | 0.999 | 0.999 | 1.000 |
| question: ไม่ hit ใน 10 | 22 | 0.614 | 0.903 | 0.969 | 0.988 | 0.995 | 0.999 | 1.000 |
| chitchat | 10 | 0.000 | 0.000 | 0.001 | 0.003 | 0.005 | 0.019 | 0.023 |
| unanswerable | 20 | 0.008 | 0.035 | 0.046 | 0.140 | 0.457 | 0.860 | 0.914 |

คำถามจริงที่คะแนนสูงกว่า unanswerable ทุกข้อ (> 0.914): 57/70 · unanswerable ที่คะแนนต่ำกว่าคำถามจริงทุกข้อ (< 0.179): 12/20
