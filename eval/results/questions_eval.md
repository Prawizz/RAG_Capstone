# Evaluation รอบจริง — questions.jsonl ผ่าน POST /v1/retrieve

- ชุดคำถาม `eval/datasets/questions.jsonl` (sha256 `804c86dd9c37…`) · question 238, unanswerable 38, chitchat 12
- กล่อง `annual-reports` (public: CPALL, PTT, KBANK — demo data เดิม) · top_k 10 · ผู้เรียก `somchai.demo` · ค่าตั้งอื่นของ API = ค่าเริ่มต้น (ไม่ได้เปลี่ยน RETRIEVE_MIN_SCORE / SEARCH_MODE / CHUNK_TARGET_CHARS)
- hit = filename ตรง `file` และ citation.page อยู่ใน `pages` · R@1 / R@5 / MRR@10 เฉพาะ type=question · แต่ละ config รีสตาร์ต API ด้วย env ของตัวเอง (ตรวจ `extras` ใน log ทุก request) · GPU ไม่มี OCR ระหว่างวัด
- วัดเมื่อ: baseline_r1 2026-11-23 12:48:00 · baseline_r2 2026-11-23 12:48:25 · reranker 2026-11-23 12:51:26 · query_expansion 2026-11-23 12:52:43 · salience_rerank 2026-11-23 12:54:19

## A. Retrieval (type=question)

| config | n | R@1 | R@5 | MRR@10 | ดีขึ้น / แย่ลง / เท่าเดิม (เทียบ baseline_r1) | R@5 ได้เพิ่ม / เสีย |
|---|---:|---:|---:|---:|---|---|
| baseline_r1 — baseline (ค่าเริ่มต้น) | 238 | 49.6% | 77.3% | 0.616 | (อ้างอิง) |  |
| baseline_r2 — baseline (ค่าเริ่มต้น) — รอบ 2 (process เดิม) | 238 | 49.6% | 77.7% | 0.616 | 8 / 8 / 222 | +1 / −0 |
| reranker — RERANKER=bge-v2-m3 (cuda) | 238 | 60.9% | 84.0% | 0.704 | 64 / 25 / 149 | +19 / −3 |
| query_expansion — QUERY_EXPANSION=1 | 238 | 50.8% | 76.9% | 0.621 | 21 / 18 / 199 | +3 / −4 |
| salience_rerank — SALIENCE_RERANK=1 | 238 | 52.1% | 79.0% | 0.636 | 38 / 22 / 178 | +8 / −4 |

ดีขึ้น/แย่ลง = อันดับแรกที่ hit (ไม่ hit ใน 10 = แย่สุด) · R@5 ได้เพิ่ม/เสีย = ข้อที่ข้ามเส้น top 5

**การแกว่งของ baseline (รันซ้ำใน process เดียวกัน):** อันดับแรกที่ hit ต่างกัน 16/238 คำถาม (q045, q047, q051, q069, q090, q101, q110, q113, q114, q117, q150, q174, q192, q215, q228, q238) · ทุก 288 request: ลำดับ top 10 ต่างกัน 116 · ชุด top 10 ต่างกัน 16 · top_score ต่างกัน 0 (ต่างสูงสุด 0.0000)

### แยกไฟล์

| config | CPALL R@1 | CPALL R@5 | CPALL MRR | PTT R@1 | PTT R@5 | PTT MRR | KBANK R@1 | KBANK R@5 | KBANK MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_r1 | 47.1% | 76.0% | 0.596 | 59.0% | 85.0% | 0.707 | 29.4% | 58.8% | 0.404 |
| baseline_r2 | 48.1% | 76.9% | 0.602 | 58.0% | 85.0% | 0.702 | 29.4% | 58.8% | 0.404 |
| reranker | 57.7% | 84.6% | 0.683 | 68.0% | 92.0% | 0.784 | 50.0% | 58.8% | 0.531 |
| query_expansion | 49.0% | 76.9% | 0.607 | 58.0% | 83.0% | 0.698 | 35.3% | 58.8% | 0.438 |
| salience_rerank | 49.0% | 78.8% | 0.618 | 62.0% | 87.0% | 0.728 | 32.4% | 55.9% | 0.417 |

n ต่อไฟล์: CPALL 104 · PTT 100 · KBANK 34

### แยกประเภทคำถาม (จาก `note` ในชุดคำถาม) — R@5

| config | ทั่วไป (n=198) | ต้อง OCR (n=34) | multi_hop (n=6) | truth หน้าเดียว (n=220) | truth หลายหน้า (n=18) |
|---|---:|---:|---:|---:|---:|
| baseline_r1 | 80.3% | 58.8% | 83.3% | 76.4% | 88.9% |
| baseline_r2 | 80.8% | 58.8% | 83.3% | 76.8% | 88.9% |
| reranker | 87.9% | 58.8% | 100.0% | 83.6% | 88.9% |
| query_expansion | 79.8% | 58.8% | 83.3% | 75.9% | 88.9% |
| salience_rerank | 82.3% | 55.9% | 100.0% | 78.6% | 83.3% |

### latency ต่อ request (ms)

| config | client p50 | client p95 | server p50 | server p95 | embed p50 | search p50 | rerank p50 | rerank p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_r1 | 78 | 193 | 74 | 188 | 49 | 21 | - | - |
| baseline_r2 | 75 | 94 | 71 | 89 | 48 | 17 | - | - |
| reranker | 350 | 445 | 346 | 441 | 54 | 31 | 253 | 348 |
| query_expansion | 79 | 108 | 75 | 103 | 50 | 24 | - | - |
| salience_rerank | 130 | 152 | 126 | 147 | 48 | 25 | 47 | 55 |

client = รอบ HTTP ทั้งหมด (เครื่องเดียวกับ API, เรียกทีละข้อ) · server = embed + search + rerank + map จาก log · warm-up 3 ครั้งไม่นับ · query embed บน CPU fp32 ทุก config (reranker บน cuda)

### ข้อที่พลาด top 5 ทุก config (baseline_r1, reranker, query_expansion, salience_rerank)

- พลาด top 5 ทุก config: **31** ข้อ · ไม่ติด top 10 เลยทุก config: 24 ข้อ
- แยกไฟล์: CPALL 11 · PTT 8 · KBANK 12 · แยกประเภท: ทั่วไป 19 · ต้อง OCR 12 · multi_hop 0
- รายข้อ (top 5): q006, q012, q015, q023, q026, q029, q050, q067, q077, q084, q100, q108, q119, q156, q179, q184, q193, q196, q199, q205, q206, q207, q210, q211, q216, q217, q218, q228, q229, q233, q236

## B. การปฏิเสธ (unanswerable / chitchat) — รายงานอย่างเดียว ไม่ได้เปลี่ยน RETRIEVE_MIN_SCORE

status ที่ได้จริงที่เกณฑ์ปัจจุบัน (0.3, baseline_r1): question: ok=238 · unanswerable: ok=38 · chitchat: ok=12

### top_score (dense cosine, baseline)

| type | n | min | p10 | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| question | 238 | 0.486 | 0.579 | 0.628 | 0.670 | 0.709 | 0.742 | 0.818 |
| unanswerable | 38 | 0.363 | 0.417 | 0.474 | 0.540 | 0.613 | 0.666 | 0.725 |
| chitchat | 12 | 0.453 | 0.466 | 0.486 | 0.534 | 0.559 | 0.569 | 0.594 |
| unanswerable: ใกล้เคียง | 18 | 0.476 | 0.541 | 0.559 | 0.610 | 0.658 | 0.703 | 0.725 |
| unanswerable: อื่น | 20 | 0.363 | 0.396 | 0.423 | 0.474 | 0.523 | 0.550 | 0.642 |

AUC: question vs unanswerable **0.853** · question vs unanswerable: ใกล้เคียง **0.719** · question vs unanswerable: อื่น **0.974** · question vs chitchat **0.962** · question vs (unanswerable+chitchat) **0.879**

| งบคำถามจริงหลุด (calibrate) | เกณฑ์ t | calibrate: question ถูกปฏิเสธผิด | test: question ถูกปฏิเสธผิด | test: unanswerable ปฏิเสธถูก | test: chitchat ปฏิเสธถูก |
|---|---:|---:|---:|---:|---:|
| ≤ 5% (หลัก) | 0.5483 | 5/119 (4.2%) | 3/119 (2.5%) | 11/19 (57.9%) | 4/6 (66.7%) |
| ≤ 2% | 0.5284 | 2/119 (1.7%) | 3/119 (2.5%) | 8/19 (42.1%) | 4/6 (66.7%) |
| ≤ 10% | 0.5677 | 11/119 (9.2%) | 7/119 (5.9%) | 13/19 (68.4%) | 5/6 (83.3%) |

### reranker score อันดับ 1 (bge-v2-m3)

| type | n | min | p10 | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| question | 238 | 0.150 | 0.828 | 0.960 | 0.989 | 0.997 | 0.999 | 1.000 |
| unanswerable | 38 | 0.000 | 0.000 | 0.003 | 0.066 | 0.356 | 0.821 | 0.986 |
| chitchat | 12 | 0.000 | 0.005 | 0.007 | 0.058 | 0.188 | 0.260 | 0.455 |
| unanswerable: ใกล้เคียง | 18 | 0.046 | 0.136 | 0.193 | 0.346 | 0.793 | 0.985 | 0.986 |
| unanswerable: อื่น | 20 | 0.000 | 0.000 | 0.001 | 0.003 | 0.031 | 0.066 | 0.707 |

AUC: question vs unanswerable **0.949** · question vs unanswerable: ใกล้เคียง **0.896** · question vs unanswerable: อื่น **0.997** · question vs chitchat **0.995** · question vs (unanswerable+chitchat) **0.960**

| งบคำถามจริงหลุด (calibrate) | เกณฑ์ t | calibrate: question ถูกปฏิเสธผิด | test: question ถูกปฏิเสธผิด | test: unanswerable ปฏิเสธถูก | test: chitchat ปฏิเสธถูก |
|---|---:|---:|---:|---:|---:|
| ≤ 5% (หลัก) | 0.5513 | 5/119 (4.2%) | 5/119 (4.2%) | 16/19 (84.2%) | 6/6 (100.0%) |
| ≤ 2% | 0.4455 | 2/119 (1.7%) | 3/119 (2.5%) | 16/19 (84.2%) | 5/6 (83.3%) |
| ≤ 10% | 0.8300 | 11/119 (9.2%) | 13/119 (10.9%) | 17/19 (89.5%) | 6/6 (100.0%) |

แบ่งข้อ stratified ตาม type (seed 20261123): calibrate question 119, unanswerable 19, chitchat 6 · test question 119, unanswerable 19, chitchat 6 · เกณฑ์ = ค่าสูงสุดที่คำถามจริงในครึ่ง calibrate ถูกปฏิเสธ (คะแนน < t) ไม่เกินงบ · ตัวเลขฝั่ง test คือผลที่รายงาน

## C. รายการให้คนตรวจ (ระบบไม่ตัดสินเอง)

- pool = ข้อที่ baseline_r1 หรือ reranker พลาดใน top 5: **57** ข้อ · ผู้สมัคร (ข้อ, file, page) จาก top 5 ของ baseline_r1, baseline_r2, reranker: **359** รายการ (ภาพหน้า 237 หน้าไม่ซ้ำ)
- หน้าตรวจ: `eval/datasets/files/questions_review/index.html` (ไม่เข้า git) · รายการ: `eval/results/questions_review_items.json` · คิด R@5 แบบ lenient: `python scripts/questions_eval.py lenient --judgments <ไฟล์ที่ดาวน์โหลด>`

