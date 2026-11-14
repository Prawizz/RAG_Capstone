# RAG API Service — API Contract v1 (ฉบับร่างเพื่อตกลงกับทีม Gateway)

สถานะ: **ร่าง v1.0.0 (14 พ.ย. 2569)** · ไฟล์อ้างอิงที่เป็นทางการ: `contract/openapi.json` และตัวอย่างใน `contract/examples/`
ถ้าเอกสารนี้กับ OpenAPI ไม่ตรงกัน ให้ยึด OpenAPI (สร้างจากโค้ดด้วย `scripts/export_openapi.py`, มี `--check` กันเอกสารเก่า)

## 1. ขอบเขตหน้าที่

| เรื่อง | Gateway | RAG API Service (เรา) |
|---|---|---|
| ยืนยันตัวตนผู้ใช้ | ✔ | — (รับค่าที่ Gateway ส่งมา) |
| งบประมาณ, คิว, generate คำตอบ, ข้อความที่แสดงผู้ใช้ | ✔ | — |
| เก็บเอกสาร, OCR, ค้นหลักฐาน, citation | — | ✔ |
| สิทธิ์ของไฟล์และสมาชิกกล่อง (แหล่งความจริงของ ACL) | — | ✔ |
| เลือก LLM ภายใน/ภายนอกจาก `visibility` ในผลค้น | ✔ | ส่งค่า `visibility` ให้ |

เราไม่ generate คำตอบ และไม่ส่งข้อความปฏิเสธที่เป็นภาษาคน — คืน `status` + คะแนน ให้ Gateway ตัดสินและเขียนข้อความเอง

## 2. หลักร่วมทุก endpoint

- prefix `/v1` · JSON UTF-8 (ยกเว้น `/v1/ingest` ที่เป็น multipart)
- **Service auth:** header `X-Service-Key` (ชั่วคราว) — ผิดหรือไม่มี → `401 unauthorized` · **ของจริงต้องตกลงกัน** (mTLS หรือ signed token อายุสั้น) ดูข้อ 6
- **ผู้ใช้ปลายทาง** (ทุก endpoint ยกเว้น health): `X-User-Id`, `X-User-Departments` (คั่นด้วยจุลภาค; ค่าว่าง = ไม่สังกัดแผนก), `X-User-Security-Level` (เลขจำนวนเต็ม ≥ 0) — header หายหรือรูปแบบผิด → `422`
- **สมาชิกกล่อง Gateway ไม่ต้องส่ง** — เราหาเองจากข้อมูลสมาชิกที่เราเก็บ (ป้องกันการปลอมสิทธิ์ผ่าน request)
- **`X-Trace-Id`:** ส่งมาได้ (ถ้าไม่ส่งหรือรูปแบบผิด เราสร้างให้) · ส่งกลับทั้งใน header และ `trace_id` ใน body ทุก response
- **Error ทุกตัวรูปแบบเดียว:** `{error_code, message, trace_id}`

| error_code | HTTP | เมื่อไร |
|---|---|---|
| unauthorized | 401 | service auth ผิด |
| invalid_request | 422 | request/header ผิดรูปแบบ |
| acl_widening | 422 | สิทธิ์ระดับไฟล์กว้างกว่ากล่อง |
| not_found | 404 | ไม่มีอยู่ **หรือไม่มีสิทธิ์** (แยกไม่ได้โดยตั้งใจ) |
| conflict | 409 | ลบไฟล์ที่ job ingest กำลังทำอยู่ (รอ job จบแล้วลบใหม่) |
| too_large | 413 | ไฟล์เกิน 50 MB |
| unsupported_type | 415 | ชนิดไฟล์ไม่รองรับ |
| unavailable | 503 | ส่วนประกอบหลักล่ม (เช่น vector DB) — เราจะไม่ตอบ "ไม่พบข้อมูล" แทน |
| internal | 500 | ข้อผิดพลาดภายใน |

- **ไม่มี 403 ในระบบ:** ไม่มีสิทธิ์ = ไม่เจอเสมอ (404 หรือผลว่าง) เพื่อไม่ให้รู้ได้ว่ามีเอกสารที่ตัวเองไม่มีสิทธิ์อยู่

## 3. ACL ที่ใช้ตัดสิน (4 ชั้น ต้องผ่านครบ)

1. เป็นสมาชิกกล่อง (role: owner / admin / viewer)
2. แผนกตรงกับที่กล่อง/ไฟล์อนุญาต
3. `security_level` ถึงเกณฑ์ของกล่อง/ไฟล์
4. สิทธิ์ระดับไฟล์ทำให้**แคบลง**จากกล่องได้อย่างเดียว

- `visibility` (public/private) เป็นคุณสมบัติของกล่อง เปลี่ยนไม่ได้หลังสร้าง
- ถอดสมาชิกมีผลทันทีกับทุก request ถัดไป (ตรวจซ้ำทุกครั้งก่อนคืนผล)

## 4. Endpoints

| Method | Path | ใช้ทำอะไร | สำเร็จ |
|---|---|---|---|
| GET | `/v1/health/live` | process ยังทำงาน (ไม่ต้อง auth) | 200 |
| GET | `/v1/health/ready` | พร้อมรับงาน (ต้องมี service key) · มี `components` (qdrant/embedding/ocr) และ `runtime` (มี GPU ไหม, device และ OCR backend ที่ใช้จริง, เหตุผลถ้าสลับไปโหมด CPU) · `components.embedding` รวมตัว embed คำถามด้วย ถ้าใช้ไม่ได้ → `not_ready` + เหตุผลใน `notes` · `runtime.embedding_device` คือตัวที่ใช้ตอน ingest ส่วนตัว embed คำถามเป็น CPU เสมอ | 200 ready / 503 degraded, not_ready |
| POST | `/v1/ingest` | ส่งไฟล์เข้ากล่อง (async) | 202 / 200 (ไฟล์ซ้ำ) |
| GET | `/v1/jobs/{job_id}` | สถานะงาน ingest | 200 |
| GET | `/v1/sources` | กล่องที่ผู้ใช้เป็นสมาชิก | 200 |
| GET | `/v1/sources/{box_id}/files` | ไฟล์ในกล่องที่ผู้ใช้เห็นได้ (แบ่งหน้าด้วย `cursor`) | 200 |
| POST | `/v1/retrieve` | ค้นหลักฐานในกล่องเดียว | 200 |
| POST | `/v1/boxes` | สร้างกล่อง (system admin) | 201 |
| PUT | `/v1/boxes/{box_id}/members` | เพิ่ม/เปลี่ยน role สมาชิก | 200 |
| DELETE | `/v1/boxes/{box_id}/members/{user_id}` | ถอดสมาชิก | 200 |
| DELETE | `/v1/files/{file_id}` | ลบไฟล์ออกจากทุกที่ที่เก็บ (ไฟล์ต้นฉบับ, ข้อความ, vector, แคช OCR) | 200 |

### 4.1 POST /v1/ingest
- form: `file`, `box_id`, `file_acl` (JSON ไม่บังคับ: `departments`, `min_security_level`), `display_name` (ไม่บังคับ)
- ผู้ส่งต้องเป็น owner/admin ของกล่อง (ไม่ใช่ → 404)
- ไฟล์เดิม (เนื้อหาเหมือนกัน) ในกล่องเดิม → 200 `duplicate: true` คืน `file_id` เดิม
- ไฟล์ที่ต้อง OCR ใช้เวลาหลายนาทีถึงหลายชั่วโมง → ให้ poll `/v1/jobs/{job_id}`
- ชนิดที่รองรับ: PDF, DOCX, PPTX, XLSX, DOC, PPT, XLS (+ หน้าสแกนใน PDF)
- ตรวจชนิดจากเนื้อไฟล์ (magic bytes) ไม่ใช่นามสกุล · ไฟล์ภาพเดี่ยว (jpg/png) และชนิดที่ไม่รู้จัก → 415
- ไฟล์ที่เข้ารหัสหรือเสีย → รับไว้ (202) แล้ว job จบเป็น failed พร้อม `encrypted` / `corrupt` ให้ดูเหตุผลที่ `/v1/jobs`
- ส่งไฟล์ซ้ำ: job เดิมยังทำอยู่หรือเสร็จแล้ว → คืน job เดิม · job เดิม failed ด้วยเหตุชั่วคราว (`unavailable`, `internal`, `conversion_timeout`) **หรือ** `done_with_gaps` ที่มี gap เป็น `needs_ocr` (OCR ใช้ไม่ได้ตอนนั้น) → เข้าคิวใหม่ ได้ `job_id` ใหม่ `file_id` เดิม (หน้าที่ OCR แล้วไม่ต้องทำซ้ำ · ระหว่าง index ใหม่ผลของไฟล์นี้หายไปชั่วครู่) — ควรส่งซ้ำเมื่อ `/v1/health/ready` รายงาน `ocr` ใช้ได้แล้วเท่านั้น ไม่งั้นจะได้ผลเดิม · failed เพราะตัวไฟล์ → คืน job เดิม (ส่งซ้ำก็ล้มเหมือนเดิม) · ไฟล์ซ้ำใช้ ACL และชื่อเดิม ไม่ใช้ค่าที่ส่งมาใหม่

### 4.2 GET /v1/jobs/{job_id}
- `status`: queued → extracting → ocr → indexing → done | done_with_gaps | failed
- `gaps`: หน้า/ส่วนที่ใช้ไม่ได้พร้อมเหตุผล — ไฟล์ที่มี gap ยังค้นส่วนอื่นได้
  - `failed` = OCR แล้วไม่ผ่านด่านตรวจคุณภาพ (ส่งซ้ำก็ได้ผลเดิม)
  - `needs_ocr` = ยังไม่ได้ OCR เพราะ OCR ใช้ไม่ได้ตอนนั้น (reason ขึ้นต้น `OCR ใช้ไม่ได้:`) — ส่งไฟล์ซ้ำเมื่อพร้อมได้
- ไม่มีหน้าที่ใช้ได้เลยเพราะ OCR ใช้ไม่ได้ → job `failed` ด้วย `unavailable` (ส่งซ้ำได้)
- `progress` นับเป็นหน่วยของไฟล์: หน้า (PDF), หัวข้อ (DOCX), สไลด์ (PPTX), ช่วงตาราง (XLSX)
- `error.code`: `encrypted`, `corrupt`, `unsupported_type`, `too_large`, `empty`, `conversion_timeout`, `internal` หรือ `unavailable` (GPU/vector DB ใช้ไม่ได้ชั่วคราว ส่งไฟล์เดิมซ้ำเพื่อลองใหม่ได้)
- เห็นได้เฉพาะผู้ส่งงานหรือ admin ของกล่อง · ไฟล์ที่ถูกลบแล้ว → 404

### 4.2.1 DELETE /v1/files/{file_id}
- ลบได้เฉพาะ owner/admin ของกล่องที่ไฟล์อยู่ · ไม่มีสิทธิ์ / ไม่มีไฟล์ / ลบไปแล้ว / กำลังลบ → 404 ทั้งหมด
- ไฟล์หายจาก `/v1/sources` และ `/v1/retrieve` **ทันที** ที่ได้ 200 · ส่งไฟล์เดิมเข้ากล่องเดิมใหม่ได้ (ได้ `file_id` ใหม่)
- job ที่ยังรอคิว → ยกเลิกแล้วลบ · job ที่กำลังทำ → 409 `conflict` (รอให้จบแล้วลบใหม่)
- vector DB ล่ม → 503 (ไฟล์ถูกซ่อนแล้ว ระบบลบส่วนที่เหลือให้เองเมื่อกลับมา) · แคช OCR ถูกลบเมื่อไม่มีไฟล์อื่นที่เนื้อหาเหมือนกันใช้อยู่
- `/v1/jobs` ของไฟล์ที่ลบแล้ว → 404

### 4.3 POST /v1/retrieve
request: `{query (1–2000 ตัวอักษร), box_id, top_k (1–20, ค่าเริ่มต้น 5)}`

| status | ความหมาย | Gateway ควรทำ |
|---|---|---|
| `ok` | มีผลที่เกี่ยวข้องกับคำถาม (ไม่ได้รับประกันว่าตอบคำถามได้) | generate คำตอบจาก `results` เท่านั้นพร้อม citation · ถ้าหลักฐานไม่ตอบคำถาม ต้องตอบว่าไม่พบ |
| `no_relevant_context` | ไม่พบหลักฐานที่ดีพอ (ยังอาจมี `results` คะแนนต่ำ) | ตัดสินเองว่าจะปฏิเสธหรือตอบแบบระวัง |
| `source_required` | ไม่ได้ระบุ `box_id` | ให้ผู้ใช้เลือกกล่อง |
| ค่าอื่นที่ไม่รู้จัก | สงวนไว้สำหรับอนาคต (เช่น `source_suggested`) | ทำเหมือน `no_relevant_context` |

- กล่องที่ไม่มีอยู่หรือไม่ได้เป็นสมาชิก → ได้ response **เหมือนกล่องว่างทุกฟิลด์** (`visibility` และ `top_score` เป็น null)
- **`visibility = private` → Gateway ต้อง generate ด้วย LLM ภายในองค์กรเท่านั้น**
- แต่ละผลมี `rank`, `chunk_id`, `text`, `score`, `file_id`, `filename`, `citation` และ `extraction`
  - `rank` = ลำดับจากการค้นแบบผสม (ความหมาย + คำ) · `score` = ความใกล้เชิงความหมาย (cosine 0–1) เทียบข้ามคำถามได้ — **score ไม่จำเป็นต้องลดลงตาม rank** · `top_score` = score สูงสุดในผล
  - `chunk_id` = `{file_id}:u{เลข unit 4 หลัก}:c{เลข chunk 3 หลัก}` · `text` = เนื้อ chunk ตรงกับช่วง `char_start`–`char_end` ในข้อความของ unit (ไม่รวมบรรทัดบริบทที่ระบบเติม)
  - `citation`: `unit_type` (page / slide / section / sheet_region) + ตำแหน่งตามชนิด (`page`, `slide`, `heading_path`, `sheet` + `cell_range`) + `label` สำหรับแสดงผล + `char_start`/`char_end` ในข้อความต้นฉบับของหน่วยนั้น
  - `extraction.method`: text_layer / ocr / office_xml / converted · `flags`: เช่น `ocr_incomplete` (เนื้อหาหน้านั้นอาจไม่ครบ)
- คำถามผ่าน normalizer ภาษาไทยตัวเดียวกับตอน ingest ก่อนค้น
- ตัวเลือก reranker (cross-encoder, ปิดเป็นค่าเริ่มต้น): เปิดแล้ว `rank` = ลำดับจาก reranker · ต้องมี GPU (บน CPU ~15 วินาทีต่อคำถาม) · โหลดไม่ได้ → ready เป็น `not_ready`
- เกณฑ์ `no_relevant_context` (`RETRIEVE_MIN_SCORE`) ยังเป็นค่าชั่วคราว จะ calibrate จากชุดคำถามจริง · ผลวัดเบื้องต้น: score แยกคำถามนอกเรื่อง (คุยเล่น) ได้ แต่แยกคำถามในเรื่องเดียวกันที่เอกสารไม่มีคำตอบไม่ได้ → การปฏิเสธกรณีนี้ต้องให้ LLM ฝั่ง Gateway ตัดสินจาก `results`

## 5. ตัวอย่าง
ดู `contract/examples/*.json` (41 ไฟล์ ครอบคลุมกรณีสำเร็จและ error ของทุก endpoint) เช่น `retrieve_ok.json`, `retrieve_not_member_box.json`, `retrieve_source_required.json`, `ingest_duplicate.json`, `jobs_done_with_gaps.json`

## 6. ค่าเริ่มต้นที่เราใช้ (แก้ได้ผ่านการตั้งค่า ถ้าไม่ตรงกับของจริงแจ้งเฉพาะข้อนั้น)

| เรื่อง | ค่าเริ่มต้นที่เสนอ |
|---|---|
| Service auth | API key ใน `X-Service-Key` ระหว่างพัฒนา · รองรับ mTLS / signed token ได้ภายหลัง |
| ตัวตนผู้ใช้ | เชื่อค่า `X-User-*` จาก Gateway (ผ่าน service auth แล้ว) · ถ้าองค์กรมี JWT ของผู้ใช้ เราตรวจเองได้ |
| แผนก / ระดับความลับ | ชื่อแผนกเป็นข้อความ · ระดับเป็นเลขจำนวนเต็ม (เช่น 0–5) |
| สมาชิกกล่อง | ดูแลผ่าน admin API ของเรา |
| Trace | ถ้า Gateway ส่ง `X-Trace-Id` มา เราใช้ตัวนั้นต่อ |
| OCR | ทำในองค์กรทั้งหมด ไม่ส่งเอกสารออกภายนอก |
| ฮาร์ดแวร์ | **อ้างอิง GPU 8 GB VRAM** (ระดับ RTX 3070 ที่ใช้พัฒนา) สำหรับ Typhoon OCR และ BGE-M3 · ถ้าไม่มี GPU ระบบยังทำงานได้: embedding บน CPU และ OCR ด้วย Tesseract (แม่นน้อยกว่า) · บน CPU วัดได้: embed คำถาม ~50 ms, embed ตอน ingest ~4 chunk/วินาที |
| เวลารอ | ใช้ภายในองค์กร รอได้ระดับหนึ่ง · `/v1/retrieve` วัดได้ p50 85 ms / p95 104 ms (embed คำถามบน CPU ~51 ms, ค้น ~28 ms; 3 เล่ม ~4,600 chunk) · ingest เป็น async (หน้าที่ต้อง OCR ใช้ ~1–4 นาทีต่อหน้าบน GPU อ้างอิง) |
| Demo | ใช้ mock Gateway ที่ทำตาม contract นี้ |
| การ deploy / ที่เก็บข้อมูล | รันแบบ instance เดียว ต้องมี persistent disk (`DATA_DIR`) + Qdrant ในเครือข่ายเดียวกัน · ข้อมูลทั้งหมด (ไฟล์ต้นฉบับ, ข้อความที่ extract, แคช OCR, vector) อยู่ในโครงสร้างพื้นฐานที่บริษัทเลือก (on-prem หรือ cloud ขององค์กร) ไม่ส่งออกภายนอก · หลาย instance / autoscale ยังไม่รองรับ |
| ข้อความที่ส่งออกจากเรา | `/v1/retrieve` คืนข้อความของ chunk ให้ Gateway — การเก็บ log ข้อความนี้ฝั่ง Gateway / LLM เป็นความรับผิดชอบของ Gateway |

## 7. การเปลี่ยนเวอร์ชัน
- เพิ่มฟิลด์หรือค่า enum ใหม่ = ไม่ถือว่าเปลี่ยนเวอร์ชัน (Gateway ต้องข้ามฟิลด์ที่ไม่รู้จัก)
- ลบหรือเปลี่ยนความหมายฟิลด์ = ขึ้น `/v2`
