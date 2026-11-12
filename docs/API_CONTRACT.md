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
| too_large | 413 | ไฟล์เกิน 50 MB |
| unsupported_type | 415 | ชนิดไฟล์ไม่รองรับ |
| unavailable | 503 | ส่วนประกอบหลักล่ม (เช่น vector DB) — เราจะไม่ตอบ "ไม่พบข้อมูล" แทน |
| internal | 500 | ข้อผิดพลาดภายใน |
| not_implemented | 501 | ชั่วคราวระหว่างพัฒนา (จะหายในเวอร์ชันใช้งานจริง) |

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
| GET | `/v1/health/ready` | พร้อมรับงาน (ต้องมี service key) · มี `components` (qdrant/embedding/ocr) และ `runtime` (มี GPU ไหม, device และ OCR backend ที่ใช้จริง, เหตุผลถ้าสลับไปโหมด CPU) | 200 ready / 503 degraded, not_ready |
| POST | `/v1/ingest` | ส่งไฟล์เข้ากล่อง (async) | 202 / 200 (ไฟล์ซ้ำ) |
| GET | `/v1/jobs/{job_id}` | สถานะงาน ingest | 200 |
| GET | `/v1/sources` | กล่องที่ผู้ใช้เป็นสมาชิก | 200 |
| GET | `/v1/sources/{box_id}/files` | ไฟล์ในกล่องที่ผู้ใช้เห็นได้ (แบ่งหน้าด้วย `cursor`) | 200 |
| POST | `/v1/retrieve` | ค้นหลักฐานในกล่องเดียว | 200 |
| POST | `/v1/boxes` | สร้างกล่อง (system admin) | 201 |
| PUT | `/v1/boxes/{box_id}/members` | เพิ่ม/เปลี่ยน role สมาชิก | 200 |
| DELETE | `/v1/boxes/{box_id}/members/{user_id}` | ถอดสมาชิก | 200 |

### 4.1 POST /v1/ingest
- form: `file`, `box_id`, `file_acl` (JSON ไม่บังคับ: `departments`, `min_security_level`), `display_name` (ไม่บังคับ)
- ผู้ส่งต้องเป็น owner/admin ของกล่อง (ไม่ใช่ → 404)
- ไฟล์เดิม (เนื้อหาเหมือนกัน) ในกล่องเดิม → 200 `duplicate: true` คืน `file_id` เดิม
- ไฟล์ที่ต้อง OCR ใช้เวลาหลายนาทีถึงหลายชั่วโมง → ให้ poll `/v1/jobs/{job_id}`
- ชนิดที่รองรับ: PDF, DOCX, PPTX, XLSX, DOC, PPT, XLS (+ หน้าสแกนใน PDF)

### 4.2 GET /v1/jobs/{job_id}
- `status`: queued → extracting → ocr → indexing → done | done_with_gaps | failed
- `gaps`: หน้า/ส่วนที่ใช้ไม่ได้พร้อมเหตุผล (เช่น OCR แล้วไม่ผ่านด่านตรวจคุณภาพ) — ไฟล์ที่มี gap ยังค้นส่วนอื่นได้
- เห็นได้เฉพาะผู้ส่งงานหรือ admin ของกล่อง

### 4.3 POST /v1/retrieve
request: `{query (1–2000 ตัวอักษร), box_id, top_k (1–20, ค่าเริ่มต้น 5)}`

| status | ความหมาย | Gateway ควรทำ |
|---|---|---|
| `ok` | มีหลักฐานที่น่าเชื่อถือ | generate คำตอบพร้อม citation |
| `no_relevant_context` | ไม่พบหลักฐานที่ดีพอ (ยังอาจมี `results` คะแนนต่ำ) | ตัดสินเองว่าจะปฏิเสธหรือตอบแบบระวัง |
| `source_required` | ไม่ได้ระบุ `box_id` | ให้ผู้ใช้เลือกกล่อง |
| ค่าอื่นที่ไม่รู้จัก | สงวนไว้สำหรับอนาคต (เช่น `source_suggested`) | ทำเหมือน `no_relevant_context` |

- กล่องที่ไม่มีอยู่หรือไม่ได้เป็นสมาชิก → ได้ response **เหมือนกล่องว่างทุกฟิลด์** (`visibility` และ `top_score` เป็น null)
- **`visibility = private` → Gateway ต้อง generate ด้วย LLM ภายในองค์กรเท่านั้น**
- แต่ละผลมี `text`, `score`, `file_id`, `filename`, `citation` และ `extraction`
  - `citation`: `unit_type` (page / slide / section / sheet_region) + ตำแหน่งตามชนิด (`page`, `slide`, `heading_path`, `sheet` + `cell_range`) + `label` สำหรับแสดงผล + `char_start`/`char_end` ในข้อความต้นฉบับของหน่วยนั้น
  - `extraction.method`: text_layer / ocr / office_xml / converted · `flags`: เช่น `ocr_incomplete` (เนื้อหาหน้านั้นอาจไม่ครบ)
- เกณฑ์ `no_relevant_context` (`RETRIEVE_MIN_SCORE`) ยังเป็นค่าชั่วคราว จะ calibrate จากชุดคำถามจริง

## 5. ตัวอย่าง
ดู `contract/examples/*.json` (38 ไฟล์ ครอบคลุมกรณีสำเร็จและ error ของทุก endpoint) เช่น `retrieve_ok.json`, `retrieve_not_member_box.json`, `retrieve_source_required.json`, `ingest_duplicate.json`, `jobs_done_with_gaps.json`

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
| เวลารอ | ใช้ภายในองค์กร รอได้ระดับหนึ่ง · `/v1/retrieve` เป้าหมายภายในไม่กี่วินาที · ingest เป็น async (หน้าที่ต้อง OCR ใช้ ~1–4 นาทีต่อหน้าบน GPU อ้างอิง) |
| Demo | ใช้ mock Gateway ที่ทำตาม contract นี้ |

## 7. การเปลี่ยนเวอร์ชัน
- เพิ่มฟิลด์หรือค่า enum ใหม่ = ไม่ถือว่าเปลี่ยนเวอร์ชัน (Gateway ต้องข้ามฟิลด์ที่ไม่รู้จัก)
- ลบหรือเปลี่ยนความหมายฟิลด์ = ขึ้น `/v2`
