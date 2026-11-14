# RAG API Service — คู่มือสำหรับ Claude Code

Capstone ร่วมกับบริษัท: **RAG API Service** ที่ต่อเข้ากับ Enterprise AI Gateway + Webchat ที่องค์กรมีอยู่แล้ว
เราไม่ทำแอปใหม่และไม่ generate คำตอบ — Gateway ถือ identity / budget / queue และเป็นคน generate
API ของเรา (contract v1, 11 endpoint): ingest, job status, list sources/files, retrieve, delete file, health, จัดการกล่องและสมาชิก — สเปกอยู่ที่ `docs/API_CONTRACT.md` + `contract/openapi.json`

## อ่านก่อนเริ่มงาน

| เอกสาร | มีอะไร |
|---|---|
| `docs/PLAN.md` | **สถานะล่าสุด (ข้อ 1)**, ลำดับงาน, tracker เรื่องค้าง/ที่ตัดสินแล้ว (ข้อ 0.1) |
| `docs/API_CONTRACT.md` | contract กับ Gateway (สรุปสำหรับคน) — ตัวจริงคือ `contract/openapi.json` |
| `docs/EVAL_QUESTIONS.md` | แนวทางเขียนชุดคำถาม `eval/datasets/questions.jsonl` |
| `README.md` | วิธีรัน API, demo bootstrap, eval, Docker, backup |
| `docs/FORMATS.md` | แนวทางรับมือไฟล์แต่ละประเภท |
| `docs/DATA_FINDINGS.md` | ปัญหาภาษาไทยที่เจอจริงในไฟล์ทดสอบ พร้อมตัวอย่าง |

`docs/` อยู่ใน repo (ยกเว้น `docs/tasks/`) — `docs/PLAN.md` มี tracker เรื่องค้าง อ่านก่อนเริ่มงาน · task spec ส่งมาทาง prompt

**สถานะ (14 พ.ย. 2569):** ingestion + OCR + embedding/Qdrant + API + ACL เสร็จและใช้งานได้จริง (commit `49847f0`) · ช่วงนี้: Evaluation บนชุดคำถามใหม่ (รอ `questions.jsonl`) → mock Gateway + demo → รายงาน · **deadline 3 ธ.ค. 2569** · รายละเอียดดู PLAN ข้อ 1

## โครง repo

```
app/extraction/   extractor ต่อประเภทไฟล์ + pipeline + OCR (Typhoon local / Tesseract, แคช, ตัดลูป, resplit)
app/thai/         normalizer
app/models.py     contract กลาง (Document / Unit / Block / สถานะ)
app/api/          FastAPI routes + schemas (contract กับ Gateway) + services/readiness
app/access.py     ACL 4 ชั้น + สมาชิกกล่อง
app/storage.py    SQLite (DATA_DIR/rag.db): กล่อง, สมาชิก, ACL ไฟล์, ไฟล์, jobs
app/ingest_service.py / ingest_worker.py   รับไฟล์ + worker FIFO ใน process (extract → OCR → chunk → embed → upsert)
app/chunking.py / embedding.py / vectorstore.py   chunk ~1000 ตัว, BGE-M3 dense+sparse, Qdrant + RRF + ACL 2 ด่าน
app/query_embedder.py / reranker.py / file_deleter.py / runtime.py
app/rerank.py, query_expansion.py, consistency.py, citations.py   โมดูลของเพื่อน (option ปิดเป็นค่าเริ่มต้น)
contract/         openapi.json + examples (สร้างด้วย scripts/export_openapi.py --check)
scripts/          bootstrap_demo, retrieval_api_eval, ocr_batch, backup/restore ฯลฯ
eval/             harness + datasets + results
tests/            unit tests (mark: qdrant = ต้องมี Qdrant จริง, slow)
docs/             เอกสารออกแบบ
```

## คำสั่ง

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest          # Windows ใช้ตัวนี้เท่านั้น · เพิ่ม -m qdrant / -m slow ตามต้องการ
docker compose up -d qdrant                  # Qdrant v1.20.0 (named volume)
```
วิธีรัน API / bootstrap / eval อยู่ใน README

## วิธีทำงานของทีม

- ทำบน main อย่างเดียว ไม่แยก branch · Claude Code ไม่ commit / push เอง — ทิ้งงานไว้ให้คน review แล้วทีม commit เอง
- ห้ามแก้ป้ายและค่า truth (judgments, expected ในชุดทดสอบ) · ไม่ชัดหรือต้องเปลี่ยน design ให้หยุดถามก่อน

## กฎเหล็ก — ห้ามละเมิด ถ้าคิดว่าต้องละเมิด ให้หยุดแล้วถามก่อน

1. **Normalizer มีตัวเดียว** — `app/thai/normalizer.py::normalize()` ใช้ทั้งตอน ingest และกับคำถามตอน retrieve ห้ามเขียน normalizer ตัวที่สอง หรือ normalize แบบเฉพาะกิจที่อื่น
2. **ตรวจชนิดไฟล์จาก magic bytes** ห้ามเชื่อนามสกุล (docx/pptx/xlsx ต้องเปิด ZIP ดูข้างใน, doc/ppt/xls ดู OLE stream)
3. **สถานะราย unit** — unit ที่พังต้องไม่ทำให้ทั้งไฟล์พัง ทุก unit ได้สถานะ `ok` / `needs_ocr` / `failed` / `rejected` พร้อมเหตุผลที่อ่านเข้าใจได้
4. **ด่านตรวจคุณภาพตัดสินจากเนื้อข้อความหลัง normalize** ห้ามตัดสินจากคุณสมบัติฟอนต์ (เช่น "ไม่มี ToUnicode = พัง" — ข้อมูลจริงพิสูจน์แล้วว่าผิด ดู DATA_FINDINGS ข้อ 2.4)
5. **Normalize ≠ Repair**
   - normalize = กฎตายตัว ไม่ต้องเดาความหมาย (เช่น `ํา` → `ำ`, ยุบเครื่องหมายที่ซ้ำติดกัน, ตัดอักขระควบคุม)
   - repair = เดาจากหลักฐาน ต้องบันทึกราย unit ว่าใช้กฎอะไร และต้องผ่านเกณฑ์สูง ไม่ถึงเกณฑ์ → `needs_ocr`
   - **ห้ามทำให้ข้อความที่พังดูเหมือนถูกด้วยการเดา** — ข้อความไทยที่ลื่นแต่ผิดอันตรายกว่าข้อความที่อ่านไม่ออก
6. **Ground truth ของ test ห้ามสร้างจาก normalizer หรือ extractor ของเราเอง** ไม่งั้นเป็นการวัดวน
7. **Unit ต่อประเภท** — PDF: หน้า · PPTX: สไลด์ · DOCX: หัวข้อ + ช่วงย่อหน้า · XLSX: ชีต + ช่วงเซลล์ ทุก unit ต้องมี `position` (เครื่องอ่าน) และ `label` (คนอ่าน เช่น "หน้า 47")
8. **Contract** (`app/models.py`, `app/api/schemas.py`) — แก้ได้ แต่ต้องอัปเดตเอกสารที่เกี่ยวข้องใน commit เดียวกัน
9. **สิทธิ์** — Public/Private เป็นคุณสมบัติของ box เปลี่ยนไม่ได้, สิทธิ์ระดับไฟล์ทำให้แคบลงจาก box ได้อย่างเดียว, กรณีไม่มีสิทธิ์ต้องตอบเหมือนกรณีหาไม่เจอทุกประการ
10. **Demo console เรียก RAG ผ่าน API สาธารณะเท่านั้น** ห้าม import module ข้างใน
11. **ห้าม commit ไฟล์ทดสอบขนาดใหญ่เพิ่ม** — PDF 3 ไฟล์ใน `eval/datasets/` (20–45 MB) ถูก commit ไปแล้ว ให้คงไว้ตามเดิม และห้ามแก้ git history เอง
12. **ห้าม fallback เงียบ** — GPU / OCR / embedding / Qdrant ใช้ไม่ได้ต้องบอกชัด (log + `/v1/health/ready` + error code) ห้ามสลับไปใช้ของสำรอง (เช่น hash embedding, in-memory store) โดยไม่บอก
13. **เอกสารไม่ออกนอกองค์กร** — OCR / embedding รันในเครื่อง · OCR ภายนอกต้องเปิดเอง (`OCR_ALLOW_OFFSITE=1`) และกล่อง Private ห้ามส่งออกเด็ดขาด

## ภาษาไทย — รู้ไว้ก่อนเขียนโค้ด

- ภาษาไทยไม่มีช่องว่างระหว่างคำ — อย่าใช้ `split()` นับคำ ใช้ PyThaiNLP (`newmm`) หรือนับอักขระ
- ต่อบรรทัดภาษาไทยห้ามแทรกช่องว่าง (ภาษาอังกฤษต้องแทรก) — ตัดสินเป็นรายจุดต่อ
- ตัดข้อความต้องตัดที่ขอบเขต grapheme (`regex` + `\X`) ไม่ใช่ code point ไม่งั้นได้สระ/วรรณยุกต์ลอย
- เลขไทย ๐–๙ ต้อง normalize ก่อนจับ pattern ด้วย `\d`
- ตัวอย่างปัญหาจริงพร้อมคำตอบที่ถูกอยู่ใน DATA_FINDINGS ข้อ 2

## งานเสร็จเมื่อ

- โค้ด + unit test ผ่าน
- รัน eval แล้วตัวเลขไม่ตกจากรอบก่อน (ถ้าตกต้องเขียนเหตุผลไว้)
- ถ้าแตะ contract อัปเดตเอกสารแล้ว
- เจออะไรที่ทำให้ design ต้องเปลี่ยน **ให้หยุดแล้วรายงาน** อย่าตัดสินใจเปลี่ยน design เอง และสรุปคำถามค้างทั้งหมดไว้ในรายงานตอนจบงาน

## ภาษาที่ใช้

ชื่อตัวแปร ฟังก์ชัน และ commit message เป็นภาษาอังกฤษ ส่วน docstring กับเอกสารเป็นภาษาไทยได้
