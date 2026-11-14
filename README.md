# RAG API Service

RAG API Service สำหรับต่อกับ Enterprise AI Gateway (ingest, retrieve, list sources, job status, health)
ดูกฎและโครง repo ใน [CLAUDE.md](CLAUDE.md)

## Setup

ต้องใช้ Python >= 3.10

ทีมใช้ Python 3.10.x (`requirements.txt` pin เวอร์ชันบน 3.10.11) ถ้าใช้เวอร์ชันอื่นอาจติดตั้งบางแพ็กเกจไม่ผ่าน

```bash
python -m venv .venv
source .venv/bin/activate        # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

**torch (GPU) ติดตั้งแยก** — `requirements.txt` มี `transformers` แต่ไม่มี `torch` เพราะต้องเลือกชุด CUDA ให้ตรงเครื่อง
ที่ต้องใช้ torch: embedding จริง (BGE-M3 ใน `app/embedding.py`) และ OCR ในเครื่อง (Typhoon) — ถ้าไม่ลง `get_embedder()` จะ raise `EmbeddingUnavailable`
(test ปกติไม่ต้องใช้ เพราะใช้ hash vector ในโหมด test)

```bash
# Windows/Linux + GPU NVIDIA (CUDA 12.8) — เลือก index ตามเวอร์ชัน CUDA ของไดรเวอร์
pip install -r requirements-ocr.txt --extra-index-url https://download.pytorch.org/whl/cu128
# หรือแค่ torch อย่างเดียว (ถ้าไม่ใช้ OCR ในเครื่อง)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"   # ต้องเป็น True
```

โมเดล BAAI/bge-m3 (~2.3 GB) ถูกดาวน์โหลดจาก Hugging Face ครั้งแรกที่ใช้ และเก็บใน cache ของเครื่อง

ถ้า `pytest` ผ่านทั้งหมด แปลว่า dev env พร้อมใช้งาน

หมายเหตุสำหรับ Windows + OneDrive: repo มี `.gitattributes` บังคับ line ending เป็น LF
สำหรับไฟล์ข้อความ ไฟล์ binary (pdf, docx, pptx, xlsx, doc, ppt, xls) จะไม่ถูกแก้

## รัน extraction ดูผล (CLI)

```bash
python scripts/extract_cli.py path/to/file.pdf                    # พิมพ์สรุปอย่างเดียว
python scripts/extract_cli.py path/to/file.pdf --out result.json  # เขียน Document เป็น JSON (ไทยไม่ถูก escape)
python scripts/extract_cli.py path/to/file.pdf --pages 1-20       # ลองเฉพาะบางหน้า (PDF)
python scripts/extract_cli.py path/to/file.docx                   # DOCX: 1 unit ต่อหัวข้อ (ดู app/extraction/docx.py)
```

สรุปที่พิมพ์: ชื่อไฟล์, ชนิด, จำนวนหน้า, จำนวน unit ต่อสถานะ, เวลารวม, เวลาเฉลี่ยและสูงสุดต่อหน้า, error (ถ้ามี)
exit code เป็น 1 เมื่อทั้งไฟล์ล้มเหลว (`Document.error`)

## Test

```bash
pytest            # test ปกติ (เร็ว) — ไม่ต้องมี docker
pytest -m slow    # test ช้า: ใช้ PDF จริงใน eval/datasets/ และ mock_corpus.pdf
pytest -m qdrant  # test ที่ต้องมี Qdrant จริง (ดูหัวข้อถัดไป); ไม่มี server = ข้าม ไม่ใช่ fail
```

บน Windows ต้องตั้ง `PYTHONUTF8=1` ไม่งั้น `tests/test_convert.py` ล้ม 6 ข้อ
(subprocess เขียนข้อความไทยออก stdout ที่เป็น cp1252) — PowerShell: `$env:PYTHONUTF8=1`

## Vector store + สิทธิ์ (ACL)

```bash
docker compose up -d qdrant        # http://localhost:6333/dashboard
cp .env.example .env               # แก้ค่าได้ถ้าไม่ได้ใช้ค่าเริ่มต้น
```

Qdrant เก็บข้อมูลใน named volume `qdrant_storage` (ไม่ใช้ bind mount เพราะ repo ที่อยู่ใต้ OneDrive บน Windows ทำให้ I/O ช้า)
`docker compose down` ข้อมูลยังอยู่ · `docker compose down -v` ลบข้อมูล · image ต้องตรง minor กับ `qdrant-client` ใน `requirements.txt`

- `app/config.py` — ค่าตั้งทั้งหมดอ่านที่เดียวจาก env + `.env`
- `app/access.py` — ACL 4 ชั้น (box · แผนก · ระดับ · สิทธิ์ระดับไฟล์) ตามกฎเหล็กข้อ 9
- `app/vectorstore.py` — Qdrant: Public/Private แยก collection, index เฉพาะ unit `ok`, สิทธิ์ 2 ด่าน
  (pre-filter ใน Qdrant + re-check กับ ACL store ก่อนคืนผล) · 1 point = 1 chunk · named vectors 3 ตัว
  (`dense`, `sparse_bge`, `sparse_bm25`) ค้นได้ 4 โหมด (`SEARCH_MODE`): `dense` · `dense+sparse_bge` (**ค่าเริ่มต้น**) · `dense+sparse_bm25` · `all`
  (โหมดที่มี sparse รวมผลด้วย RRF ของ Qdrant; `sparse_bm25` index ไว้แต่ไม่ใช่ค่าเริ่มต้น) · `upsert_chunks` ลบ point เก่าของไฟล์ก่อนเขียนเสมอ
- `app/chunking.py` — `make_chunks`: chunk ตามขอบประโยคขนาด `CHUNK_TARGET_CHARS` (ค่าเริ่มต้น 1000) ซ้อน 1 ประโยค ไม่ข้าม unit
  เก็บ offset ประโยคใน `Unit.text` ไว้ cite · context line ของ unit ถูกแปะหน้า chunk ตอน embed เท่านั้น
- `app/embedding.py` — BGE-M3 dense + sparse (lexical weights) ผ่าน `transformers` (bf16 บน GPU ปรับด้วย `EMBEDDING_DTYPE`; `release()` คืน VRAM)
  **ไม่ fallback เงียบ**: โหลดโมเดลไม่ได้ = raise `EmbeddingUnavailable` hash vector ใช้ได้เฉพาะเมื่อตั้ง `EMBEDDING_TEST_MODE=1`
  (test ตั้งให้เอง) — ต้องมี `torch` (ดูหัวข้อติดตั้งด้านบน) ทำ OCR ให้เสร็จก่อนแล้วค่อย embed
  อย่าให้ Typhoon กับ BGE-M3 อยู่บน GPU พร้อมกัน

ทดลองขนาด chunk x โหมดค้นบนชุด dev: `python scripts/retrieval_dev_eval.py` (ต้องมี Qdrant + BGE-M3; ผลที่ `eval/results/retrieval_dev_eval.md`)

เมทริกซ์สิทธิ์ที่คนเขียนอยู่ที่ `eval/datasets/access_control.jsonl` (box / file / principal / case อย่างละบรรทัด)
เพิ่มกรณีทดสอบได้โดยเพิ่มบรรทัด `case` — `pytest tests/test_access.py` จะเทียบให้เอง

### ดูของจริงทำงาน

```bash
python scripts/acl_demo.py                    # mock_corpus.pdf + Qdrant ในกระบวนการ
python scripts/acl_demo.py --server           # ใช้ Qdrant จาก docker compose
python scripts/acl_demo.py --file eval/datasets/ONEREPORTPTTT.PDF --pages 1-20
```

ingest ไฟล์จริงแล้วถามคำถามเดียวกันด้วยสอง persona ที่ต่างกันแค่แผนก — ฝั่งที่ไม่มีสิทธิ์ต้องตอบ
เหมือนหาไม่เจอทุกประการ **อันดับผลค้นยังไม่มีความหมาย** เพราะ `app/embedding.py` ยังไม่มีโมเดลจริง
(สคริปต์พิมพ์เตือนไว้)

## รัน API + ตั้ง demo + วัดผลผ่าน API

> **DATA_DIR ห้ามอยู่ในโฟลเดอร์ที่มีโปรแกรม sync ทำงานอยู่** (OneDrive, Dropbox, Google Drive ฯลฯ) — SQLite (`rag.db`, โหมด WAL)
> กับไฟล์ต้นฉบับจะเสียหายหรือถูกล็อกเมื่อโปรแกรม sync แตะไฟล์ระหว่างเขียน · ค่าเริ่มต้น `data/` ในโฟลเดอร์ repo ใช้ได้ต่อเมื่อปิด sync ของโฟลเดอร์นั้นแล้ว

```powershell
docker compose up -d qdrant
$env:SERVICE_KEY = "<ค่าลับที่ Gateway ใช้>"           # หรือใส่ใน .env
$env:SYSTEM_ADMIN_USERS = "demo-sysadmin"              # ผู้สร้างกล่องใน demo
$env:PYTHONUTF8 = 1
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 *> api.log   # log ไว้ให้ eval แยกเวลา embed/search
```

ตอนเริ่ม API: log อุปกรณ์ที่ใช้ · โหลด query embedder (BGE-M3 อีกตัวบน **CPU fp32** แยกจาก embedder ของ worker บน GPU) ·
worker thread เริ่ม (งานที่ค้างจาก process ก่อนกลับเข้าคิว) — ดูสถานะที่ `GET /v1/health/ready` (ต้องส่ง `X-Service-Key`)

```bash
python scripts/bootstrap_demo.py --base-url http://127.0.0.1:8000   # key จาก env SERVICE_KEY / .env
python scripts/bootstrap_demo.py --reset                            # ลบ DATA_DIR (ยกเว้น ocr_cache) + collection (ถามยืนยัน, หยุด API ก่อน)
python scripts/retrieval_api_eval.py --api-log api.log               # ชุด dev ผ่าน /v1/retrieve -> eval/results/retrieval_api_eval.*
```

- `bootstrap_demo.py` สร้างกล่อง `annual-reports` (public: KBANK, CPALL, PTT) และ `finance-private` (private, แผนก finance ระดับ ≥ 2:
  สำเนา PTT) + สมาชิกชื่อสมมติ (`*.demo`) ผ่าน admin API แล้ว ingest ผ่าน `POST /v1/ingest` และรอ `/v1/jobs` จนจบ พิมพ์สรุปต่อไฟล์
  และรายชื่อ persona พร้อมค่า header `X-User-*` สำหรับเดโม ACL
- **ไม่ OCR ใหม่**: ก่อนส่งไฟล์ ตรวจว่าทุกหน้าที่ต้อง OCR มีในแคช OCR แล้ว (ขาด = หยุด เว้นแต่ใส่ `--allow-ocr`) และหลังจบรายงานจำนวน OCR ใหม่
  (ต้องเป็น 0) — ต้องรันบนเครื่องเดียวกับ API
- รันซ้ำได้: กล่องที่มีอยู่แล้วตรวจผ่าน `/v1/sources` · ไฟล์ซ้ำได้ `200 duplicate` ไม่ index ใหม่
- `--reset` ใช้เมื่อเปลี่ยน payload / schema ของ index (เช่น point เก่าที่ยังไม่มี `flags` ของ unit) — หยุด API ก่อน แล้วเริ่มใหม่และรัน bootstrap อีกครั้ง

### ลบไฟล์

`DELETE /v1/files/{file_id}` (owner/admin ของกล่องที่เห็นไฟล์นั้น — นอกนั้น 404 เหมือนไม่มีไฟล์): ไฟล์หายจาก `/v1/sources` และ
`/v1/retrieve` ทันที แล้วลบ point ใน Qdrant, ต้นฉบับ, ผล extract และแคช OCR ของเนื้อไฟล์นั้น (เฉพาะเมื่อไม่มีไฟล์อื่นในกล่องใดใช้
เนื้อไฟล์เดียวกัน) · งาน ingest ที่ยัง queued ถูกยกเลิก · งานที่กำลังทำ = `409 conflict` (ลบใหม่หลังงานจบ) · พังกลางทาง /
Qdrant ล่ม = ไฟล์ค้างสถานะ `deleting` (ซ่อนอยู่) แล้วถูกลบต่อจนครบตอนเริ่ม API / ตอน worker ว่าง
ฐาน SQLite schema v1 ถูก migrate เป็น v2 อัตโนมัติตอนเริ่ม API ครั้งแรกหลังอัปเดต (สำรองเป็น `rag.db.bak-v1` ก่อน)

## สำรองและกู้ข้อมูล

```bash
python scripts/backup.py                                   # -> backups/rag-backup-<UTC>.zip (อ่านอย่างเดียว)
python scripts/restore.py backups/rag-backup-....zip --data-dir D:/rag-restore \
    --collections restore_public restore_private           # กู้ลงที่ใหม่ แล้วตรวจจำนวนแถว/point/ไฟล์กับ manifest
python scripts/restore.py <zip> --data-dir data --force     # ทับของเดิม (หยุด API ก่อน)
```

- **ต้องสำรอง `data/` กับ Qdrant พร้อมกันเสมอ** — point ใน Qdrant อ้าง `file_id` ใน `rag.db` สำรองแยกเวลากันได้ index ที่ไม่ตรงกับทะเบียนไฟล์
  `backup.py` จึงทำทั้งสองอย่างในครั้งเดียว และไม่ยอมสำรองขณะมีงาน ingest กำลังทำหรือไฟล์ค้าง `deleting` (`--allow-busy` = ข้าม)
- SQLite ใช้ online backup API (ไม่ copy `rag.db` ตรงๆ — ฐานเป็น WAL ข้อมูลล่าสุดอยู่ใน `rag.db-wal`) · Qdrant ใช้ snapshot API
  (สร้างบน server -> ดาวน์โหลด -> ลบ snapshot บน server) · zip มี `manifest.json` (เวลา, จำนวนแถว/ไฟล์/point, sha256 ทุกไฟล์)
  — `restore.py` ตรวจ sha256 ก่อนเขียนอะไร และไม่ทับปลายทางที่มีข้อมูลเว้นแต่ `--force`
- **Docker บน Windows: named volume (ข้อมูล Qdrant และ `rag_data` ของ API) อยู่ในดิสก์ของ WSL2 ไม่ใช่โฟลเดอร์บน Windows** —
  copy ไฟล์ของ volume ตรงๆ ไม่ได้ (และ copy ไฟล์ของ Qdrant ขณะรันได้ข้อมูลเสีย) ให้ใช้สคริปต์นี้ (API ผ่าน HTTP) เท่านั้น
  · รัน API ใน container: `docker compose exec api python scripts/backup.py --out /data/backup.zip` แล้ว `docker compose cp api:/data/backup.zip .`

## Docker (API + Qdrant)

```powershell
$env:SERVICE_KEY = "<ค่าลับ>"; $env:SYSTEM_ADMIN_USERS = "demo-sysadmin"
$env:HF_CACHE_DIR = "C:/Users/<ชื่อ>/.cache/huggingface"    # ใช้โมเดลที่โหลดไว้แล้ว (ไม่ตั้ง = volume hf_cache โหลดใหม่ครั้งแรก)
docker compose up -d --build                                 # ภาพ rag-api:cpu (TORCH_VARIANT=cpu)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build   # GPU: ตั้ง $env:TORCH_VARIANT = "cu128" ก่อน
```

- ภาพมี Tesseract (ไทย) + LibreOffice + ฟอนต์ไทย · **ไม่มีโมเดลในภาพ** (mount HF cache ที่ `/hf`) · ค่าลับผ่าน env ตอนรันเท่านั้น
  · ขนาดภาพ (วัด 14 พ.ย. 2026): `rag-api:cpu` 2.52 GB บนดิสก์ (≈0.59 GB บีบอัด) · `rag-api:cu128` 12.5 GB (≈4.4 GB บีบอัด)
- ข้อมูลของ API อยู่ใน named volume `rag_data` (`/data`: `rag.db`, `files/`, `documents/`, `ocr_cache/`)
- **ภาพ cpu + แคช OCR เดิม (Typhoon): ต้องตั้ง `OCR_BACKEND=typhoon-local`** — key ของแคชมีชื่อ backend อยู่ด้วย
  ค่าเริ่มต้น `auto` บนภาพ cpu = `tesseract-local` -> ไม่ตรงแคช Typhoon เลยสักหน้า -> OCR ใหม่ทั้งหมดด้วย Tesseract (ผลคนละชุด)
  - หน้าที่อยู่ในแคช: ใช้ผลเดิม (ไม่โหลดโมเดล) · `/v1/health/ready` = degraded (`ocr: down` — ไม่มี GPU) เป็นเรื่องปกติของโหมดนี้
  - หน้าที่ **ไม่อยู่ในแคช**: ไม่รัน Typhoon บน CPU (ตรวจ CUDA ก่อนโหลดโมเดลเสมอ) — หน้านั้น**คงสถานะ `needs_ocr`** (ยังไม่ได้ OCR
    ไม่ใช่ OCR แล้วพัง) เป็น gap พร้อมเหตุผล `OCR ใช้ไม่ได้: ไม่พบ GPU CUDA — Typhoon OCR รันบน CPU ไม่ได้ในระบบนี้ (ไม่ถอยไปใช้ CPU)`
    หน้าอื่นทำต่อตามปกติ -> งาน `done_with_gaps` · ไม่มีหน้าที่ใช้ได้เลย -> งาน `failed` รหัส `unavailable` (ชั่วคราว)
  - **ส่งไฟล์เดิมซ้ำเมื่อ OCR พร้อมแล้ว** (เช่น ย้ายไปภาพ cu128 / เครื่องที่มี GPU) -> เข้าคิวใหม่ (`200`, `duplicate: true`,
    `status: queued`, job_id ใหม่, file_id/ACL/ชื่อเดิม) — OCR เฉพาะหน้าที่ขาด (หน้าที่ OCR แล้วได้จากแคช) และเขียน index ทับของเดิม
    (point ไม่ซ้ำ) · ใช้กับงาน `failed`/`unavailable` และ `done_with_gaps` ที่มี gap `needs_ocr` เท่านั้น — gap `failed`
    (OCR แล้วไม่ผ่านด่านตรวจ เช่น `latin_in_thai`) ส่งซ้ำได้งานเดิม (ผลเดิม)
  - ตรวจล่วงหน้าว่าแคชครบไหม: `scripts/bootstrap_demo.py` ตรวจทุกหน้าก่อนส่ง (ขาด = หยุด) สำหรับไฟล์ demo
- ใส่แคช OCR เดิมเข้า volume: `docker compose create api` -> `docker compose cp data/ocr_cache/. api:/data/ocr_cache/` -> `docker compose up -d`
- ทดสอบ stack ที่สองโดยไม่แตะของจริง: `QDRANT_CONTAINER=ragtest_qdrant QDRANT_PORT=16333 QDRANT_GRPC_PORT=16334 API_PORT=18000 docker compose -p ragtest up -d`

## ส่วนเสริมของ /v1/retrieve (ทดลอง ปิดเป็นค่าเริ่มต้น)

| env | ค่า | ทำอะไร |
|---|---|---|
| `RERANKER` | `off` \| `bge-v2-m3` | cross-encoder BAAI/bge-reranker-v2-m3 จัด 20 อันดับแรกจาก RRF ใหม่ แล้วคืน top_k (`RERANKER_DEVICE=cpu\|cuda`) |
| `QUERY_EXPANSION` | `0` \| `1` | ขยายคำถาม (คำย่อ, พ.ศ.↔ค.ศ.) ด้วย `app/query_expansion.py` ก่อน embed |
| `SALIENCE_RERANK` | `0` \| `1` | จัด 20 อันดับแรกใหม่ด้วย `app/rerank.py::rerank_hits` (เปิดพร้อม `RERANKER` ไม่ได้) |

- `score` ใน response ยังเป็น dense cosine ทุกโหมด (ไม่เปลี่ยนความหมาย) · rank = ลำดับหลัง rerank · คะแนน reranker อยู่ใน log (`rerank_top`)
- `QUERY_EXPANSION` / `SALIENCE_RERANK` **ห้ามวัดบนชุด dev** (คำย่อ/น้ำหนักตั้งจากข้อที่พลาดในชุดนั้น) — `retrieval_api_eval.py` ไม่ยอมเขียนผล
  ใช้ `eval/datasets/questions.jsonl` แทน (schema: `eval/datasets/questions.schema.json`, ตรวจ: `python -m eval.questions`):
  `python scripts/retrieval_api_eval.py --questions eval/datasets/questions.jsonl --name questions_eval --api-log api.log`
- ผลวัด reranker บนชุด dev: `eval/results/reranker_eval.md`

## วัดผล Thai normalizer

```bash
python scripts/normalize_report.py                       # 3 PDF ใน eval/datasets/ + mock_corpus.pdf พิมพ์ออกจอ
python scripts/normalize_report.py --samples 20 --out report.txt
```

แสดงจำนวนที่แก้รายกฎ, signals, 10 หน้าที่โดนแก้มากที่สุด และตัวอย่างก่อน/หลังรายกฎ (พร้อมไฟล์+หน้า)
เพื่อให้คนตรวจ คู่ทดสอบอยู่ที่ `eval/datasets/thai_normalization.jsonl`

### เติม truth (ให้คนตรวจแทนการไล่กฎ)

1. คู่ที่ไม่ใช่ synthetic มีช่อง `"truth": null` เปิด PDF ตาม `source` (ไฟล์ + หน้า) หาข้อความช่วงเดียวกับ `input`
2. พิมพ์ข้อความช่วงนั้นตามที่ตาเห็นลงใน `truth` ไม่ต้องสนใจกฎของ normalizer เว้นวรรคหรือขึ้นบรรทัดไม่ตรงกับ `input` ได้
3. ถ้าในภาพก็อ่านไม่ออก ให้ปล่อย `truth` เป็น `null` แล้วรัน `python scripts/check_truth.py` ดูผลรายคู่
   (exact / evidence / needs_repair / fail) และ `pytest` จะเทียบให้เอง

## ด่านตรวจคุณภาพ (needs_ocr) และการติดป้าย

ผลตัดสินอยู่ที่ `Unit.status` + `quality.flags` (กฎอยู่ใน `app/extraction/quality.py`) ดูสรุปรายไฟล์ด้วย
`python scripts/gate_report.py` (เหตุผลที่ส่ง OCR, หน้าที่โดนกฎระดับไฟล์, การกระจาย dictionary_ratio)

ติดป้ายหน้าเพื่อตรวจเกณฑ์ (ไม่ให้เกณฑ์วัดตัวเอง):
1. `python scripts/make_label_set.py` สร้าง `eval/datasets/ocr_pages.jsonl` (36 หน้า label เป็น null) และ `eval/datasets/files/label_pages.html`
2. เปิด HTML ดูภาพหน้า PDF เทียบข้อความที่ดึงได้ (ผลของด่านตรวจซ่อนอยู่ อย่าเปิดดูก่อนติดป้าย) แล้วแก้ `label` ใน jsonl:
   `"ok"` = ข้อความตรงกับ PDF พอใช้ค้นและอ้างอิงได้ (ช่องว่าง/ลำดับไม่ต้องสน) · `"broken"` = เจอคำที่ผิดจาก PDF ทั้งที่ต้นฉบับไม่ได้ผิด · `"skip"` = ไม่แน่ใจ
3. รัน `python scripts/gate_report.py` จะได้ตาราง ป้าย x ผลด่านตรวจ และรายการหน้าที่ไม่ตรงกัน (สคริปต์จะไม่เขียนทับ jsonl ที่มีข้อมูลแล้ว)
