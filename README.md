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
  (`dense`, `sparse_bge`, `sparse_bm25`) ค้นได้ 4 โหมด (`SEARCH_MODE`): `dense` · `dense+sparse_bge` · `dense+sparse_bm25` · `all`
  (โหมดที่มี sparse รวมผลด้วย RRF ของ Qdrant) · `upsert_chunks` ลบ point เก่าของไฟล์ก่อนเขียนเสมอ
- `app/chunking.py` — `make_chunks`: chunk ตามขอบประโยคขนาด `CHUNK_TARGET_CHARS` (ค่าเริ่มต้น 1000) ซ้อน 1 ประโยค ไม่ข้าม unit
  เก็บ offset ประโยคใน `Unit.text` ไว้ cite · context line ของ unit ถูกแปะหน้า chunk ตอน embed เท่านั้น
- `app/embedding.py` — BGE-M3 dense + sparse (lexical weights) ผ่าน `transformers` (fp16 บน GPU; `release()` คืน VRAM)
  **ไม่ fallback เงียบ**: โหลดโมเดลไม่ได้ = raise `EmbeddingUnavailable` hash vector ใช้ได้เฉพาะเมื่อตั้ง `EMBEDDING_TEST_MODE=1`
  (test ตั้งให้เอง) — ต้องมี `torch` + `transformers` (อยู่ใน `requirements-ocr.txt`) ทำ OCR ให้เสร็จก่อนแล้วค่อย embed
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
