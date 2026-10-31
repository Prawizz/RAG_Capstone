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
pytest            # test ปกติ (เร็ว)
pytest -m slow    # test ช้า: ใช้ PDF จริงใน eval/datasets/ และ mock_corpus.pdf
```

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
