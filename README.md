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
```

สรุปที่พิมพ์: ชื่อไฟล์, ชนิด, จำนวนหน้า, จำนวน unit ต่อสถานะ, เวลารวม, เวลาเฉลี่ยและสูงสุดต่อหน้า, error (ถ้ามี)
exit code เป็น 1 เมื่อทั้งไฟล์ล้มเหลว (`Document.error`)

## Test

```bash
pytest            # test ปกติ (เร็ว)
pytest -m slow    # test ช้า: ใช้ PDF จริงใน eval/datasets/ และ mock_corpus.pdf
```
