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
