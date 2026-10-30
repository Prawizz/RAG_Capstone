# RAG API Service — คู่มือสำหรับ Claude Code

Capstone ร่วมกับบริษัท: **RAG API Service** ที่ต่อเข้ากับ Enterprise AI Gateway + Webchat ที่องค์กรมีอยู่แล้ว
เราไม่ทำแอปใหม่และไม่ generate คำตอบ — Gateway ถือ identity / budget / queue และเป็นคน generate
API ของเรา: ingest, retrieve, list sources, job status, health

## อ่านก่อนเริ่มงาน

| เอกสาร | มีอะไร |
|---|---|
| `docs/PLAN.md` | ลำดับงานและเกณฑ์ผ่านของแต่ละเวอร์ชัน |
| `docs/FORMATS.md` | แนวทางรับมือไฟล์แต่ละประเภท |
| `docs/DATA_FINDINGS.md` | ปัญหาภาษาไทยที่เจอจริงในไฟล์ทดสอบ พร้อมตัวอย่าง |

`docs/` ถูก gitignore ไว้ — มีบนเครื่องผู้ที่ทำงานหลัก แต่ไม่อยู่ใน repo ถ้าไม่มีโฟลเดอร์นี้ ให้ยึดกฎใน CLAUDE.md เป็นหลัก task spec จะส่งมาทาง prompt

**ช่วงนี้คือ ingestion-first** — ทำ ingestion ให้เกือบสมบูรณ์ก่อน แล้วค่อยไป retrieval / ACL / integration

## โครง repo

```
app/extraction/   extractor ต่อประเภทไฟล์ + pipeline
app/thai/         normalizer
app/models.py     contract กลาง (Document / Unit / Block / สถานะ)
app/api/          FastAPI routes + schemas (contract กับ Gateway)
app/access.py     ACL
eval/             harness + datasets
tests/            unit tests
docs/             เอกสารออกแบบ
```

## คำสั่ง

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

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
9. **สิทธิ์** (เมื่อถึงเฟส ACL) — Public/Private เป็นคุณสมบัติของ box เปลี่ยนไม่ได้, สิทธิ์ระดับไฟล์ทำให้แคบลงจาก box ได้อย่างเดียว, กรณีไม่มีสิทธิ์ต้องตอบเหมือนกรณีหาไม่เจอทุกประการ
10. **Demo console เรียก RAG ผ่าน API สาธารณะเท่านั้น** ห้าม import module ข้างใน
11. **ห้าม commit ไฟล์ทดสอบขนาดใหญ่เพิ่ม** — PDF 3 ไฟล์ใน `eval/datasets/` (20–45 MB) ถูก commit ไปแล้ว ให้คงไว้ตามเดิม และห้ามแก้ git history เอง

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
