# mock Enterprise AI Gateway (`gateway_mock/`)

ตัวแทน Gateway จริงสำหรับ demo และทดสอบ — ทำตามหน้าที่ฝั่ง Gateway ใน `docs/API_CONTRACT.md` (หัวข้อ 1, 4.3, 6)
**เรียก RAG ผ่าน HTTP สาธารณะเท่านั้น ห้าม `import app`** (กฎเหล็กข้อ 10 · มี test ตรวจ)

## รัน

```bash
# 1) RAG API + demo data พร้อมแล้ว (ดู README "รัน API + ตั้ง demo")
export SERVICE_KEY=<ค่าเดียวกับที่ RAG API ใช้>
uvicorn gateway_mock.main:create_app --factory --port 9000
curl -s localhost:9000/chat -H 'X-Demo-User: wanida.demo' -H 'Content-Type: application/json' \
     -d '{"question":"เงินปันผลต่อหุ้นเท่าไร","box_id":"annual-reports"}'
```

env: `RAG_BASE_URL` (ค่าเริ่มต้น `http://127.0.0.1:8000`) · `SERVICE_KEY` · `GATEWAY_BUDGET_PER_USER` (ค่าเริ่มต้น 50, 0 = ไม่จำกัด)

## Endpoint (ฝั่งผู้ใช้ปลายทาง — ไม่ใช่ contract ของ RAG)

| Method | Path | ทำอะไร |
|---|---|---|
| POST | `/chat` | `{question, box_id?, top_k?}` → `answered` (มี `answer`, `citations`, `llm`) / `choose_source` (มีรายการกล่อง) / `not_found` |
| GET | `/sources` | กล่องที่ผู้ใช้เป็นสมาชิก (ส่งต่อจาก RAG) |
| GET | `/me` | ตัวตนที่ Gateway ถือ + โควตาที่ใช้ไป |
| GET | `/health` | ไม่ต้อง auth |

ผู้ใช้ยืนยันตัวตนด้วย header `X-Demo-User` (ชื่อใน `gateway_mock/users.py` ตรงกับ persona ของ `scripts/bootstrap_demo.py`) — ของจริงคือ SSO
ไม่มี/ไม่รู้จัก → 401 · `X-User-*` ที่ผู้ใช้ส่งมาถูกเมินเสมอ (แผนกและระดับมาจากไดเรกทอรีของ Gateway)

## กติกาที่ mock บังคับ

| เรื่อง | พฤติกรรม |
|---|---|
| เลือก LLM | `visibility = public` → LLM ภายนอก · **อย่างอื่นทุกค่า (private / ว่าง / ไม่รู้จัก) → LLM ภายใน** (fail closed) |
| `ok` | generate จาก `results` + แนบ citation · `ocr_incomplete` ขึ้นหมายเหตุในคำตอบ |
| `source_required` | `choose_source` + รายการกล่องของผู้ใช้ (ไม่เรียก LLM) |
| `no_relevant_context` / status ที่ไม่รู้จัก | `not_found` ไม่เรียก LLM · ข้อความ/ฟิลด์เหมือนกันทุกกรณี **ไม่แยก "ไม่ใช่สมาชิก" ออกจาก "กล่องว่าง"** และไม่คืน `visibility`/คะแนน (กฎข้อ 9) |
| RAG ล่ม / timeout / 5xx | 503 `unavailable` — **ไม่แปลงเป็น "ไม่พบข้อมูล"** · คืนโควตาให้ผู้ใช้ |
| RAG ปฏิเสธ service key (401) | 502 `upstream_misconfigured` (ความผิดของค่าตั้ง Gateway) |
| งบประมาณ | นับต่อ user ในหน่วยความจำ · เกิน → 429 ก่อนเรียก RAG |
| trace | ส่ง `X-Trace-Id` ต่อให้ RAG และคืน `trace_id` ทุก response |

## ต่อ LLM จริง

implement `LLMBackend` (`name`, `location`, `generate(question, passages)`) แล้วส่งเข้า `LLMRouter(internal=..., external=...)` ใน `create_app(router=...)`
`LLMRouter` จะไม่ยอมให้สลับตำแหน่ง (backend ที่ `location` ไม่ตรงช่อง → `ValueError`)
`ExtractiveLLM` ที่ให้มายกข้อความจากหลักฐาน 2 อันดับแรก **ไม่ได้ตัดสินว่าหลักฐานตอบคำถามได้จริงไหม** — contract ให้ LLM ฝั่ง Gateway ทำ (score ของเราแยกไม่ได้)

## Test

```bash
pytest tests/test_gateway_mock.py     # ตอบจากตัวอย่างใน contract/examples (ไม่ได้สร้างจากโค้ดเรา)
pytest tests/test_gateway_e2e.py      # RAG API ตัวจริง (ACL จริง, Qdrant ในกระบวนการ) + persona ของ demo
```

## ข้อจำกัด

- ยังไม่ได้ทดลองกับ RAG ที่รันจริงบน Qdrant + BGE-M3 (test ใช้ hash embedding) — ลองด้วยคำสั่งข้างบนหลัง `bootstrap_demo.py`
- ไม่มีคิว, ไม่มี streaming, งบเป็นตัวนับในหน่วยความจำ (รีสตาร์ตแล้วหาย) · `X-Demo-User` ไม่ใช่การยืนยันตัวตนจริง
- demo console (งานแยก) ควรเรียก mock นี้หรือ RAG API โดยตรงก็ได้ แต่ห้าม import `app`
