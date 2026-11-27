"""mock Enterprise AI Gateway — ตัวแทน Gateway จริงสำหรับ demo/ทดสอบ (ทำตาม docs/API_CONTRACT.md)

หน้าที่ของ Gateway ตาม contract ที่ mock นี้จำลอง: ยืนยันตัวตนผู้ใช้ · ส่ง X-User-* ให้ RAG · งบประมาณราย user ·
เลือก LLM ภายใน/ภายนอกจาก ``visibility`` · เขียนคำตอบและข้อความปฏิเสธเอง (RAG ไม่ generate)

**กฎเหล็กข้อ 10:** แพ็กเกจนี้เรียก RAG ผ่าน HTTP สาธารณะเท่านั้น ห้าม ``import app`` (มี test ตรวจ)
"""
