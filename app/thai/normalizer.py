"""Normalizer กลาง (ตัวเดียวของระบบ)

นี่คือจุดเดียวที่ข้อความทุกส่วนต้องผ่าน ทั้งตอน ingest (``Unit.text`` / ``Block.text``)
และตอนรับคำถามตอน retrieve ห้ามเขียน normalizer ตัวที่สอง (CLAUDE.md ข้อ 1)

ตอนนี้เป็น identity (คืนข้อความเดิม) เพื่อวางตำแหน่งให้ pipeline เรียกใช้ได้ถูกที่ตั้งแต่แรก
T03 จะมาแทนที่ด้วยกฎจริงและเปลี่ยน ``NORMALIZER_VERSION``
"""

NORMALIZER_VERSION = "0-identity"


def normalize(text: str) -> str:
    return text
