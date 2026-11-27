"""LLM จำลอง + ตัวเลือก LLM ตาม visibility (นโยบายหลักของ Gateway)

- กล่อง ``private`` -> LLM ภายในองค์กรเท่านั้น (ห้ามส่งข้อความออกนอกองค์กร)
- กล่อง ``public`` -> LLM ภายนอกได้
- ``visibility`` ที่ไม่ใช่ \"public\" เป๊ะๆ (ว่าง/ค่าที่ไม่รู้จัก) -> ภายใน (fail closed: ไม่แน่ใจ = ถือว่า private)

ของจริงต่อ LLM ผ่าน backend ที่ implement ``LLMBackend``; ตัวที่ให้มาเป็น extractive (ยกข้อความจากหลักฐาน ไม่แต่งเพิ่ม)
เพื่อให้ demo/test ได้ผลซ้ำได้โดยไม่ต้องมี LLM จริง — ไม่ได้ประเมินว่าหลักฐาน \"ตอบคำถามได้จริงไหม\" (งานของ LLM จริง)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

Location = Literal["internal", "external"]


@dataclass(frozen=True)
class Passage:
    n: int  # เลขอ้างอิง [n] ในคำตอบ
    text: str
    label: str
    filename: str
    flags: tuple[str, ...] = field(default_factory=tuple)


class LLMBackend(Protocol):
    name: str
    location: Location

    def generate(self, question: str, passages: list[Passage]) -> str: ...


class ExtractiveLLM:
    """ตอบโดยยกข้อความจากหลักฐาน 2 อันดับแรกพร้อมเลขอ้างอิง — ไม่แต่งข้อความเอง"""

    def __init__(self, name: str, location: Location, max_passages: int = 2) -> None:
        self.name = name
        self.location = location
        self._max = max_passages

    def generate(self, question: str, passages: list[Passage]) -> str:
        used = passages[: self._max]
        body = "\n\n".join(f"{p.text.strip()} [{p.n}]" for p in used)
        notes = [f"หมายเหตุ: [{p.n}] มาจาก OCR ที่อาจอ่านไม่ครบ" for p in used if "ocr_incomplete" in p.flags]
        head = f"(คำตอบจำลองจาก {self.name} — ยกข้อความจากหลักฐานโดยตรง)"
        return "\n\n".join([head, body, *notes])


class LLMRouter:
    def __init__(self, internal: LLMBackend, external: LLMBackend) -> None:
        if internal.location != "internal":
            raise ValueError(f"backend ภายในต้องมี location=internal (ได้ {internal.location})")
        if external.location != "external":
            raise ValueError(f"backend ภายนอกต้องมี location=external (ได้ {external.location})")
        self._internal = internal
        self._external = external

    def pick(self, visibility: Optional[str]) -> LLMBackend:
        return self._external if visibility == "public" else self._internal
