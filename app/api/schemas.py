"""[M7] Pydantic schemas for API request / response contract (Gateway-facing)

ปฏิบัติตาม Hard Rule 9 ใน PLAN.md:
- ห้ามส่งออก `denied_layer` หรือ `dropped_by_recheck` ออกทาง API โดยเด็ดขาด
  เพื่อป้องกันไม่ให้ผู้เรียกสามารถแยกแยะได้ว่า "ไม่มีสิทธิ์" หรือ "หาไม่พบ" (Fail Closed)
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field

from app.access import Principal


class PrincipalRequest(BaseModel):
    """ข้อมูลผู้สอบถามจาก Gateway"""

    user_id: str = Field(min_length=1, description="รหัสผู้ใช้งาน")
    departments: list[str] = Field(default_factory=list, description="แผนกที่สังกัด")
    security_level: int = Field(default=0, ge=0, description="ระดับชั้นความลับของผู้ใช้")
    box_ids: list[str] = Field(default_factory=list, description="กล่องที่ผู้ใช้เป็นสมาชิก")

    def to_domain(self) -> Principal:
        """แปลงเป็นโมเดล Domain Principal สำหรับใช้ใน ACL Layer"""
        return Principal(
            user_id=self.user_id,
            departments=frozenset(self.departments),
            security_level=self.security_level,
            box_ids=frozenset(self.box_ids),
        )


class RetrieveRequest(BaseModel):
    """คำขอสืบค้นข้อมูลสำหรับ RAG"""

    query: str = Field(min_length=1, description="คำถามของผู้ใช้")
    principal: PrincipalRequest = Field(description="ข้อมูลสิทธิ์ของผู้ถาม")
    box_id: str = Field(min_length=1, description="รหัสกล่องเอกสารที่ต้องการค้น")
    limit: int = Field(default=5, ge=1, le=50, description="จำนวน chunks สูงสุดที่ต้องการ")
    search_mode: Optional[str] = Field(
        default=None,
        description="โหมดค้น: 'dense' | 'dense+sparse_bge' | 'dense+sparse_bm25' | 'all'",
    )
    enable_rerank: bool = Field(default=True, description="เปิดใช้งาน Lexical-Semantic Reranking")
    enable_abstention: bool = Field(default=True, description="เปิดใช้งาน Abstention Gate")


class CitationItemSchema(BaseModel):
    """โครงสร้างข้อมูลอ้างอิงระดับประโยคสำหรับ Gateway / UI"""

    sentence_id: str = Field(description="รหัสแท็ก เช่น S1, S2")
    doc_id: str = Field(description="รหัสไฟล์เอกสารต้นทาง")
    label: str = Field(default="", description="ป้ายระบุตำแหน่ง เช่น หน้า 7")
    char_start: int = Field(description="ตำแหน่งออฟเซ็ตเริ่มต้นในข้อความเดิม")
    char_end: int = Field(description="ตำแหน่งออฟเซ็ตสิ้นสุดในข้อความเดิม")
    snippet: str = Field(description="ข้อความประโยคอ้างอิง")
    position: dict[str, Any] = Field(default_factory=dict, description="พิกัดหน้า/กล่องข้อความ")
    score: float = Field(default=0.0, description="คะแนนความเกี่ยวข้อง")


class HitItemSchema(BaseModel):
    """รายการ Chunk ที่ผ่านการคัดเลือกและผ่านสิทธิ์"""

    file_id: str
    filename: str
    unit_index: int
    label: str
    text: str
    score: float
    position: dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    """ผลการสืบค้นข้อมูลที่พร้อมส่งให้ LLM Gateway"""

    query: str
    hits: list[HitItemSchema] = Field(default_factory=list)
    formatted_context: str = Field(
        default="",
        description="ข้อความบริบทที่ประกอบแท็ก [S1], [S2] เรียบร้อยแล้วสำหรับส่งเป็น LLM Context",
    )
    citations: dict[str, CitationItemSchema] = Field(
        default_factory=dict,
        description="แคตตาล็อกอ้างอิงระดับประโยคสำหรับ Frontend นำไปทำ Citation Highlight",
    )
    is_abstain: bool = Field(
        default=False,
        description="ระบบตัดสินใจปฏิเสธการตอบ (True เมื่อหลักฐานไม่เพียงพอ หรือไม่มีสิทธิ์เข้าถึง)",
    )
    abstain_reason: Optional[str] = Field(
        default=None,
        description="เหตุผลการปฏิเสธ: 'NO_EVIDENCE_FOUND' | 'INSUFFICIENT_EVIDENCE' | 'EMPTY_QUERY'",
    )
    fallback_message: Optional[str] = Field(
        default=None,
        description="ข้อความตอบกลับมาตรฐานเมื่อระบบปฏิเสธการตอบ",
    )
    has_conflict: bool = Field(
        default=False,
        description="พบข้อมูลเชิงตัวเลขที่ขัดแย้งกันในเอกสารหรือไม่",
    )
    conflict_guidance: Optional[str] = Field(
        default=None,
        description="คำแนะนำสำหรับ LLM ในการตอบเพื่อคลายความขัดแย้งของตัวเลข",
    )
    expanded_terms: list[str] = Field(
        default_factory=list,
        description="คำค้นที่ระบบขยายเพิ่มเติม (เช่น ปี พ.ศ./ค.ศ. หรือคำย่อ)",
    )
