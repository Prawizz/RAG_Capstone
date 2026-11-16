"""ตรรกะของ Gateway ต่อหนึ่งคำถาม: งบ -> ค้นหลักฐานผ่าน RAG -> ตัดสินตาม status -> เลือก LLM -> คำตอบ + citation

สิ่งที่ตั้งใจไม่ทำ (ตาม contract): ไม่แยกกรณี \"ไม่ใช่สมาชิก\" ออกจาก \"กล่องว่าง\" — ข้อความและฟิลด์ต้องเหมือนกันทุกประการ
ไม่งั้นผู้ใช้ไล่เดา box_id ที่ตัวเองไม่มีสิทธิ์ได้ (กฎเหล็กข้อ 9)
"""

from __future__ import annotations

from typing import Optional

from gateway_mock.budget import Budget
from gateway_mock.llm import LLMRouter, Passage
from gateway_mock.rag_client import RagClient, RagError
from gateway_mock.schemas import ChatResponse, Citation, LlmInfo, SourceInfo
from gateway_mock.users import User

MSG_NOT_FOUND = "ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามนี้ในกล่องที่เลือก"
MSG_CHOOSE_SOURCE = "กรุณาเลือกกล่องเอกสารที่ต้องการค้นก่อน"


class ChatService:
    def __init__(self, rag: RagClient, router: LLMRouter, budget: Budget) -> None:
        self._rag = rag
        self._router = router
        self._budget = budget

    def sources(self, user: User, trace_id: Optional[str] = None) -> list[SourceInfo]:
        body = self._rag.sources(user, trace_id)
        return [SourceInfo(box_id=s["box_id"], name=s.get("name"), visibility=s.get("visibility"))
                for s in body.get("sources", [])]

    def chat(self, user: User, question: str, box_id: Optional[str], top_k: int,
             trace_id: Optional[str] = None) -> ChatResponse:
        self._budget.charge(user.user_id)  # เกินงบ -> BudgetExceeded (ก่อนเรียก RAG/LLM)
        try:
            body = self._rag.retrieve(user, question, box_id, top_k, trace_id)
        except RagError:
            self._budget.refund(user.user_id)  # คำถามไม่ได้รับบริการ — ไม่ตัดโควตาผู้ใช้
            raise
        status = body.get("status")
        results = body.get("results") or []
        out_trace = body.get("trace_id") or trace_id

        if status == "source_required":
            return ChatResponse(status="choose_source", message=MSG_CHOOSE_SOURCE,
                                sources=self.sources(user, out_trace), trace_id=out_trace)

        if status != "ok" or not results:  # no_relevant_context + ค่าที่ไม่รู้จัก (สงวนไว้ในอนาคต) ทำเหมือนกัน
            return ChatResponse(status="not_found", message=MSG_NOT_FOUND, box_id=box_id, trace_id=out_trace)

        visibility = body.get("visibility")
        passages = [
            Passage(n=i, text=r["text"], label=r["citation"]["label"], filename=r["filename"],
                    flags=tuple(r.get("extraction", {}).get("flags", [])))
            for i, r in enumerate(results, start=1)
        ]
        backend = self._router.pick(visibility)
        answer = backend.generate(question, passages)
        citations = [
            Citation(n=p.n, label=p.label, filename=p.filename, file_id=r["file_id"], chunk_id=r["chunk_id"],
                     score=r["score"], text=p.text, flags=list(p.flags))
            for p, r in zip(passages, results)
        ]
        return ChatResponse(
            status="answered", answer=answer, box_id=body.get("box_id"),
            visibility=visibility if visibility in ("public", "private") else None,
            llm=LlmInfo(name=backend.name, location=backend.location), citations=citations, trace_id=out_trace)
