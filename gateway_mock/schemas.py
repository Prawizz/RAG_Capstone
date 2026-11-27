"""รูปแบบ response ของ mock Gateway (ฝั่งผู้ใช้ปลายทาง/demo console) — ไม่ใช่ contract ของ RAG"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ChatStatus = Literal["answered", "choose_source", "not_found"]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    box_id: Optional[str] = Field(default=None, max_length=128)
    top_k: int = Field(default=5, ge=1, le=20)


class LlmInfo(BaseModel):
    name: str
    location: Literal["internal", "external"]


class Citation(BaseModel):
    n: int
    label: str
    filename: str
    file_id: str
    chunk_id: str
    score: float
    text: str
    flags: list[str] = []


class SourceInfo(BaseModel):
    box_id: str
    name: Optional[str] = None
    visibility: Optional[str] = None


class ChatResponse(BaseModel):
    status: ChatStatus
    message: Optional[str] = None
    answer: Optional[str] = None
    box_id: Optional[str] = None
    visibility: Optional[Literal["public", "private"]] = None
    llm: Optional[LlmInfo] = None
    citations: list[Citation] = []
    sources: list[SourceInfo] = []
    trace_id: Optional[str] = None


class MeResponse(BaseModel):
    user_id: str
    departments: str
    security_level: int
    budget_used: int
