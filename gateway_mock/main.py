"""mock Gateway (FastAPI)

    export SERVICE_KEY=<ค่าเดียวกับที่ RAG API ใช้>
    uvicorn gateway_mock.main:create_app --factory --port 9000
    curl -s localhost:9000/chat -H 'X-Demo-User: wanida.demo' -H 'Content-Type: application/json' \\
         -d '{"question":"เงินปันผลต่อหุ้นเท่าไร","box_id":"annual-reports"}'

ตัวแปรสภาพแวดล้อม: ``RAG_BASE_URL`` (ค่าเริ่มต้น http://127.0.0.1:8000) · ``SERVICE_KEY`` · ``GATEWAY_BUDGET_PER_USER`` (ค่าเริ่มต้น 50, 0 = ไม่จำกัด)
ผู้ใช้ยืนยันตัวตนด้วย header ``X-Demo-User`` (ชื่อใน ``users.py``) — ของจริงคือ SSO ขององค์กร · ``X-User-*`` ที่ผู้ใช้ส่งมาถูกเมินเสมอ
"""

from __future__ import annotations

import os
import uuid
from typing import Annotated, Optional

import httpx
from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from gateway_mock.budget import Budget, BudgetExceeded
from gateway_mock.llm import ExtractiveLLM, LLMRouter
from gateway_mock.rag_client import RagBadRequest, RagClient, RagError, RagMisconfigured, RagUnavailable
from gateway_mock.schemas import ChatRequest, ChatResponse, MeResponse, SourceInfo
from gateway_mock.service import ChatService
from gateway_mock.users import User, UserDirectory

DEFAULT_RAG_URL = "http://127.0.0.1:8000"
DEFAULT_BUDGET = 50


class GatewayError(Exception):
    def __init__(self, status: int, code: str, message: str, trace_id: Optional[str] = None) -> None:
        self.status, self.code, self.message, self.trace_id = status, code, message, trace_id


def current_user(request: Request, x_demo_user: Annotated[Optional[str], Header()] = None) -> User:
    user = request.app.state.users.get(x_demo_user)
    if user is None:  # ไม่มี header / ไม่รู้จัก ตอบเหมือนกัน
        raise GatewayError(401, "unauthorized", "ยังไม่ได้ยืนยันตัวตน")
    return user


# alias ต้องอยู่ระดับ module: ไฟล์นี้ใช้ ``from __future__ import annotations`` FastAPI จึง resolve ชื่อจาก globals ของ module
CurrentUser = Annotated[User, Depends(current_user)]
TraceId = Annotated[Optional[str], Header(alias="X-Trace-Id")]


def create_app(*, rag: Optional[RagClient] = None, router: Optional[LLMRouter] = None,
               budget: Optional[Budget] = None, users: Optional[UserDirectory] = None) -> FastAPI:
    """สร้างแอป — ส่ง ``rag`` / ``router`` / ``budget`` / ``users`` เองได้ (test) ไม่ส่ง = อ่านจาก env"""
    rag = rag or RagClient(os.environ.get("RAG_BASE_URL", DEFAULT_RAG_URL), os.environ.get("SERVICE_KEY", ""))
    router = router or LLMRouter(ExtractiveLLM("LLM ภายใน (จำลอง)", "internal"),
                                 ExtractiveLLM("LLM ภายนอก (จำลอง)", "external"))
    budget = budget or Budget(int(os.environ.get("GATEWAY_BUDGET_PER_USER", DEFAULT_BUDGET)))
    users = users or UserDirectory()
    service = ChatService(rag, router, budget)

    app = FastAPI(title="Mock Enterprise AI Gateway", version="0.1.0")
    app.state.users = users

    @app.exception_handler(GatewayError)
    def _gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={
            "error_code": exc.code, "message": exc.message, "trace_id": exc.trace_id or uuid.uuid4().hex})

    @app.exception_handler(RequestValidationError)
    def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={
            "error_code": "invalid_request", "message": "request ไม่ถูกต้อง", "trace_id": uuid.uuid4().hex})

    def upstream(exc: RagError) -> GatewayError:
        if isinstance(exc, RagUnavailable):  # ห้ามแปลงเป็น \"ไม่พบข้อมูล\"
            return GatewayError(503, "unavailable", "ระบบค้นเอกสารไม่พร้อมใช้งานชั่วคราว กรุณาลองใหม่ภายหลัง", exc.trace_id)
        if isinstance(exc, RagMisconfigured):
            return GatewayError(502, "upstream_misconfigured", "Gateway เชื่อมต่อระบบค้นเอกสารไม่สำเร็จ (ค่าตั้งผิด)", exc.trace_id)
        if isinstance(exc, RagBadRequest):
            return GatewayError(422, "invalid_request", "คำขอไม่ถูกต้อง (เช่น รหัสกล่องผิดรูปแบบ)", exc.trace_id)
        return GatewayError(502, "upstream_error", "ระบบค้นเอกสารตอบกลับผิดปกติ", exc.trace_id)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/me", response_model=MeResponse)
    def me(user: CurrentUser) -> MeResponse:
        return MeResponse(user_id=user.user_id, departments=user.departments, security_level=user.security_level,
                          budget_used=budget.used(user.user_id))

    @app.get("/sources", response_model=list[SourceInfo])
    def sources(user: CurrentUser, trace_id: TraceId = None) -> list[SourceInfo]:
        try:
            return service.sources(user, trace_id)
        except RagError as e:
            raise upstream(e) from e

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, user: CurrentUser, trace_id: TraceId = None) -> ChatResponse:
        try:
            return service.chat(user, body.question, body.box_id, body.top_k, trace_id)
        except BudgetExceeded:
            raise GatewayError(429, "budget_exceeded", "ใช้โควตาคำถามครบแล้ว", trace_id) from None
        except RagError as e:
            raise upstream(e) from e

    return app
