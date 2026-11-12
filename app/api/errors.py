"""error เดียวทั้ง API ({error_code, message, trace_id}) + trace id ต่อ request

- ``TraceIdMiddleware``: อ่าน ``X-Trace-Id`` (รูปแบบไม่ถูก/ไม่ส่ง -> สร้างใหม่) เก็บใน ``request.state.trace_id``
  และใส่ header ``X-Trace-Id`` ในทุก response ที่ผ่าน middleware
- ``UploadLimitMiddleware``: ``Content-Length`` เกินเพดาน (เฉพาะ path ที่กำหนด) -> 413 too_large ก่อนอ่าน body
- ``install_error_handlers``: ``ApiError`` / validation (-> 422 invalid_request) / HTTPException ของ framework /
  error ที่ไม่คาดคิด (-> 500 internal ไม่เผยรายละเอียด) ทุกตัวได้ ErrorResponse + header X-Trace-Id
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api import schemas as s
from app.api.schemas import ERROR_STATUS, ApiErrorCode, ErrorResponse

log = logging.getLogger(__name__)

TRACE_HEADER = "X-Trace-Id"
TRACE_PATTERN = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")


class ApiError(Exception):
    """error ที่ตั้งใจคืนให้ผู้เรียก — status มาจาก ``ERROR_STATUS`` ของ code"""

    def __init__(self, code: ApiErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def status(self) -> int:
        return ERROR_STATUS[self.code]


def not_found() -> ApiError:
    """ไม่มีสิทธิ์ และหาไม่เจอ ใช้ข้อความเดียวกันทุกที่ (กฎเหล็กข้อ 9)"""
    return ApiError(ApiErrorCode.NOT_FOUND, "ไม่พบ")


def not_implemented() -> ApiError:
    return ApiError(ApiErrorCode.NOT_IMPLEMENTED, "endpoint นี้อยู่ในช่วง contract freeze ยังไม่เปิดใช้งาน")


def new_trace_id() -> str:
    return uuid.uuid4().hex


def accept_trace_id(value: Optional[str]) -> str:
    return value if value and TRACE_PATTERN.match(value) else new_trace_id()


def trace_id_of(request: Request) -> str:
    tid = getattr(request.state, "trace_id", None)
    if not tid:  # ไม่ผ่าน middleware (ไม่ควรเกิด) — ยังต้องมี trace_id
        tid = accept_trace_id(request.headers.get(TRACE_HEADER))
        request.state.trace_id = tid
    return tid


def error_response(trace_id: str, code: ApiErrorCode, message: str) -> JSONResponse:
    body = ErrorResponse(error_code=code, message=message, trace_id=trace_id)
    return JSONResponse(body.model_dump(mode="json"), status_code=ERROR_STATUS[code],
                        headers={TRACE_HEADER: trace_id})


class TraceIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        raw = dict(scope.get("headers") or []).get(TRACE_HEADER.lower().encode())
        tid = accept_trace_id(raw.decode("latin-1") if raw else None)
        scope.setdefault("state", {})["trace_id"] = tid

        async def send_with_trace(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != TRACE_HEADER.lower().encode()]
                headers.append((TRACE_HEADER.lower().encode(), tid.encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_trace)


class UploadLimitMiddleware:
    """ปฏิเสธ body ที่ประกาศขนาดเกินเพดาน ก่อน framework อ่าน multipart (ขนาดจริงตรวจซ้ำใน route)"""

    def __init__(self, app: ASGIApp, paths: tuple[str, ...], max_bytes_of) -> None:
        self.app = app
        self.paths = paths
        self.max_bytes_of = max_bytes_of  # callable -> int (อ่านจาก Settings ตอน request)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path") in self.paths:
            raw = dict(scope.get("headers") or []).get(b"content-length")
            try:
                declared = int(raw) if raw is not None else None
            except ValueError:
                declared = None
            limit = self.max_bytes_of()
            if declared is not None and declared > limit:
                tid = scope.get("state", {}).get("trace_id") or new_trace_id()
                resp = error_response(tid, ApiErrorCode.TOO_LARGE, f"ไฟล์ใหญ่เกิน {limit // (1024 * 1024)} MB")
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)


_CODE_EXAMPLE = {
    ApiErrorCode.UNAUTHORIZED: s.EX_ERR_UNAUTHORIZED,
    ApiErrorCode.INVALID_REQUEST: s.EX_ERR_INVALID,
    ApiErrorCode.NOT_FOUND: s.EX_ERR_NOT_FOUND,
    ApiErrorCode.TOO_LARGE: s.EX_ERR_TOO_LARGE,
    ApiErrorCode.UNSUPPORTED_TYPE: s.EX_ERR_UNSUPPORTED,
    ApiErrorCode.ACL_WIDENING: s.EX_ERR_ACL_WIDENING,
    ApiErrorCode.UNAVAILABLE: s.EX_ERR_UNAVAILABLE,
    ApiErrorCode.INTERNAL: s.EX_ERR_INTERNAL,
    ApiErrorCode.NOT_IMPLEMENTED: s.EX_ERR_NOT_IMPLEMENTED,
}
COMMON_ERRORS = (ApiErrorCode.UNAUTHORIZED, ApiErrorCode.INVALID_REQUEST, ApiErrorCode.INTERNAL,
                 ApiErrorCode.NOT_IMPLEMENTED)


def error_docs(*codes: ApiErrorCode, common: bool = True) -> dict[int, dict]:
    """``responses=`` ของ route: ทุก status ใช้ ErrorResponse พร้อมตัวอย่างของแต่ละ error_code
    (422 ที่ระบุเองแทน HTTPValidationError ของ FastAPI — ไม่มี schema validation ของ framework หลุดเข้า OpenAPI)"""
    wanted = list(dict.fromkeys((COMMON_ERRORS if common else ()) + codes))
    out: dict[int, dict] = {}
    for code in wanted:
        status = ERROR_STATUS[code]
        entry = out.setdefault(status, {"model": ErrorResponse, "description": "",
                                        "content": {"application/json": {"examples": {}}}})
        entry["description"] = " / ".join(filter(None, [entry["description"], code.value]))
        entry["content"]["application/json"]["examples"][code.value] = {"value": _CODE_EXAMPLE[code]}
    return out


def _validation_message(exc: RequestValidationError) -> str:
    parts = []
    for err in exc.errors()[:5]:
        loc = [str(x) for x in err.get("loc", ())]
        where = f"header {loc[1]}" if loc[:1] == ["header"] and len(loc) > 1 else ".".join(loc)
        parts.append(f"{where}: {err.get('msg', 'ไม่ถูกต้อง')}")
    return "; ".join(parts) or "request ไม่ถูกต้อง"


_HTTP_TO_CODE = {401: ApiErrorCode.UNAUTHORIZED, 404: ApiErrorCode.NOT_FOUND, 413: ApiErrorCode.TOO_LARGE,
                 415: ApiErrorCode.UNSUPPORTED_TYPE, 503: ApiErrorCode.UNAVAILABLE}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(trace_id_of(request), exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(trace_id_of(request), ApiErrorCode.INVALID_REQUEST, _validation_message(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # path ที่ไม่มี / method ไม่รองรับ ฯลฯ ของ framework — ไม่มี 403 (ห้าม) ; อื่นๆ ที่ไม่อยู่ในรายการ = invalid_request
        code = _HTTP_TO_CODE.get(exc.status_code)
        if code is None:
            code = ApiErrorCode.INVALID_REQUEST if exc.status_code < 500 else ApiErrorCode.INTERNAL
        message = "ไม่พบ" if code == ApiErrorCode.NOT_FOUND else str(exc.detail)
        return error_response(trace_id_of(request), code, message)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        tid = trace_id_of(request)
        log.exception("unhandled error trace_id=%s", tid)
        return error_response(tid, ApiErrorCode.INTERNAL, "เกิดข้อผิดพลาดภายใน (อ้างอิง trace_id)")
