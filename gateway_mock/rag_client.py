"""ไคลเอนต์ HTTP ของ RAG API (public contract v1 เท่านั้น)

แปลงสถานะของ RAG เป็นข้อผิดพลาด 4 แบบที่ Gateway ต้องแยก: ใช้งานไม่ได้ชั่วคราว · ตั้งค่า service auth ผิด · request ผิด · อื่นๆ
**ห้ามเปลี่ยน unavailable เป็น \"ไม่พบข้อมูล\"** (contract: เราจะไม่ตอบ \"ไม่พบข้อมูล\" แทนตอนส่วนประกอบหลักล่ม)
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from gateway_mock.users import User


class RagError(Exception):
    def __init__(self, message: str, trace_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.trace_id = trace_id


class RagUnavailable(RagError):
    """เชื่อมต่อไม่ได้ / 5xx — ลองใหม่ภายหลังได้"""


class RagMisconfigured(RagError):
    """RAG ปฏิเสธ service credential (401) — เป็นความผิดของค่าตั้ง Gateway ไม่ใช่ของผู้ใช้"""


class RagBadRequest(RagError):
    """RAG ตอบ 4xx ว่า request ผิดรูปแบบ (เช่น box_id ไม่ตรงรูปแบบ)"""


class RagClient:
    def __init__(self, base_url: str, service_key: str, *, transport: Optional[httpx.BaseTransport] = None,
                 timeout: float = 30.0) -> None:
        self._http = httpx.Client(base_url=base_url.rstrip("/"), transport=transport,
                                  timeout=httpx.Timeout(timeout, connect=5.0))
        self._key = service_key

    def close(self) -> None:
        self._http.close()

    def _headers(self, user: User, trace_id: Optional[str]) -> dict[str, str]:
        headers = {"X-Service-Key": self._key, "X-User-Id": user.user_id,
                   "X-User-Departments": user.departments, "X-User-Security-Level": str(user.security_level)}
        if trace_id:
            headers["X-Trace-Id"] = trace_id
        return headers

    def _request(self, method: str, path: str, user: User, trace_id: Optional[str], **kwargs: Any) -> dict:
        try:
            resp = self._http.request(method, path, headers=self._headers(user, trace_id), **kwargs)
        except httpx.TransportError as e:
            raise RagUnavailable(f"เชื่อมต่อ RAG ไม่ได้ ({type(e).__name__})") from e
        try:
            body = resp.json()
        except ValueError:
            body = {}
        upstream_trace = body.get("trace_id") if isinstance(body, dict) else None
        message = body.get("message", "") if isinstance(body, dict) else ""
        if resp.status_code == 200:
            return body
        if resp.status_code >= 500:
            raise RagUnavailable(message or f"RAG ตอบ {resp.status_code}", upstream_trace)
        if resp.status_code == 401:
            raise RagMisconfigured(message or "RAG ปฏิเสธ service credential", upstream_trace)
        if resp.status_code == 422:
            raise RagBadRequest(message or "request ไม่ถูกต้อง", upstream_trace)
        raise RagError(message or f"RAG ตอบ {resp.status_code} ที่ไม่คาดคิด", upstream_trace)

    def retrieve(self, user: User, query: str, box_id: Optional[str], top_k: int,
                 trace_id: Optional[str] = None) -> dict:
        body: dict[str, Any] = {"query": query, "top_k": top_k}
        if box_id:
            body["box_id"] = box_id
        return self._request("POST", "/v1/retrieve", user, trace_id, json=body)

    def sources(self, user: User, trace_id: Optional[str] = None) -> dict:
        return self._request("GET", "/v1/sources", user, trace_id)
