"""FastAPI app ของ RAG API Service (contract v1) — Gateway ขององค์กรเป็นผู้เรียก

server-to-server: ไม่เปิด CORS (browser ไม่ได้เรียกตรง) · export OpenAPI ด้วย ``scripts/export_openapi.py``
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from app.api import (routes_boxes, routes_files, routes_health, routes_ingest, routes_jobs, routes_retrieve,
                     routes_sources, schemas)
from app.api.deps import get_settings
from app.api.errors import TRACE_HEADER, TraceIdMiddleware, UploadLimitMiddleware, install_error_handlers
from app.api.readiness import log_runtime
from app.api.services import services_of
from app.api.schemas import API_VERSION, EX_FILE_ACL
from app.runtime import configure_logging

_AUTO_INGEST_FORM = "Body_ingest_v1_ingest_post"
INGEST_FORM_EXAMPLE = {
    "file": "<binary: งบการเงิน_ตัวอย่าง.pdf>",
    "box_id": "example-finance",
    "file_acl": json.dumps(EX_FILE_ACL, ensure_ascii=False),
    "display_name": "งบการเงิน_ตัวอย่าง",
}

TITLE = "RAG API Service"
DESCRIPTION = (
    "บริการค้นหลักฐานจากเอกสารองค์กรให้ Enterprise AI Gateway (เราไม่ generate คำตอบ) · "
    "service auth ด้วย X-Service-Key · ตัวตนผู้ใช้ปลายทางทาง header X-User-* · "
    "ทุก response มี trace_id (body) และ X-Trace-Id (header) · ไม่มีสิทธิ์ = ไม่เจอ (ไม่มี HTTP Forbidden)"
)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # ตอนเริ่ม process: log ว่าใช้อุปกรณ์/OCR backend อะไร (ไม่ fallback เงียบ) — ready รายงานค่าเดียวกัน
        configure_logging()
        settings = application.dependency_overrides.get(get_settings, get_settings)()
        log_runtime(settings)
        services = services_of(application, settings)  # ที่เก็บถาวร (DATA_DIR/rag.db) + worker
        services.query_embedder.load()  # BGE-M3 บน CPU สำหรับคำถาม (ครั้งเดียว; ไม่ได้ = ready รายงาน down, retrieve 503)
        services.reranker.load()  # RERANKER=off = ไม่ทำอะไร
        try:  # ไฟล์ที่ค้าง deleting จาก process ก่อน: ลบต่อจนครบ (Qdrant ล่ม = worker ลองใหม่ตอนว่าง)
            services.deleter.finish_pending()
        except Exception:
            logging.getLogger("app.api").exception("ลบไฟล์ที่ค้าง deleting ไม่สำเร็จ — worker จะลองใหม่")
        services.worker.start()  # งานที่ค้างกลางทางจาก process ก่อนกลับเข้าคิวที่นี่
        try:
            yield
        finally:
            services.close()

    app = FastAPI(title=TITLE, description=DESCRIPTION, version=API_VERSION, lifespan=lifespan)
    for module in (routes_health, routes_ingest, routes_jobs, routes_sources, routes_retrieve, routes_boxes,
                   routes_files):
        app.include_router(module.router)
    install_error_handlers(app)

    def max_upload_bytes() -> int:  # อ่านจาก Settings ตอน request (ตาม dependency override ของ test ด้วย)
        return app.dependency_overrides.get(get_settings, get_settings)().max_upload_bytes

    # add_middleware ซ้อนจากในออกนอก: TraceId เป็นชั้นนอกสุด (413 จาก UploadLimit ก็ได้ trace id เดียวกัน)
    app.add_middleware(UploadLimitMiddleware, paths=("/v1/ingest",), max_bytes_of=max_upload_bytes)
    app.add_middleware(TraceIdMiddleware)
    app.openapi = lambda: _openapi_with_trace(app)  # type: ignore[method-assign]
    return app


def _openapi_with_trace(app: FastAPI) -> dict:
    """OpenAPI ของ FastAPI + header X-Trace-Id (request: ไม่บังคับ · response: ทุก status) ซึ่งทำใน middleware
    จึงไม่ปรากฏเองใน schema"""
    if app.openapi_schema:
        return app.openapi_schema
    spec = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
    trace_param = {"name": TRACE_HEADER, "in": "header", "required": False,
                   "description": "ไม่ส่ง (หรือรูปแบบไม่ใช่ [A-Za-z0-9._:-]{1,128}) = สร้างให้; ส่งกลับทั้ง header และ body",
                   "schema": {"type": "string", "maxLength": 128}}
    trace_header = {"description": "trace id ของ request นี้ (ค่าเดียวกับ trace_id ใน body)",
                    "schema": {"type": "string"}}
    # body multipart ที่ FastAPI สร้างเอง: ตั้งชื่อให้อ่านรู้เรื่อง + ใส่ตัวอย่าง (ทุก model ต้องมี example)
    raw = json.dumps(spec, ensure_ascii=False).replace(_AUTO_INGEST_FORM, "IngestForm")
    spec = json.loads(raw)
    components = spec.get("components", {}).get("schemas", {})
    form = components.get("IngestForm")
    if form is not None:
        form["examples"] = [INGEST_FORM_EXAMPLE]
    # get_openapi เข้ารหัสทั้งก้อนด้วย exclude_none -> ค่า null ในตัวอย่างหาย (เช่น box_id: null ของ source_required)
    # ตัวอย่างจึงผิด schema; ใส่ตัวอย่างต้นฉบับจาก model กลับ
    for name, schema in components.items():
        model = getattr(schemas, name, None)
        extra = getattr(model, "model_config", {}).get("json_schema_extra") if isinstance(model, type) else None
        if isinstance(extra, dict) and "examples" in extra:
            schema["examples"] = json.loads(json.dumps(extra["examples"], ensure_ascii=False))
    for ops in spec.get("paths", {}).values():
        for op in ops.values():
            op.setdefault("parameters", []).append(trace_param)
            for resp in op.get("responses", {}).values():
                resp.setdefault("headers", {})[TRACE_HEADER] = trace_header
    app.openapi_schema = spec
    return spec


app = create_app()
