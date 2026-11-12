"""ตรวจความพร้อมสำหรับ ``GET /v1/health/ready`` — เบา ไม่โหลดโมเดล

- qdrant: เรียก ``get_collections`` ด้วย timeout สั้น (``READY_QDRANT_TIMEOUT``) ได้ = ok ไม่ได้ = down
- embedding: ตัดสินอุปกรณ์ด้วย ``app.runtime.resolve_embedding_device`` (ตัวเดียวกับ ``BGEM3Embedder``) — ตั้ง cuda แต่ไม่มี GPU = down,
  โหมด test (hash vector ไม่ใช่ BGE-M3) = degraded
- ocr: ``app.runtime.resolve_ocr_backend`` — typhoon-local ไม่มี GPU = down; tesseract-local ต้องมีโปรแกรมและไฟล์ภาษา (สร้าง
  ``TesseractLocalBackend`` ซึ่งตรวจแค่ไฟล์ ไม่รัน OCR)
สถานะรวม: ready = ทุกตัว ok · not_ready = qdrant หรือ embedding down · degraded = นอกนั้น
health ถูกเรียกบ่อย จึงตัดสินแบบ quiet (ไม่ log ซ้ำทุก probe) — log ของการตัดสินจริงเกิดตอนสร้าง embedder/OCR backend
และตอน API เริ่มทำงาน (``log_runtime``)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.api.schemas import ComponentState, Components, ReadyStatus, Runtime
from app.config import Settings
from app.runtime import DeviceUnavailable, cuda_available, resolve_embedding_device, resolve_ocr_backend

READY_QDRANT_TIMEOUT = 2.0


@dataclass
class Readiness:
    status: ReadyStatus
    components: Components
    runtime: Runtime
    notes: list[str] = field(default_factory=list)


def check_qdrant(settings: Settings) -> ComponentState:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key,
                              timeout=READY_QDRANT_TIMEOUT, check_compatibility=False)
        try:
            client.get_collections()
        finally:
            client.close()
        return ComponentState.OK
    except Exception:
        return ComponentState.DOWN


def check_tesseract() -> tuple[bool, str]:
    from app.extraction.ocr import OcrUnavailable

    try:
        from app.extraction.ocr_tesseract import TesseractLocalBackend

        TesseractLocalBackend()
        return True, ""
    except OcrUnavailable as e:
        return False, str(e)
    except Exception as e:  # dependency ขาด ฯลฯ
        return False, f"{type(e).__name__}: {e}"


def probe(settings: Settings, *, gpu: bool | None = None, qdrant=check_qdrant, tesseract=check_tesseract) -> Readiness:
    gpu = cuda_available() if gpu is None else gpu
    notes: list[str] = []

    try:
        dev = resolve_embedding_device(settings.embedding_device, settings.embedding_dtype, cuda=gpu, quiet=True)
        emb_device, emb_dtype = dev.device, dev.dtype
        if dev.fallback:
            notes.append(f"embedding: {dev.reason}")
        embedding = ComponentState.DEGRADED if settings.embedding_test_mode else ComponentState.OK
        if settings.embedding_test_mode:
            notes.append("embedding: EMBEDDING_TEST_MODE เปิดอยู่ (hash vector ไม่ใช่ BGE-M3)")
    except DeviceUnavailable as e:
        emb_device = emb_dtype = None
        embedding = ComponentState.DOWN
        notes.append(f"embedding: {e}")

    ocr_choice = resolve_ocr_backend(settings.ocr_backend, cuda=gpu, quiet=True)
    if ocr_choice.fallback or not ocr_choice.usable:
        notes.append(f"ocr: {ocr_choice.reason}")
    if not ocr_choice.usable:
        ocr = ComponentState.DOWN
    elif ocr_choice.name == "tesseract-local":
        ok, why = tesseract()
        ocr = ComponentState.OK if ok else ComponentState.DOWN
        if not ok:
            notes.append(f"ocr: {why}")
    else:
        ocr = ComponentState.OK

    components = Components(qdrant=qdrant(settings), embedding=embedding, ocr=ocr)
    if ComponentState.DOWN in (components.qdrant, components.embedding):
        status = ReadyStatus.NOT_READY
    elif all(c == ComponentState.OK for c in (components.qdrant, components.embedding, components.ocr)):
        status = ReadyStatus.READY
    else:
        status = ReadyStatus.DEGRADED
    runtime = Runtime(gpu_available=gpu, embedding_requested=settings.embedding_device, embedding_device=emb_device,
                      embedding_dtype=emb_dtype, ocr_requested=settings.ocr_backend, ocr_backend=ocr_choice.name,
                      notes=notes)
    return Readiness(status=status, components=components, runtime=runtime, notes=notes)


def log_runtime(settings: Settings) -> None:
    """log การตัดสินอุปกรณ์/backend ตอน API เริ่มทำงาน (ไม่ quiet) — ตั้ง cuda บนเครื่องไม่มี GPU ก็ log error ไม่ล้ม process
    (ready จะรายงาน not_ready)"""
    try:
        resolve_embedding_device(settings.embedding_device, settings.embedding_dtype)
    except DeviceUnavailable:
        pass
    resolve_ocr_backend(settings.ocr_backend)
