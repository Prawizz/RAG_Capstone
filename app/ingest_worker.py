"""ingest worker: thread เดียวใน process ของ API ทำงานทีละงานตามลำดับเข้าคิว (FIFO)

ขั้นของงาน (สถานะใน jobs + files อัปเดตทุกขั้น):
  queued -> extracting (``extract_file``) -> ocr (``run_ocr`` เฉพาะ unit needs_ocr, ใช้แคช OCR เดิม; ข้ามถ้าไม่มี)
  -> indexing (``make_document_chunks`` -> embed -> ``upsert_chunks`` ซึ่งลบ point เก่าของไฟล์ก่อนเสมอ)
  -> done | done_with_gaps | failed
progress: ``pages_total`` = จำนวน unit (หน้า/สไลด์/ส่วน/ช่วงชีต) · ``pages_done`` = unit ที่ไม่ต้อง OCR + unit ที่ OCR แล้ว

ผลลัพธ์ของงาน (จาก ``Document.status``):
- done / done_with_gaps -> ตามนั้น (gaps = unit ที่ไม่ ok พร้อมเหตุผล) · หน้าที่ OCR ใช้ไม่ได้ (backend สร้างไม่ได้ หรือ
  ``run_ocr`` เจอ ``OcrUnavailable``) = gap สถานะ ``needs_ocr`` reason ``OCR ใช้ไม่ได้: <เหตุผล>`` — ส่งไฟล์ซ้ำ = เข้าคิวใหม่
  (``app/ingest_service.py``) · หน้าที่ OCR แล้วไม่ผ่านด่านตรวจ = gap ``failed`` (ส่งซ้ำได้ผลเดิม)
- Document.error (เข้ารหัส/เสีย/ว่าง ฯลฯ) -> failed พร้อม error ของ Document
- ไม่มี unit ok เลย และมีหน้าที่ยังรอ OCR เพราะ OCR ใช้ไม่ได้ -> failed ``unavailable`` (ชั่วคราว: ส่งซ้ำได้)
- ไม่มี unit ที่ใช้ได้เลย (ทุก unit failed/rejected) -> failed ``empty``
- vector DB / embedding โหลดไม่ได้ -> failed ``unavailable`` · error อื่น -> failed ``internal`` (log พร้อม traceback)

GPU (อ้างอิง 8 GB): OCR กับ embedding ห้ามอยู่บน GPU พร้อมกัน — ก่อน OCR ปล่อย embedder (ถ้าไม่ได้อยู่บน CPU),
หลัง OCR ปล่อยโมเดล OCR (``ocr_typhoon.unload``) ก่อนโหลด embedder
CUDA error ที่ context ใช้ต่อไม่ได้ (``is_fatal_cuda``) -> งานนั้น failed ``unavailable`` + ``mark_gpu_broken``: worker หยุดหยิบงาน
(งานที่เหลือคงสถานะ queued) และ ``/v1/health/ready`` รายงาน ocr/embedding ที่ใช้ GPU เป็น down จนกว่าจะเริ่ม process ใหม่
เริ่ม worker (``start``) -> งานที่ค้างกลางทาง (extracting/ocr/indexing จาก process ก่อนที่ตาย) กลับเข้าคิวก่อน
ผล extract ทั้งไฟล์ (หลัง OCR) เก็บที่ ``DATA_DIR/documents/<file_id>.json`` ไว้ re-index
ตอนว่าง (ไม่มีงาน) เรียก ``idle_hook`` อย่างมากทุก ``idle_seconds`` — ใช้ลบไฟล์ที่ค้างสถานะ deleting ต่อจนครบ
(``FileDeleter.finish_pending``) · job ของไฟล์ที่ถูกลบ/กำลังลบไม่ถูกหยิบ (``SqliteStorage.claim_next_job``)
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Callable, Optional

from app.access import AclStore
from app.api.schemas import Gap, JobErrorCode, JobStatus
from app.chunking import make_document_chunks
from app.config import Settings
from app.embedding import Embedder, EmbeddingUnavailable
from app.extraction.ocr import OcrBackend, OcrUnavailable, ocr_unavailable_unit, run_ocr
from app.extraction.pipeline import extract_file
from app.models import Document, DocumentStatus, UnitStatus
from app.runtime import DeviceUnavailable, gpu_broken_reason, is_fatal_cuda, mark_gpu_broken
from app.storage import JobRecord, SqliteStorage, utcnow
from app.vectorstore import VectorStore, embed_chunks

log = logging.getLogger("app.ingest_worker")

POLL_SECONDS = 2.0
IDLE_SECONDS = 30.0


class JobFailed(Exception):
    def __init__(self, code: JobErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def unload_ocr_models() -> None:
    """ปล่อยโมเดล OCR ที่โหลดอยู่ (ไม่ import ของหนักถ้ายังไม่เคยใช้)"""
    mod = sys.modules.get("app.extraction.ocr_typhoon")
    if mod is not None:
        mod.unload()


def unit_counts(doc: Document) -> dict[str, int]:
    out = {s.value: 0 for s in (UnitStatus.OK, UnitStatus.NEEDS_OCR, UnitStatus.FAILED, UnitStatus.REJECTED)}
    for u in doc.units:
        out[u.status.value] += 1
    return out


def build_gaps(doc: Document) -> list[dict]:
    gaps = []
    for u in doc.units:
        if u.status == UnitStatus.OK:
            continue
        gaps.append(Gap(unit_index=u.index, label=u.label, status=u.status, error_code=None,
                        reason=u.reason or u.status.value).model_dump(mode="json"))
    return gaps


def mark_ocr_unavailable(doc: Document, note: str) -> Document:
    """OCR backend สร้างไม่ได้: ทุกหน้าที่รอ OCR ได้ reason เดียวกับที่ ``run_ocr`` ให้เมื่อเจอ ``OcrUnavailable``"""
    units = [ocr_unavailable_unit(u, note) if u.status == UnitStatus.NEEDS_OCR else u for u in doc.units]
    return doc.model_copy(update={"units": units})


class IngestWorker:
    def __init__(
        self,
        storage: SqliteStorage,
        acl: AclStore,
        settings: Settings,
        *,
        vector_store_factory: Callable[[], VectorStore],
        embedder_factory: Callable[[], Embedder],
        ocr_backend_factory: Callable[[], OcrBackend],
        ocr_unloader: Callable[[], None] = unload_ocr_models,
        poll_seconds: float = POLL_SECONDS,
        idle_hook: Optional[Callable[[], object]] = None,
        idle_seconds: float = IDLE_SECONDS,
    ) -> None:
        self.storage = storage
        self.acl = acl
        self.settings = settings
        self._vector_store_factory = vector_store_factory
        self._embedder_factory = embedder_factory
        self._ocr_backend_factory = ocr_backend_factory
        self._ocr_unloader = ocr_unloader
        self._poll = poll_seconds
        self._idle_hook = idle_hook
        self._idle_seconds = idle_seconds
        self._next_idle = 0.0
        self._embedder: Optional[Embedder] = None
        self._store: Optional[VectorStore] = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.events: list[str] = []  # ลำดับการโหลด/ปล่อยโมเดล (ไว้ตรวจใน test และ debug)

    # ---------- วงจรชีวิต ----------

    def recover(self) -> list[str]:
        requeued = self.storage.requeue_interrupted()
        if requeued:
            log.warning("งานที่ค้างกลางทางจาก process ก่อน กลับเข้าคิว: %s", ", ".join(requeued))
        return requeued

    def start(self) -> None:
        if self._thread is not None:
            return
        self.recover()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ingest-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def notify(self) -> None:
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                did = self.process_next()
            except Exception:  # กันไม่ให้ thread ตาย (process_next จับ error ของงานเองแล้ว — นี่คือ error ของ storage)
                log.exception("ingest worker: error นอกงาน")
                did = None
            if did is None:
                self.run_idle()
                self._wake.wait(self._poll)
                self._wake.clear()

    def run_idle(self, force: bool = False) -> None:
        if self._idle_hook is None or (not force and time.monotonic() < self._next_idle):
            return
        self._next_idle = time.monotonic() + self._idle_seconds
        try:
            self._idle_hook()
        except Exception:
            log.exception("ingest worker: idle hook ล้ม")

    # ---------- งานหนึ่งชิ้น ----------

    def process_next(self) -> Optional[str]:
        """ทำงานที่ queued เก่าสุดหนึ่งงาน คืน job_id (ไม่มีงาน หรือ GPU ใช้ต่อไม่ได้ = None — งานคงอยู่ในคิว)"""
        if gpu_broken_reason() is not None:
            return None
        job = self.storage.claim_next_job()
        if job is None:
            return None
        try:
            self._process(job)
        except JobFailed as e:
            self._fail(job, e.code, e.message)
        except Exception as e:
            if is_fatal_cuda(e):
                reason = f"{type(e).__name__}: {e}"
                mark_gpu_broken(reason)
                self._embedder = None  # ไม่เรียก release (แตะ CUDA context ที่เสียแล้ว) — ทิ้ง reference อย่างเดียว
                self._fail(job, JobErrorCode.UNAVAILABLE, f"GPU ใช้ต่อไม่ได้ (CUDA error) — ส่งไฟล์เดิมซ้ำหลังเริ่มระบบใหม่: {reason}")
            else:
                log.exception("ingest job %s ล้ม", job.job_id)
                self._fail(job, JobErrorCode.INTERNAL, f"{type(e).__name__}: {e}")
        return job.job_id

    def _set(self, job: JobRecord, status: JobStatus, **job_fields) -> None:
        with self.storage.transaction():
            self.storage.update_job(job.job_id, status=status.value, **job_fields)
            self.storage.update_file(job.file_id, status=status.value)

    def _fail(self, job: JobRecord, code: JobErrorCode, message: str) -> None:
        log.warning("ingest job %s failed (%s): %s", job.job_id, code.value, message)
        with self.storage.transaction():
            self.storage.update_job(job.job_id, status=JobStatus.FAILED.value, error_code=code.value,
                                    error_message=message, finished_at=utcnow())
            self.storage.update_file(job.file_id, status=JobStatus.FAILED.value)

    def _process(self, job: JobRecord) -> None:
        file = self.storage.get_file(job.file_id)
        path = self.storage.original_path(file.file_id)
        log.info("ingest job %s: %s (%s)", job.job_id, file.filename, file.file_id)

        doc = extract_file(path, file_id=file.file_id).model_copy(update={"filename": file.filename})
        self.storage.update_file(file.file_id, detected_type=None if doc.detected_type is None else doc.detected_type.value)
        if doc.error is not None:
            self._save_document(doc)
            with self.storage.transaction():
                self.storage.update_job(job.job_id, document_status=doc.status.value)
                self.storage.update_file(file.file_id, units_total=0, gaps=0)
            message = doc.error.message  # ห้ามเผย path บนเซิร์ฟเวอร์ (DATA_DIR/files/<file_id>) ให้ Gateway
            for raw in (str(path), str(path).replace("\\", "\\\\")):
                message = message.replace(raw, file.filename)
            raise JobFailed(JobErrorCode(doc.error.code.value), message)

        total = len(doc.units)
        done = sum(1 for u in doc.units if u.status != UnitStatus.NEEDS_OCR)
        self.storage.update_job(job.job_id, pages_total=total, pages_done=done)

        if done < total:
            self._set(job, JobStatus.OCR)
            doc = self._ocr(job, doc, path, done)
        self._save_document(doc)

        self._set(job, JobStatus.INDEXING, pages_done=total)
        self._index(doc)

        counts = unit_counts(doc)
        gaps = build_gaps(doc)
        status = doc.status
        with self.storage.transaction():
            self.storage.update_job(job.job_id, document_status=status.value, gaps=gaps, units=counts)
            self.storage.update_file(file.file_id, units_total=total, units_ok=counts["ok"],
                                     units_needs_ocr=counts["needs_ocr"], units_failed=counts["failed"],
                                     units_rejected=counts["rejected"], gaps=len(gaps))
        if status == DocumentStatus.NEEDS_OCR:  # ไม่มี unit ok เลย และหน้าที่ต้อง OCR ยังไม่ได้ OCR (OCR ใช้ไม่ได้)
            waiting = [u for u in doc.units if u.status == UnitStatus.NEEDS_OCR]
            raise JobFailed(JobErrorCode.UNAVAILABLE,
                            f"ไม่มีส่วนที่ใช้ได้: {len(waiting)} หน้าที่ต้อง OCR ยังไม่ได้ OCR — {waiting[0].reason} "
                            "(ส่งไฟล์เดิมซ้ำเมื่อ OCR พร้อม)")
        if status == DocumentStatus.FAILED:
            raise JobFailed(JobErrorCode.EMPTY, "ไม่มีส่วนที่ใช้ได้เลย (ดู gaps)")
        final = JobStatus.DONE if status == DocumentStatus.DONE else JobStatus.DONE_WITH_GAPS
        now = utcnow()
        with self.storage.transaction():
            self.storage.update_job(job.job_id, status=final.value, finished_at=now)
            self.storage.update_file(file.file_id, status=final.value, ingested_at=now)
        log.info("ingest job %s: %s (ok %d / %d)", job.job_id, final.value, counts["ok"], total)

    def _ocr(self, job: JobRecord, doc: Document, path, done: int) -> Document:
        self._release_embedder()  # ห้ามมี embedding กับ OCR บน GPU พร้อมกัน
        try:
            backend = self._ocr_backend_factory()
        except OcrUnavailable as e:
            log.warning("ingest job %s: OCR ใช้ไม่ได้ — %s", job.job_id, e)
            return mark_ocr_unavailable(doc, str(e))
        self.events.append("ocr:start")
        progress = [done]

        def on_unit(_unit) -> None:
            progress[0] += 1
            self.storage.update_job(job.job_id, pages_done=progress[0])

        try:
            out = run_ocr(doc, path, backend, on_unit=on_unit)  # OcrUnavailable รายหน้า -> หน้านั้นคง needs_ocr
            waiting = sum(1 for u in out.units if u.status == UnitStatus.NEEDS_OCR)
            if waiting:
                log.warning("ingest job %s: OCR ใช้ไม่ได้ %d หน้า (คง needs_ocr)", job.job_id, waiting)
            return out
        except OcrUnavailable as e:  # กันพลาด (run_ocr จับรายหน้าแล้ว)
            log.warning("ingest job %s: OCR ใช้ไม่ได้ — %s", job.job_id, e)
            return mark_ocr_unavailable(doc, str(e))
        finally:
            if gpu_broken_reason() is None:
                self._ocr_unloader()
                self.events.append("ocr:unloaded")

    def _index(self, doc: Document) -> None:
        ok = [u for u in doc.units if u.status == UnitStatus.OK and u.text.strip()]
        chunks = make_document_chunks(ok, self.settings.chunk_target_chars)
        indexed = embed_chunks(chunks, self._get_embedder()) if chunks else []
        try:
            store = self._get_store()
            store.upsert_chunks(doc, indexed)  # ลบ point เก่าของไฟล์ทั้งสอง collection ก่อนเขียนเสมอ
        except JobFailed:
            raise
        except Exception as e:
            if is_fatal_cuda(e):
                raise
            self._store = None  # เชื่อมใหม่รอบหน้า
            raise JobFailed(JobErrorCode.UNAVAILABLE, f"vector store ใช้ไม่ได้: {type(e).__name__}: {e}") from e

    # ---------- โมเดล / ที่เก็บ ----------

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._ocr_unloader()  # กันพลาด: OCR ต้องไม่อยู่บน GPU ตอนโหลด embedding
            try:
                self._embedder = self._embedder_factory()
            except (EmbeddingUnavailable, DeviceUnavailable) as e:
                if is_fatal_cuda(e) or is_fatal_cuda(e.__cause__ or e):
                    raise
                raise JobFailed(JobErrorCode.UNAVAILABLE, f"embedding ใช้ไม่ได้: {e}") from e
            self.events.append("embedder:loaded")
        return self._embedder

    def _release_embedder(self) -> None:
        if self._embedder is not None and getattr(self._embedder, "device", None) != "cpu":
            self._embedder.release()
            self._embedder = None
            self.events.append("embedder:released")

    def _get_store(self) -> VectorStore:
        if self._store is None:
            store = self._vector_store_factory()
            store.ensure_collections()
            self._store = store
        return self._store

    def _save_document(self, doc: Document) -> None:
        path = self.storage.document_path(doc.file_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(doc.model_dump_json(), encoding="utf-8")
        tmp.replace(path)
