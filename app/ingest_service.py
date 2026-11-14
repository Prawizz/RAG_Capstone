"""รับไฟล์เข้ากล่อง (ฝั่ง request) — ตรวจสิทธิ์, ชนิดไฟล์, ไฟล์ซ้ำ, เก็บต้นฉบับ แล้วเข้าคิวให้ ``app/ingest_worker.py``

ลำดับใน ``submit`` (ผิดข้อแรกที่เจอ):
1. ผู้ส่งต้องเป็น owner/admin ของกล่อง — ไม่ใช่ / ไม่มีกล่อง -> ``IngestNotFound`` (API ตอบ 404 เหมือนกันทุกกรณี)
2. ``file_acl`` ทำให้กว้างกว่ากล่อง -> ``AclWideningError`` (422 acl_widening)
3. ชนิดไฟล์จาก magic bytes (``app/extraction/detect.py``; ไม่เชื่อนามสกุล) — ไม่รู้จัก หรือเป็นภาพ jpg/png (ไม่อยู่ในรายการที่
   contract รองรับ) -> ``UnsupportedFile`` (415) · ไฟล์เข้ารหัส/เสีย **รับเข้าคิว** แล้วงานล้มพร้อมรหัส encrypted / corrupt
   (Gateway เห็นเหตุผลที่ /v1/jobs)
4. ไฟล์ซ้ำ = content_hash เดียวกันในกล่องเดียวกัน:
   - งานล่าสุด failed ด้วย error ชั่วคราว (``RETRYABLE_JOB_ERRORS``: unavailable / internal / conversion_timeout) หรือ
     done_with_gaps ที่มี gap ``needs_ocr`` (หน้าที่ยังไม่ได้ OCR เพราะ OCR ใช้ไม่ได้ — หลังขั้น OCR หน้าที่ OCR แล้วเป็นได้แค่
     ok / failed) -> สร้างงานใหม่ให้ file_id เดิม (status queued, job_id ใหม่) worker extract + OCR ใหม่ (หน้าที่ OCR แล้ว
     ได้จากแคช) แล้ว ``upsert_chunks`` ลบ point เดิมของไฟล์ก่อนเขียน (ไม่ซ้ำซ้อน)
   - นอกนั้น (done, done_with_gaps ที่ gap เป็น failed/rejected เท่านั้น, กำลังทำ, failed เพราะตัวไฟล์) -> คืนงานเดิมกับสถานะเดิม
   ทั้งสองกรณี duplicate = true และ ``file_acl`` / ``display_name`` ของ request ซ้ำไม่ถูกใช้ (คงของเดิม)
5. ไฟล์ใหม่: เก็บต้นฉบับที่ ``DATA_DIR/files/<file_id>`` แล้วลงทะเบียน FileAcl + ไฟล์ + งาน ในก้อนเดียว
   (อัปโหลดไฟล์เดียวกันพร้อมกัน: UNIQUE(box_id, content_hash) ให้แถวเดียวชนะ อีกคำขอกลายเป็นไฟล์ซ้ำ)
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from app.access import MANAGER_ROLES, AclStore, FileAcl, Principal, effective_acl
from app.api.schemas import RETRYABLE_JOB_ERRORS, FileAclInput, JobStatus
from app.extraction.detect import detect
from app.extraction.errors import ExtractionError
from app.models import DetectedType, ErrorCode, UnitStatus
from app.storage import FileRecord, JobRecord, SqliteStorage

# ชนิดที่ contract v1 รองรับ (docs/API_CONTRACT.md 4.1) — ภาพเดี่ยว jpg/png ไม่อยู่ในรายการ
ACCEPTED_TYPES = frozenset({DetectedType.PDF, DetectedType.DOCX, DetectedType.PPTX, DetectedType.XLSX,
                            DetectedType.DOC, DetectedType.PPT, DetectedType.XLS})
FILENAME_MAX = 255


class IngestNotFound(Exception):
    """ไม่มีกล่อง หรือไม่มีสิทธิ์ส่งไฟล์ — แยกไม่ได้โดยตั้งใจ"""


class UnsupportedFile(Exception):
    pass


@dataclass
class Submitted:
    job: JobRecord
    file: FileRecord
    duplicate: bool


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def clean_filename(display_name: Optional[str], upload_name: Optional[str]) -> str:
    for raw in (display_name, upload_name):
        if raw and raw.strip():
            name = Path(raw.strip().replace("\\", "/")).name.strip()  # ตัด path ที่ client ใส่มา
            if name:
                return name[:FILENAME_MAX]
    return "ไม่มีชื่อ"


def waiting_for_ocr(job: JobRecord) -> bool:
    """งานจบแล้วแต่มีหน้าที่ยังไม่ได้ OCR (OCR ใช้ไม่ได้ตอนนั้น)"""
    return job.status == JobStatus.DONE_WITH_GAPS.value and any(
        g.get("status") == UnitStatus.NEEDS_OCR.value for g in job.gaps)


def retryable(job: JobRecord) -> bool:
    """ส่งไฟล์เดิมซ้ำแล้วควรเข้าคิวใหม่: ล้มด้วยเหตุชั่วคราว หรือยังมีหน้ารอ OCR"""
    transient = {c.value for c in RETRYABLE_JOB_ERRORS}
    return (job.status == JobStatus.FAILED.value and job.error_code in transient) or waiting_for_ocr(job)


class IngestService:
    def __init__(self, storage: SqliteStorage, acl: AclStore, notify: Callable[[], None] = lambda: None) -> None:
        self.storage = storage
        self.acl = acl
        self.notify = notify

    def can_submit(self, principal: Principal, box_id: str) -> bool:
        return self.acl.role_of(principal.user_id, box_id) in MANAGER_ROLES

    def submit(self, principal: Principal, box_id: str, data: bytes, *, upload_name: Optional[str] = None,
               display_name: Optional[str] = None, file_acl: Optional[FileAclInput] = None) -> Submitted:
        if not self.can_submit(principal, box_id):
            raise IngestNotFound(box_id)
        box = self.acl.get_box(box_id)
        acl_input = FileAcl(
            file_id="pending", box_id=box_id,
            departments=None if file_acl is None or file_acl.departments is None else frozenset(file_acl.departments),
            min_security_level=None if file_acl is None else file_acl.min_security_level,
        )
        effective_acl(box, acl_input)  # กว้างกว่ากล่อง -> AclWideningError ก่อนแตะอะไร

        tmp = self.storage.data_dir / "tmp" / uuid.uuid4().hex
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(data)
        try:
            detected = self._detect(tmp)
            content_hash = hashlib.sha256(data).hexdigest()
            dup = self._duplicate(box_id, content_hash, principal.user_id)
            if dup is not None:
                return dup
            file_id, job_id = new_id("file"), new_id("job")
            final = self.storage.original_path(file_id)
            os.replace(tmp, final)
            try:
                with self.storage.transaction():
                    self.acl.register_file(acl_input.model_copy(update={"file_id": file_id}))
                    file = self.storage.insert_file(FileRecord(
                        file_id=file_id, box_id=box_id, filename=clean_filename(display_name, upload_name),
                        content_hash=content_hash, size_bytes=len(data),
                        detected_type=None if detected is None else detected.value,
                        status=JobStatus.QUEUED.value, submitted_by=principal.user_id))
                    job = self.storage.insert_job(JobRecord(
                        job_id=job_id, file_id=file_id, box_id=box_id, submitted_by=principal.user_id,
                        status=JobStatus.QUEUED.value))
            except sqlite3.IntegrityError:  # อีกคำขอส่งไฟล์เดียวกันเข้ากล่องเดียวกันชนะไปก่อน
                self.acl.remove_file(file_id)
                final.unlink(missing_ok=True)
                dup = self._duplicate(box_id, content_hash, principal.user_id)
                if dup is None:
                    raise
                return dup
            except BaseException:
                self.acl.remove_file(file_id)
                final.unlink(missing_ok=True)
                raise
        finally:
            tmp.unlink(missing_ok=True)
        self.notify()
        return Submitted(job=job, file=file, duplicate=False)

    @staticmethod
    def _detect(path: Path) -> Optional[DetectedType]:
        """ชนิดจาก magic bytes — None = รับไว้แต่ระบุไม่ได้ตอนนี้ (เข้ารหัส/เสีย: งานจะล้มพร้อมรหัสตอน extract)"""
        try:
            detected = detect(path)
        except ExtractionError as e:
            if e.code == ErrorCode.UNSUPPORTED_TYPE:
                raise UnsupportedFile(e.message) from None
            return None
        if detected not in ACCEPTED_TYPES:
            raise UnsupportedFile(f"ไม่รองรับไฟล์ชนิด {detected.value} (รองรับ: PDF, DOCX, PPTX, XLSX, DOC, PPT, XLS)")
        return detected

    def _duplicate(self, box_id: str, content_hash: str, submitter: str) -> Optional[Submitted]:
        existing = self.storage.find_file(box_id, content_hash)
        if existing is None:
            return None
        job = self.storage.latest_job_for_file(existing.file_id)
        if job is None or retryable(job):
            with self.storage.transaction():
                job = self.storage.insert_job(JobRecord(
                    job_id=new_id("job"), file_id=existing.file_id, box_id=box_id,
                    submitted_by=submitter, status=JobStatus.QUEUED.value))  # ผู้ส่งซ้ำเป็นเจ้าของงานใหม่
                self.storage.update_file(existing.file_id, status=JobStatus.QUEUED.value)
            self.notify()
            existing = self.storage.get_file(existing.file_id)
        return Submitted(job=job, file=existing, duplicate=True)
