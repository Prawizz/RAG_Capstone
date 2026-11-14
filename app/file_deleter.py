"""ลบไฟล์ออกจากระบบ (``DELETE /v1/files/{file_id}``) — ทำต่อได้ถ้าพังกลางทาง

ใครลบได้: owner/admin ของกล่องที่ไฟล์อยู่ **และ** เห็นไฟล์นั้น (ACL ระดับไฟล์) — ไม่ใช่ / ไม่มีไฟล์ / ลบไปแล้ว / กำลังลบ
-> ``DeleteNotFound`` (API ตอบ 404 เหมือนกันทุกกรณี — กฎเหล็กข้อ 9)

ลำดับ:
1. ``mark_deleting`` (ก้อนเดียว ``BEGIN IMMEDIATE`` — ชนกับ ``claim_next_job`` ของ worker ไม่ได้):
   - มีงานกำลังทำ (extracting/ocr/indexing) -> ``DeleteConflict`` (409) ไม่แตะอะไร
   - งานที่ยัง queued -> ``deleting`` (``UPDATE ... WHERE status = 'queued'`` นับแถวที่เปลี่ยน = งานที่ถูกยกเลิก) worker จึงไม่หยิบ
   - ``files.status`` -> ``deleting`` + ``AclStore.withdraw_file`` (ด่าน 2 ตัดทันที) — ไฟล์หายจาก /sources, /retrieve, /jobs
2. ``finish`` (ทุกขั้นเรียกซ้ำได้): point ใน Qdrant (ทั้งสอง collection) -> ``files/<id>`` -> ``documents/<id>.json``
   -> แคช OCR ของ content_hash นี้ **เฉพาะเมื่อไม่มีไฟล์อื่น (กล่องไหนก็ได้ ที่ยังไม่ถูกลบ) ใช้ hash เดียวกัน**
   -> ก้อนเดียว: ``files.status = deleted`` (tombstone — jobs อ้างถึงเป็นประวัติ) + job ``deleting`` -> ``deleted``
   + ลบ FileAcl (``file_acls``)
   vector store ใช้ไม่ได้ -> ``DeleteUnavailable`` (API 503) ไฟล์คงสถานะ deleting (ซ่อนอยู่) แล้วลบต่อภายหลัง
3. ``finish_pending``: ไฟล์ที่ค้าง deleting (process ตายกลางทาง / Qdrant ล่ม) — เรียกตอนเริ่ม API (lifespan) และจาก worker ตอนว่าง

แคช OCR: ไฟล์แคชหนึ่งไฟล์ = หนึ่งหน้า/ชิ้น เก็บ ``fields.content_hash`` (SHA-256 ของไบต์ไฟล์ = ``files.content_hash``)
ผลกับแคช OCR (``ocr_cache`` / ``ocr_cache_entries`` ของ ``DeleteReport``) **ลง log เท่านั้น ห้ามส่งออก API** — "เก็บไว้เพราะมีไฟล์อื่นใช้"
บอกผู้ลบว่ามีไฟล์เนื้อหาเดียวกันในกล่องที่เขาอาจไม่มีสิทธิ์เห็น (กฎเหล็กข้อ 9)
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.access import MANAGER_ROLES, AclStore, Principal
from app.extraction.ocr_cache import cache_dir
from app.storage import DELETED, DELETING, HIDDEN_FILE_STATUSES, IN_PROGRESS, SqliteStorage, utcnow
from app.vectorstore import VectorStore

log = logging.getLogger("app.file_deleter")

# สิ่งที่ทำกับแคช OCR ของเนื้อไฟล์ — ภายใน (log) เท่านั้น
OCR_CACHE_DELETED = "deleted"  # ไม่มีไฟล์อื่นในระบบใช้เนื้อไฟล์เดียวกัน
OCR_CACHE_KEPT_SHARED = "kept_shared"  # ยังมีไฟล์อื่น (กล่องใดก็ได้) ที่มีเนื้อไฟล์เดียวกัน
OCR_CACHE_NONE = "none"  # ไม่มีแคช OCR ของเนื้อไฟล์นี้


class DeleteNotFound(Exception):
    """ไม่มีไฟล์ / ไม่มีสิทธิ์ลบ / ลบไปแล้ว — แยกไม่ได้โดยตั้งใจ"""


class DeleteConflict(Exception):
    def __init__(self, job_id: str, status: str) -> None:
        super().__init__(f"งาน {job_id} ของไฟล์นี้กำลังทำอยู่ ({status})")
        self.job_id = job_id
        self.status = status


class DeleteUnavailable(Exception):
    """vector store ใช้ไม่ได้ — ไฟล์ถูกซ่อนแล้ว (deleting) และจะลบต่อเมื่อกลับมา"""


@dataclass
class DeleteReport:
    file_id: str
    box_id: str
    points_deleted: int
    original_deleted: bool
    document_deleted: bool
    ocr_cache: str  # OCR_CACHE_* — log เท่านั้น
    ocr_cache_entries: int  # log เท่านั้น
    cancelled_jobs: int = 0


def _unlink(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def ocr_cache_entries(directory: Path, content_hash: str) -> list[Path]:
    """ไฟล์แคช OCR ทุกไฟล์ของเนื้อไฟล์นี้ (อ่าน ``fields.content_hash``; ไฟล์เสีย/อ่านไม่ได้ = ข้าม)"""
    out: list[Path] = []
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.json")):
        try:
            fields = json.loads(path.read_text(encoding="utf-8")).get("fields") or {}
        except (OSError, ValueError):
            continue
        if fields.get("content_hash") == content_hash:
            out.append(path)
    return out


class FileDeleter:
    def __init__(self, storage: SqliteStorage, acl: AclStore, store: Callable[[], VectorStore],
                 ocr_cache_dir: Callable[[], Path] = cache_dir) -> None:
        self.storage = storage
        self.acl = acl
        self._store = store
        self._ocr_cache_dir = ocr_cache_dir
        self._lock = threading.Lock()  # finish ทีละไฟล์ (request กับ worker ไม่ลบไฟล์เดียวกันซ้อนกัน)

    # ---------- ขั้น 1 ----------

    def can_delete(self, principal: Principal, file_id: str) -> bool:
        rec = self.storage.get_file(file_id)
        if rec is None or rec.status in HIDDEN_FILE_STATUSES:
            return False
        effective = self.acl.get_effective(file_id)
        return (effective is not None and effective.box_id == rec.box_id
                and self.acl.role_of(principal.user_id, rec.box_id) in MANAGER_ROLES
                and self.acl.can_access(principal, file_id))

    def mark_deleting(self, file_id: str) -> int:
        """ซ่อนไฟล์ + ยกเลิกงาน queued ในก้อนเดียว คืนจำนวนงานที่ยกเลิก (ไม่ตรวจสิทธิ์ — ผู้เรียกตรวจก่อน)"""
        withdrawn = False
        try:
            with self.storage.transaction() as conn:
                row = conn.execute("SELECT status FROM files WHERE file_id = ?", (file_id,)).fetchone()
                if row is None or row["status"] in HIDDEN_FILE_STATUSES:
                    raise DeleteNotFound(file_id)
                marks = ", ".join("?" * len(IN_PROGRESS))
                busy = conn.execute(f"SELECT job_id, status FROM jobs WHERE file_id = ? AND status IN ({marks}) "
                                    "ORDER BY rowid DESC LIMIT 1", (file_id, *IN_PROGRESS)).fetchone()
                if busy is not None:
                    raise DeleteConflict(busy["job_id"], busy["status"])
                now = utcnow()
                cancelled = conn.execute("UPDATE jobs SET status = ?, updated_at = ? WHERE file_id = ? AND status = 'queued'",
                                         (DELETING, now, file_id)).rowcount
                changed = conn.execute("UPDATE files SET status = ?, updated_at = ? WHERE file_id = ? AND status = ?",
                                       (DELETING, now, file_id, row["status"])).rowcount
                if changed != 1:  # ไม่ควรเกิด (BEGIN IMMEDIATE) — กันไว้
                    raise DeleteNotFound(file_id)
                self.acl.withdraw_file(file_id)  # ก่อน commit: ไม่มีช่วงที่ SQLite บอก deleting แต่ด่าน 2 ยังปล่อยผ่าน
                withdrawn = True
        except BaseException:
            if withdrawn:  # commit ล้ม
                self.acl.restore_file(file_id)
            raise
        log.info("delete %s: deleting (ยกเลิกงาน queued %d)", file_id, cancelled)
        return cancelled

    def delete(self, principal: Principal, file_id: str) -> DeleteReport:
        if not self.can_delete(principal, file_id):
            raise DeleteNotFound(file_id)
        cancelled = self.mark_deleting(file_id)
        report = self.finish(file_id)
        report.cancelled_jobs = cancelled
        return report

    # ---------- ขั้น 2 ----------

    def finish(self, file_id: str) -> DeleteReport:
        with self._lock:
            rec = self.storage.get_file(file_id)
            if rec is None or rec.status != DELETING:
                raise DeleteNotFound(file_id)
            try:
                points = self._store().delete_file(file_id)
            except Exception as e:
                log.warning("delete %s: vector store ใช้ไม่ได้ (%s) — คงสถานะ deleting ไว้ลบต่อภายหลัง",
                            file_id, type(e).__name__)
                raise DeleteUnavailable(file_id) from None
            original = _unlink(self.storage.original_path(file_id))
            document = _unlink(self.storage.document_path(file_id))
            _unlink(self.storage.document_path(file_id).with_suffix(".json.tmp"))

            entries = ocr_cache_entries(self._ocr_cache_dir(), rec.content_hash)
            if not entries:
                action = OCR_CACHE_NONE
            elif self.storage.live_files_with_hash(rec.content_hash, exclude_file_id=file_id):
                action = OCR_CACHE_KEPT_SHARED
            else:
                action = OCR_CACHE_DELETED
                for path in entries:
                    _unlink(path)

            now = utcnow()
            with self.storage.transaction() as conn:
                self.storage.update_file(file_id, status=DELETED)
                conn.execute("UPDATE jobs SET status = ?, finished_at = COALESCE(finished_at, ?), updated_at = ? "
                             "WHERE file_id = ? AND status = ?", (DELETED, now, now, file_id, DELETING))
                self.acl.remove_file(file_id)  # ลบแถว file_acls ในก้อนเดียวกัน + หน่วยความจำ
            report = DeleteReport(file_id=file_id, box_id=rec.box_id, points_deleted=points, original_deleted=original,
                                  document_deleted=document, ocr_cache=action, ocr_cache_entries=len(entries))
            log.info("delete %s: deleted (points %d, original %s, document %s, ocr_cache %s %d)", file_id, points,
                     original, document, action, len(entries))
            return report

    def finish_pending(self) -> list[DeleteReport]:
        """ลบไฟล์ที่ค้าง deleting ให้ครบ — vector store ใช้ไม่ได้ = หยุด (ลองใหม่รอบหน้า)"""
        done: list[DeleteReport] = []
        for rec in self.storage.files_with_status(DELETING):
            self.acl.withdraw_file(rec.file_id)  # กันพลาด (ควรถูก withdraw ตั้งแต่เริ่ม process แล้ว)
            try:
                done.append(self.finish(rec.file_id))
            except DeleteUnavailable:
                break
            except DeleteNotFound:  # อีก thread ลบเสร็จไปแล้ว
                continue
        if done:
            log.warning("ลบไฟล์ที่ค้างกลางทางต่อจนครบ: %s", ", ".join(r.file_id for r in done))
        return done

    def withdraw_pending(self) -> list[str]:
        """ตอนเริ่ม process: ไฟล์ที่ค้าง deleting ต้องถูกซ่อนก่อนรับ request แรก (AclStore อยู่ในหน่วยความจำ)"""
        ids = [rec.file_id for rec in self.storage.files_with_status(DELETING)]
        for fid in ids:
            self.acl.withdraw_file(fid)
        return ids
