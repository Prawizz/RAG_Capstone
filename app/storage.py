"""ที่เก็บถาวร: SQLite ไฟล์เดียว (``DATA_DIR/rag.db``) ด้วย ``sqlite3`` ในตัว + ไฟล์บนดิสก์

    DATA_DIR/rag.db                  กล่อง, สมาชิก, FileAcl, ทะเบียนไฟล์, jobs
    DATA_DIR/files/<file_id>         ไฟล์ต้นฉบับ (ไบต์ตามที่อัปโหลด)
    DATA_DIR/documents/<file_id>.json  ผล extract ทั้งไฟล์ (``Document`` หลัง OCR) ไว้ re-index โดยไม่ต้อง extract/OCR ใหม่

- connection แยกต่อ thread (API ใช้ thread pool + ingest worker อีก thread) · WAL + busy_timeout
- ``transaction()`` = ``BEGIN IMMEDIATE`` (จองสิทธิ์เขียนตั้งแต่ต้น -> ตรวจแล้วเขียนเป็นก้อนเดียว ไม่มีใครแทรก) ซ้อนกันได้ใน thread เดียว
  (ชั้นในไม่เปิดใหม่) · นอก transaction แต่ละคำสั่ง commit เอง (autocommit)
- ``SqliteMemberships`` = ``MembershipBackend`` ตัวจริง: ``transaction(box_id)`` ใช้ BEGIN IMMEDIATE ของฐานเดียวกัน
  ตรวจ "owner คนสุดท้าย" + เขียน จึงไม่ชนกันแม้มาจากคนละ thread / คนละ process
- เวลาเก็บเป็นข้อความ ISO-8601 UTC (มีไมโครวินาที — เรียงตามตัวอักษรได้ตรงกับเวลา)

ลบไฟล์ (``app/file_deleter.py``): ``files.status`` = ``deleting`` (ซ่อนทันที, กำลังลบของที่เกี่ยวข้อง) -> ``deleted``
(tombstone: แถวคงไว้เพราะ jobs อ้างถึงเป็นประวัติ) · ไฟล์สองสถานะนี้ (``HIDDEN_FILE_STATUSES``) ไม่ปรากฏใน
``files_page`` / ``files_in_box`` / ``find_file`` และ job ของไฟล์เหล่านี้ไม่ถูกหยิบ
schema v2: ``UNIQUE(box_id, content_hash)`` เป็น partial unique index ที่ไม่นับไฟล์ deleting/deleted
(ลบแล้วอัปโหลดไฟล์เดิมเข้ากล่องเดิมได้ = file_id ใหม่) · ฐาน v1 migrate อัตโนมัติตอนเปิด โดยสำรอง
``rag.db`` -> ``rag.db.bak-v1`` ก่อน (SQLite online backup API)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from app.access import BoxRole, Box, FileAcl, Visibility

log = logging.getLogger("app.storage")

SCHEMA_VERSION = 2

META = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

FILES_TABLE = """
CREATE TABLE IF NOT EXISTS {name} (
    file_id         TEXT PRIMARY KEY,
    box_id          TEXT NOT NULL REFERENCES boxes(box_id),
    filename        TEXT NOT NULL,
    content_hash    TEXT NOT NULL,        -- SHA-256 ของไบต์ที่อัปโหลด
    size_bytes      INTEGER NOT NULL,
    detected_type   TEXT,                 -- จาก magic bytes (ตอนรับไฟล์)
    status          TEXT NOT NULL,        -- JobStatus ของงานล่าสุด | deleting | deleted (tombstone)
    units_total     INTEGER NOT NULL DEFAULT 0,
    units_ok        INTEGER NOT NULL DEFAULT 0,
    units_needs_ocr INTEGER NOT NULL DEFAULT 0,
    units_failed    INTEGER NOT NULL DEFAULT 0,
    units_rejected  INTEGER NOT NULL DEFAULT 0,
    gaps            INTEGER NOT NULL DEFAULT 0,
    submitted_by    TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    ingested_at     TEXT                  -- เวลาที่ index เสร็จล่าสุด (done / done_with_gaps)
);
"""

# ไฟล์ซ้ำในกล่องเดียวกัน (กันอัปโหลดพร้อมกันได้สองแถว) — ไม่นับไฟล์ที่ถูกลบ/กำลังลบ
FILE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS files_by_box ON files(box_id, created_at, file_id)",
    "CREATE INDEX IF NOT EXISTS files_by_hash ON files(content_hash)",
    "CREATE UNIQUE INDEX IF NOT EXISTS files_live_hash ON files(box_id, content_hash) "
    "WHERE status NOT IN ('deleting', 'deleted')",
)

SCHEMA = META + """
CREATE TABLE IF NOT EXISTS boxes (
    box_id             TEXT PRIMARY KEY,
    name               TEXT,
    visibility         TEXT NOT NULL CHECK (visibility IN ('public', 'private')),
    departments        TEXT,              -- JSON array (เรียงแล้ว) หรือ NULL = ไม่จำกัดแผนก; '[]' = ไม่มีแผนกไหน
    min_security_level INTEGER NOT NULL CHECK (min_security_level >= 0),
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS members (
    box_id     TEXT NOT NULL REFERENCES boxes(box_id),
    user_id    TEXT NOT NULL,
    role       TEXT NOT NULL CHECK (role IN ('owner', 'admin', 'viewer')),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (box_id, user_id)
);
CREATE INDEX IF NOT EXISTS members_by_user ON members(user_id);
CREATE TABLE IF NOT EXISTS file_acls (
    file_id            TEXT PRIMARY KEY,
    box_id             TEXT NOT NULL REFERENCES boxes(box_id),
    departments        TEXT,              -- JSON array หรือ NULL = สืบทอดจากกล่อง
    min_security_level INTEGER            -- NULL = สืบทอดจากกล่อง
);
""" + FILES_TABLE.format(name="files") + ";\n".join(FILE_INDEXES) + """;
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    file_id         TEXT NOT NULL REFERENCES files(file_id),
    box_id          TEXT NOT NULL,
    submitted_by    TEXT NOT NULL,
    status          TEXT NOT NULL,        -- queued|extracting|ocr|indexing|done|done_with_gaps|failed|deleting|deleted
    pages_total     INTEGER,
    pages_done      INTEGER NOT NULL DEFAULT 0,
    document_status TEXT,
    gaps_json       TEXT NOT NULL DEFAULT '[]',  -- [{unit_index, label, status, error_code, reason}]
    units_json      TEXT NOT NULL DEFAULT '{}',  -- {ok, needs_ocr, failed, rejected} ของงานนี้
    error_code      TEXT,
    error_message   TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT
);
CREATE INDEX IF NOT EXISTS jobs_by_status ON jobs(status);
CREATE INDEX IF NOT EXISTS jobs_by_file ON jobs(file_id);
"""

IN_PROGRESS = ("extracting", "ocr", "indexing")
FINISHED = ("done", "done_with_gaps", "failed")
DELETING = "deleting"
DELETED = "deleted"
HIDDEN_FILE_STATUSES = (DELETING, DELETED)  # ไฟล์ที่ถูกลบ/กำลังลบ: ไม่มีใครเห็น (ไม่อยู่ใน JobStatus ของ contract)
_HIDDEN_SQL = "('deleting', 'deleted')"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _dump_departments(d: Optional[frozenset[str]]) -> Optional[str]:
    return None if d is None else json.dumps(sorted(d), ensure_ascii=False)


def _load_departments(raw: Optional[str]) -> Optional[frozenset[str]]:
    return None if raw is None else frozenset(json.loads(raw))


@dataclass
class FileRecord:
    file_id: str
    box_id: str
    filename: str
    content_hash: str
    size_bytes: int
    status: str
    submitted_by: str
    created_at: str = ""
    updated_at: str = ""
    detected_type: Optional[str] = None
    units_total: int = 0
    units_ok: int = 0
    units_needs_ocr: int = 0
    units_failed: int = 0
    units_rejected: int = 0
    gaps: int = 0
    ingested_at: Optional[str] = None


@dataclass
class JobRecord:
    job_id: str
    file_id: str
    box_id: str
    submitted_by: str
    status: str
    created_at: str = ""
    updated_at: str = ""
    pages_total: Optional[int] = None
    pages_done: int = 0
    document_status: Optional[str] = None
    gaps: list[dict] = field(default_factory=list)
    units: dict = field(default_factory=dict)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


_FILE_COLS = [f.name for f in fields(FileRecord)]
_JSON_COLS = {"gaps": "gaps_json", "units": "units_json"}
_JOB_COLS = [_JSON_COLS.get(f.name, f.name) for f in fields(JobRecord)]


def _job_from_row(row: sqlite3.Row) -> JobRecord:
    data = {k: row[k] for k in row.keys()}
    data["gaps"] = json.loads(data.pop("gaps_json") or "[]")
    data["units"] = json.loads(data.pop("units_json") or "{}")
    return JobRecord(**data)


class SqliteStorage:
    """ที่เก็บถาวรของ service (เป็น ``AclPersistence`` ให้ ``AclStore`` ด้วย) — ดู docstring ของโมดูล"""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.files_dir = self.data_dir / "files"
        self.documents_dir = self.data_dir / "documents"
        for d in (self.data_dir, self.files_dir, self.documents_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "rag.db"
        self._local = threading.local()
        self._all: list[sqlite3.Connection] = []
        self._all_lock = threading.Lock()
        self.migrated_from: Optional[int] = None  # schema เดิมก่อน migrate (None = ไม่ได้ migrate)
        self.backup_path: Optional[Path] = None  # ไฟล์สำรองก่อน migrate
        conn = self._conn()
        conn.executescript(META)
        found = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        version = None if found is None else int(found["value"])
        if version == 1:
            self._migrate_v1_to_v2()
        elif version not in (None, SCHEMA_VERSION):
            raise RuntimeError(f"{self.db_path}: schema_version {version} ไม่ตรงกับโค้ด ({SCHEMA_VERSION})")
        conn.executescript(SCHEMA)
        if version is None:
            conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        self.memberships = SqliteMemberships(self)

    # ---------- migrate ----------

    def _backup_before_migrate(self, version: int) -> Path:
        """สำรองทั้งฐานด้วย SQLite online backup API (ได้สำเนาที่สอดคล้องรวม WAL) — มีไฟล์ชื่อนี้อยู่แล้ว = ใช้ชื่อมีเวลาต่อท้าย
        (ไม่เขียนทับสำรองเดิม)"""
        dest = self.db_path.with_name(f"{self.db_path.name}.bak-v{version}")
        if dest.exists():
            stamp = utcnow().replace(":", "").replace("-", "").replace(".", "")
            dest = self.db_path.with_name(f"{self.db_path.name}.bak-v{version}-{stamp}")
        target = sqlite3.connect(dest)
        try:
            self._conn().backup(target)
            target.execute("PRAGMA journal_mode=DELETE")  # ไฟล์สำรองเป็นไฟล์เดียวจบ (ไม่ต้องมี -wal คู่)
        finally:
            target.close()
        return dest

    def _migrate_v1_to_v2(self) -> None:
        """v1 -> v2: สร้างตาราง files ใหม่ (UNIQUE เดิมเป็นข้อกำหนดของตาราง ถอดใน SQLite ไม่ได้) ตามขั้นตอนของ SQLite
        (ปิด foreign_keys ระหว่างย้าย แล้วตรวจ ``foreign_key_check`` ก่อน commit) — ทุกแถวย้ายครบในก้อนเดียว ล้ม = rollback"""
        self.backup_path = self._backup_before_migrate(1)
        conn = self._conn()
        cols = ", ".join(_FILE_COLS)
        conn.execute("PRAGMA foreign_keys=OFF")  # ต้องตั้งนอก transaction
        try:
            with self.transaction():
                before = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
                conn.execute("DROP TABLE IF EXISTS files_v2")
                conn.execute(FILES_TABLE.format(name="files_v2"))
                conn.execute(f"INSERT INTO files_v2 ({cols}) SELECT {cols} FROM files")
                conn.execute("DROP TABLE files")
                conn.execute("ALTER TABLE files_v2 RENAME TO files")
                for stmt in FILE_INDEXES:  # executescript จะ COMMIT เอง — ใช้ execute ทีละคำสั่งใน transaction
                    conn.execute(stmt)
                after = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
                bad = conn.execute("PRAGMA foreign_key_check").fetchall()
                if bad or after != before:
                    raise RuntimeError(f"migrate v1 -> v2 ไม่ครบ: files {before} -> {after}, foreign key เสีย {len(bad)} แถว")
                conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION),))
        finally:
            conn.execute("PRAGMA foreign_keys=ON")
        self.migrated_from = 1
        log.warning("migrate %s: schema v1 -> v2 (สำรองไว้ที่ %s)", self.db_path, self.backup_path)

    # ---------- connection / transaction ----------

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
            self._local.depth = 0
            with self._all_lock:
                self._all.append(conn)
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """``BEGIN IMMEDIATE`` ... ``COMMIT`` (error = ``ROLLBACK``) ซ้อนกันได้ใน thread เดียว"""
        conn = self._conn()
        if self._local.depth:
            self._local.depth += 1
            try:
                yield conn
            finally:
                self._local.depth -= 1
            return
        conn.execute("BEGIN IMMEDIATE")
        self._local.depth = 1
        try:
            yield conn
        except BaseException:
            self._local.depth = 0
            conn.execute("ROLLBACK")
            raise
        self._local.depth = 0
        conn.execute("COMMIT")

    def close(self) -> None:
        with self._all_lock:
            for conn in self._all:
                try:
                    conn.close()
                except Exception:
                    pass
            self._all.clear()
        self._local = threading.local()

    # ---------- กล่อง / FileAcl (AclPersistence) ----------

    def save_box(self, box: Box) -> None:
        now = utcnow()
        self._conn().execute(
            "INSERT INTO boxes (box_id, name, visibility, departments, min_security_level, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(box_id) DO UPDATE SET name = excluded.name, "
            "departments = excluded.departments, min_security_level = excluded.min_security_level, "
            "updated_at = excluded.updated_at",  # visibility ไม่อยู่ใน UPDATE: เปลี่ยนไม่ได้ (AclStore ปฏิเสธก่อนแล้ว)
            (box.box_id, box.name, box.visibility.value, _dump_departments(box.departments),
             box.min_security_level, now, now),
        )

    def load_boxes(self) -> list[Box]:
        rows = self._conn().execute("SELECT * FROM boxes ORDER BY box_id").fetchall()
        return [Box(box_id=r["box_id"], name=r["name"], visibility=Visibility(r["visibility"]),
                    departments=_load_departments(r["departments"]), min_security_level=r["min_security_level"])
                for r in rows]

    def save_file_acl(self, file: FileAcl) -> None:
        self._conn().execute(
            "INSERT INTO file_acls (file_id, box_id, departments, min_security_level) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(file_id) DO UPDATE SET box_id = excluded.box_id, departments = excluded.departments, "
            "min_security_level = excluded.min_security_level",
            (file.file_id, file.box_id, _dump_departments(file.departments), file.min_security_level),
        )

    def delete_file_acl(self, file_id: str) -> None:
        self._conn().execute("DELETE FROM file_acls WHERE file_id = ?", (file_id,))

    def load_file_acls(self) -> list[FileAcl]:
        rows = self._conn().execute("SELECT * FROM file_acls ORDER BY file_id").fetchall()
        return [FileAcl(file_id=r["file_id"], box_id=r["box_id"], departments=_load_departments(r["departments"]),
                        min_security_level=r["min_security_level"]) for r in rows]

    # ---------- ทะเบียนไฟล์ ----------

    def insert_file(self, rec: FileRecord) -> FileRecord:
        now = utcnow()
        rec.created_at = rec.created_at or now
        rec.updated_at = rec.updated_at or now
        cols = ", ".join(_FILE_COLS)
        self._conn().execute(f"INSERT INTO files ({cols}) VALUES ({', '.join('?' * len(_FILE_COLS))})",
                             [getattr(rec, c) for c in _FILE_COLS])
        return rec

    def get_file(self, file_id: str) -> Optional[FileRecord]:
        row = self._conn().execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
        return None if row is None else FileRecord(**{k: row[k] for k in row.keys()})

    def find_file(self, box_id: str, content_hash: str) -> Optional[FileRecord]:
        row = self._conn().execute(f"SELECT * FROM files WHERE box_id = ? AND content_hash = ? "
                                   f"AND status NOT IN {_HIDDEN_SQL}", (box_id, content_hash)).fetchone()
        return None if row is None else FileRecord(**{k: row[k] for k in row.keys()})

    def update_file(self, file_id: str, **values: Any) -> None:
        values["updated_at"] = utcnow()
        bad = set(values) - set(_FILE_COLS)
        if bad:
            raise KeyError(f"ไม่มีคอลัมน์ {sorted(bad)} ในตาราง files")
        sets = ", ".join(f"{k} = ?" for k in values)
        self._conn().execute(f"UPDATE files SET {sets} WHERE file_id = ?", [*values.values(), file_id])

    def files_page(self, box_id: str, after: Optional[tuple[str, str]], limit: int) -> list[FileRecord]:
        """ไฟล์ในกล่องเรียงตาม (created_at, file_id) หลังตำแหน่ง ``after`` (ไม่กรองสิทธิ์ — ผู้เรียกกรองเอง;
        ไม่รวมไฟล์ที่ถูกลบ/กำลังลบ)"""
        if after is None:
            rows = self._conn().execute(
                f"SELECT * FROM files WHERE box_id = ? AND status NOT IN {_HIDDEN_SQL} "
                "ORDER BY created_at, file_id LIMIT ?", (box_id, limit)).fetchall()
        else:
            rows = self._conn().execute(
                f"SELECT * FROM files WHERE box_id = ? AND status NOT IN {_HIDDEN_SQL} "
                "AND (created_at, file_id) > (?, ?) ORDER BY created_at, file_id LIMIT ?",
                (box_id, after[0], after[1], limit)).fetchall()
        return [FileRecord(**{k: r[k] for k in r.keys()}) for r in rows]

    def files_in_box(self, box_id: str) -> list[FileRecord]:
        rows = self._conn().execute(f"SELECT * FROM files WHERE box_id = ? AND status NOT IN {_HIDDEN_SQL} "
                                    "ORDER BY created_at, file_id", (box_id,)).fetchall()
        return [FileRecord(**{k: r[k] for k in r.keys()}) for r in rows]

    def files_with_status(self, status: str) -> list[FileRecord]:
        rows = self._conn().execute("SELECT * FROM files WHERE status = ? ORDER BY updated_at, file_id",
                                    (status,)).fetchall()
        return [FileRecord(**{k: r[k] for k in r.keys()}) for r in rows]

    def live_files_with_hash(self, content_hash: str, exclude_file_id: Optional[str] = None) -> list[FileRecord]:
        """ไฟล์ที่ยังอยู่ (ทุกกล่อง ไม่นับ deleting/deleted) ที่มี content_hash นี้ — ตัดสินว่าแคช OCR ของ hash ยังมีคนใช้ไหม"""
        rows = self._conn().execute(
            f"SELECT * FROM files WHERE content_hash = ? AND status NOT IN {_HIDDEN_SQL} AND file_id != ? "
            "ORDER BY file_id", (content_hash, exclude_file_id or "")).fetchall()
        return [FileRecord(**{k: r[k] for k in r.keys()}) for r in rows]

    def original_path(self, file_id: str) -> Path:
        return self.files_dir / file_id

    def document_path(self, file_id: str) -> Path:
        return self.documents_dir / f"{file_id}.json"

    # ---------- jobs ----------

    def insert_job(self, rec: JobRecord) -> JobRecord:
        now = utcnow()
        rec.created_at = rec.created_at or now
        rec.updated_at = rec.updated_at or now
        attr = {v: k for k, v in _JSON_COLS.items()}
        values = [json.dumps(getattr(rec, attr[c]), ensure_ascii=False) if c in attr else getattr(rec, c)
                  for c in _JOB_COLS]
        self._conn().execute(f"INSERT INTO jobs ({', '.join(_JOB_COLS)}) VALUES ({', '.join('?' * len(_JOB_COLS))})",
                             values)
        return rec

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        row = self._conn().execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return None if row is None else _job_from_row(row)

    def latest_job_for_file(self, file_id: str) -> Optional[JobRecord]:
        row = self._conn().execute("SELECT * FROM jobs WHERE file_id = ? ORDER BY rowid DESC LIMIT 1",
                                   (file_id,)).fetchone()
        return None if row is None else _job_from_row(row)

    def update_job(self, job_id: str, **values: Any) -> None:
        for name, col in _JSON_COLS.items():
            if name in values:
                values[col] = json.dumps(values.pop(name), ensure_ascii=False)
        values["updated_at"] = utcnow()
        bad = set(values) - set(_JOB_COLS)
        if bad:
            raise KeyError(f"ไม่มีคอลัมน์ {sorted(bad)} ในตาราง jobs")
        sets = ", ".join(f"{k} = ?" for k in values)
        self._conn().execute(f"UPDATE jobs SET {sets} WHERE job_id = ?", [*values.values(), job_id])

    def claim_next_job(self) -> Optional[JobRecord]:
        """job ที่ queued เก่าสุด (FIFO ตามลำดับที่เข้าคิว) -> เปลี่ยนเป็น extracting ในก้อนเดียวกัน (ไม่มีใครหยิบซ้ำ)

        job queued ของไฟล์ที่ถูกลบ/กำลังลบ (เช่น ส่งซ้ำแทรกเข้ามาระหว่างลบ) -> ``deleting`` ไม่ถูกหยิบ
        (ไฟล์เหล่านั้น ``FileDeleter.finish_pending`` ลบต่อจนครบ — worker เรียกตอนว่าง)"""
        with self.transaction() as conn:
            now = utcnow()
            conn.execute(f"UPDATE jobs SET status = 'deleting', updated_at = ? WHERE status = 'queued' AND file_id IN "
                         f"(SELECT file_id FROM files WHERE status IN {_HIDDEN_SQL})", (now,))
            row = conn.execute("SELECT job_id FROM jobs WHERE status = 'queued' ORDER BY rowid LIMIT 1").fetchone()
            if row is None:
                return None
            conn.execute("UPDATE jobs SET status = 'extracting', started_at = ?, updated_at = ? WHERE job_id = ?",
                         (now, now, row["job_id"]))
            job = self.get_job(row["job_id"])
            conn.execute("UPDATE files SET status = 'extracting', updated_at = ? WHERE file_id = ?", (now, job.file_id))
            return job

    def requeue_interrupted(self) -> list[str]:
        """job ที่ค้างกลางทาง (process ตายระหว่างทำ) -> queued อีกครั้ง (ไฟล์ด้วย) คืน job_id ที่ถูกคืนคิว"""
        marks = ", ".join("?" * len(IN_PROGRESS))
        with self.transaction() as conn:
            rows = conn.execute(f"SELECT job_id, file_id FROM jobs WHERE status IN ({marks}) ORDER BY rowid",
                                IN_PROGRESS).fetchall()
            now = utcnow()
            for r in rows:
                conn.execute("UPDATE jobs SET status = 'queued', pages_done = 0, updated_at = ? WHERE job_id = ?",
                             (now, r["job_id"]))
                conn.execute("UPDATE files SET status = 'queued', updated_at = ? WHERE file_id = ?", (now, r["file_id"]))
            return [r["job_id"] for r in rows]

    def count_jobs(self, status: str) -> int:
        return self._conn().execute("SELECT COUNT(*) FROM jobs WHERE status = ?", (status,)).fetchone()[0]


class SqliteMemberships:
    """``MembershipBackend`` ตัวจริงบน SQLite (ตารางเดียวกับกล่อง) — ``transaction`` = BEGIN IMMEDIATE ของฐานเดียวกัน"""

    def __init__(self, storage: SqliteStorage) -> None:
        self._s = storage

    @contextmanager
    def transaction(self, box_id: str) -> Iterator[None]:
        with self._s.transaction():
            yield

    def get_role(self, box_id: str, user_id: str) -> Optional[BoxRole]:
        row = self._s._conn().execute("SELECT role FROM members WHERE box_id = ? AND user_id = ?",
                                      (box_id, user_id)).fetchone()
        return None if row is None else BoxRole(row["role"])

    def set_role(self, box_id: str, user_id: str, role: BoxRole) -> None:
        self._s._conn().execute(
            "INSERT INTO members (box_id, user_id, role, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(box_id, user_id) DO UPDATE SET role = excluded.role, updated_at = excluded.updated_at",
            (box_id, user_id, BoxRole(role).value, utcnow()),
        )

    def remove(self, box_id: str, user_id: str) -> None:
        self._s._conn().execute("DELETE FROM members WHERE box_id = ? AND user_id = ?", (box_id, user_id))

    def members(self, box_id: str) -> dict[str, BoxRole]:
        rows = self._s._conn().execute("SELECT user_id, role FROM members WHERE box_id = ?", (box_id,)).fetchall()
        return {r["user_id"]: BoxRole(r["role"]) for r in rows}

    def boxes_of(self, user_id: str) -> dict[str, BoxRole]:
        rows = self._s._conn().execute("SELECT box_id, role FROM members WHERE user_id = ?", (user_id,)).fetchall()
        return {r["box_id"]: BoxRole(r["role"]) for r in rows}
