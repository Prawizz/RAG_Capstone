"""migrate SQLite schema v1 -> v2 (UNIQUE(box_id, content_hash) -> partial unique index ที่ไม่นับไฟล์ deleting/deleted)

- ฐาน v1 สังเคราะห์ (schema v1 ตามที่ใช้ก่อนขั้นลบไฟล์) -> เปิดด้วย ``SqliteStorage`` -> ข้อมูลครบ + สำรอง rag.db.bak-v1
- สำเนาของ ``data/rag.db`` จริง (ข้อมูล demo): คัดลอกแบบอ่านอย่างเดียวด้วย SQLite backup API ไปที่ tmp ก่อน **ไม่แตะตัวจริง**
  (ถ้าตัวจริงถูก migrate ไปแล้ว ใช้ ``data/rag.db.bak-v1`` ซึ่งเป็น v1 ที่สำรองไว้ตอน migrate; ไม่มีทั้งคู่ = skip)
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from app.access import AclStore
from app.storage import FileRecord, SqliteStorage

REAL_DATA = Path(__file__).resolve().parent.parent / "data"

# schema v1 (ก่อนขั้นลบไฟล์) — ตาราง files มี UNIQUE (box_id, content_hash) เป็นข้อกำหนดของตาราง
SCHEMA_V1 = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE boxes (box_id TEXT PRIMARY KEY, name TEXT, visibility TEXT NOT NULL CHECK (visibility IN ('public', 'private')),
    departments TEXT, min_security_level INTEGER NOT NULL CHECK (min_security_level >= 0),
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE members (box_id TEXT NOT NULL REFERENCES boxes(box_id), user_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('owner', 'admin', 'viewer')), updated_at TEXT NOT NULL, PRIMARY KEY (box_id, user_id));
CREATE INDEX members_by_user ON members(user_id);
CREATE TABLE file_acls (file_id TEXT PRIMARY KEY, box_id TEXT NOT NULL REFERENCES boxes(box_id), departments TEXT,
    min_security_level INTEGER);
CREATE TABLE files (file_id TEXT PRIMARY KEY, box_id TEXT NOT NULL REFERENCES boxes(box_id), filename TEXT NOT NULL,
    content_hash TEXT NOT NULL, size_bytes INTEGER NOT NULL, detected_type TEXT, status TEXT NOT NULL,
    units_total INTEGER NOT NULL DEFAULT 0, units_ok INTEGER NOT NULL DEFAULT 0, units_needs_ocr INTEGER NOT NULL DEFAULT 0,
    units_failed INTEGER NOT NULL DEFAULT 0, units_rejected INTEGER NOT NULL DEFAULT 0, gaps INTEGER NOT NULL DEFAULT 0,
    submitted_by TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, ingested_at TEXT,
    UNIQUE (box_id, content_hash));
CREATE INDEX files_by_box ON files(box_id, created_at, file_id);
CREATE TABLE jobs (job_id TEXT PRIMARY KEY, file_id TEXT NOT NULL REFERENCES files(file_id), box_id TEXT NOT NULL,
    submitted_by TEXT NOT NULL, status TEXT NOT NULL, pages_total INTEGER, pages_done INTEGER NOT NULL DEFAULT 0,
    document_status TEXT, gaps_json TEXT NOT NULL DEFAULT '[]', units_json TEXT NOT NULL DEFAULT '{}', error_code TEXT,
    error_message TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, started_at TEXT, finished_at TEXT);
CREATE INDEX jobs_by_status ON jobs(status);
CREATE INDEX jobs_by_file ON jobs(file_id);
INSERT INTO meta VALUES ('schema_version', '1');
"""

TABLES = ("boxes", "members", "file_acls", "files", "jobs")


def counts(db: Path) -> dict[str, int]:
    conn = sqlite3.connect(db)
    try:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
    finally:
        conn.close()


def schema_version(db: Path) -> int:
    conn = sqlite3.connect(db)
    try:
        return int(conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0])
    finally:
        conn.close()


def make_v1(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True)
    db = data_dir / "rag.db"
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA_V1)
    t = "2026-11-01T00:00:00.000000Z"
    conn.execute("INSERT INTO boxes VALUES ('b', 'กล่อง', 'public', NULL, 0, ?, ?)", (t, t))
    conn.execute("INSERT INTO members VALUES ('b', 'owner1', 'owner', ?)", (t,))
    for i in range(3):
        conn.execute("INSERT INTO file_acls VALUES (?, 'b', NULL, NULL)", (f"f{i}",))
        conn.execute("INSERT INTO files (file_id, box_id, filename, content_hash, size_bytes, status, submitted_by, "
                     "created_at, updated_at) VALUES (?, 'b', ?, ?, 1, 'done', 'owner1', ?, ?)",
                     (f"f{i}", f"ไฟล์ {i}.pdf", f"h{i}", t, t))
        conn.execute("INSERT INTO jobs (job_id, file_id, box_id, submitted_by, status, created_at, updated_at) "
                     "VALUES (?, ?, 'b', 'owner1', 'done', ?, ?)", (f"j{i}", f"f{i}", t, t))
    conn.commit()
    conn.close()
    return db


def test_v1_database_is_migrated_with_a_backup_and_nothing_lost(tmp_path):
    db = make_v1(tmp_path / "data")
    before = counts(db)
    s = SqliteStorage(tmp_path / "data")
    try:
        assert s.migrated_from == 1 and s.backup_path == db.with_name("rag.db.bak-v1")
        assert schema_version(s.backup_path) == 1 and counts(s.backup_path) == before
        assert schema_version(db) == 2 and counts(db) == before
        assert s._conn().execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert s._conn().execute("PRAGMA foreign_key_check").fetchall() == []
        assert s.get_file("f1").filename == "ไฟล์ 1.pdf" and s.get_job("j1").file_id == "f1"
        # partial unique index: ไฟล์ที่ยังอยู่ยังซ้ำไม่ได้ / tombstone ไม่กันอัปโหลดใหม่
        with pytest.raises(sqlite3.IntegrityError):
            s.insert_file(FileRecord(file_id="dup", box_id="b", filename="x", content_hash="h0", size_bytes=1,
                                     status="queued", submitted_by="owner1"))
        s.update_file("f0", status="deleted")
        s.insert_file(FileRecord(file_id="again", box_id="b", filename="x", content_hash="h0", size_bytes=1,
                                 status="queued", submitted_by="owner1"))
        assert s.find_file("b", "h0").file_id == "again"
    finally:
        s.close()
    again = SqliteStorage(tmp_path / "data")  # เปิดซ้ำ: ไม่ migrate ซ้ำ ไม่สำรองซ้ำ
    try:
        assert again.migrated_from is None
        assert sorted(p.name for p in (tmp_path / "data").glob("rag.db.bak-*")) == ["rag.db.bak-v1"]
    finally:
        again.close()


def test_existing_backup_is_never_overwritten(tmp_path):
    db = make_v1(tmp_path / "data")
    old = db.with_name("rag.db.bak-v1")
    old.write_bytes(b"older backup")
    s = SqliteStorage(tmp_path / "data")
    try:
        assert old.read_bytes() == b"older backup"
        assert s.backup_path != old and s.backup_path.name.startswith("rag.db.bak-v1-")
        assert schema_version(s.backup_path) == 1
    finally:
        s.close()


def _real_v1_source(tmp: Path) -> Path | None:
    """ตัวจริงที่เป็น v1: rag.db (ยังไม่ migrate) หรือ rag.db.bak-v1 — ตรวจรุ่นจากสำเนาใน tmp (ไม่เปิดตัวจริง)"""
    for candidate in (REAL_DATA / "rag.db", REAL_DATA / "rag.db.bak-v1"):
        if not candidate.is_file():
            continue
        probe = tmp / f"probe_{candidate.name}"
        probe.mkdir()
        for suffix in ("", "-wal"):
            src = candidate.with_name(candidate.name + suffix)
            if src.is_file():
                shutil.copyfile(src, probe / ("rag.db" + suffix))
        try:
            if schema_version(probe / "rag.db") == 1:
                return candidate
        except sqlite3.Error:
            continue
    return None


def test_migrate_a_copy_of_the_real_demo_database(tmp_path):
    source = _real_v1_source(tmp_path)
    if source is None:
        pytest.skip("ไม่มี data/rag.db (หรือ rag.db.bak-v1) ที่เป็น schema v1 ในเครื่องนี้")
    mtime_before = source.stat().st_mtime
    # อ่านไบต์ของตัวจริงอย่างเดียว (ไม่เปิดด้วย SQLite — เปิดแม้แบบ read-only ก็ยังเขียน -shm) แล้วให้ SQLite
    # ประกอบสำเนาจาก db + wal ที่ tmp (API ต้องหยุดอยู่ ซึ่งเป็นสภาพตอนรัน test)
    raw = tmp_path / "raw"
    raw.mkdir()
    for suffix in ("", "-wal"):
        src = source.with_name(source.name + suffix)
        if src.is_file():
            shutil.copyfile(src, raw / ("rag.db" + suffix))
    target = tmp_path / "data"
    target.mkdir()
    src_conn = sqlite3.connect(raw / "rag.db")
    dst_conn = sqlite3.connect(target / "rag.db")
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()
    before = counts(target / "rag.db")
    assert before["files"] > 0 and before["members"] > 0 and before["file_acls"] > 0  # มีข้อมูล demo จริง

    s = SqliteStorage(target)
    try:
        assert s.migrated_from == 1 and counts(target / "rag.db") == before
        assert counts(s.backup_path) == before
        acl = AclStore(memberships=s.memberships, persistence=s)  # FileAcl/กล่องโหลดได้ครบเหมือนเดิม
        assert len(acl._files) == before["file_acls"] and len(acl._boxes) == before["boxes"]
        boxes = {r[0] for r in s._conn().execute("SELECT DISTINCT box_id FROM files")}
        assert sum(len(s.files_in_box(b)) for b in boxes) == before["files"]  # ยังไม่มีไฟล์ถูกลบ: เห็นครบ
    finally:
        s.close()
    assert source.stat().st_mtime == mtime_before  # ตัวจริงไม่ถูกแตะ
