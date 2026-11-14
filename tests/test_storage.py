"""ที่เก็บถาวร SQLite (app/storage.py) + AclStore ที่เขียนผ่าน — โหลดคืนหลังเปิดใหม่, transaction ของสมาชิก (2 thread),
คิวงาน FIFO และการคืนคิวงานที่ค้างกลางทาง"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import nullcontext

import pytest

from app.access import (
    AclError,
    AclLastOwnerError,
    AclStore,
    Box,
    BoxRole,
    FileAcl,
    InMemoryMemberships,
    Principal,
    Visibility,
)
from app.storage import FileRecord, JobRecord, SqliteMemberships, SqliteStorage


@pytest.fixture
def storage(tmp_path):
    s = SqliteStorage(tmp_path / "data")
    yield s
    s.close()


def acl_on(storage: SqliteStorage, memberships=None) -> AclStore:
    return AclStore(memberships=memberships or storage.memberships, persistence=storage)


def test_layout_and_schema(storage, tmp_path):
    assert (tmp_path / "data" / "rag.db").is_file()
    assert storage.files_dir.is_dir() and storage.documents_dir.is_dir()
    tables = {r[0] for r in storage._conn().execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"meta", "boxes", "members", "file_acls", "files", "jobs"} <= tables
    assert storage._conn().execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_acl_store_survives_reopen(tmp_path):
    s1 = SqliteStorage(tmp_path / "d")
    acl = acl_on(s1)
    acl.create_box(Box(box_id="b", visibility=Visibility.PRIVATE, departments=frozenset({"hr", "it"}),
                       min_security_level=2, name="กล่องตัวอย่าง"), "own")
    acl.set_member("b", "v", BoxRole.VIEWER)
    acl.register_file(FileAcl(file_id="f", box_id="b", departments=frozenset({"hr"}), min_security_level=3))
    acl.register_file(FileAcl(file_id="g", box_id="b"))
    acl.remove_file("g")
    s1.close()

    s2 = SqliteStorage(tmp_path / "d")
    again = acl_on(s2)
    box = again.get_box("b")
    assert (box.name, box.visibility, box.departments, box.min_security_level) == (
        "กล่องตัวอย่าง", Visibility.PRIVATE, frozenset({"hr", "it"}), 2)
    assert again.members("b") == {"own": BoxRole.OWNER, "v": BoxRole.VIEWER}
    eff = again.get_effective("f")
    assert (eff.departments, eff.min_security_level) == (frozenset({"hr"}), 3)
    assert again.get_effective("g") is None
    assert again.can_access(again.principal_for("v", {"hr"}, 3), "f")
    with pytest.raises(AclError):  # visibility เปลี่ยนไม่ได้ แม้หลังโหลดใหม่
        again.register_box(Box(box_id="b", visibility=Visibility.PUBLIC))
    s2.close()


def test_departments_none_and_empty_are_distinct_after_reload(tmp_path):
    s1 = SqliteStorage(tmp_path / "d")
    acl = acl_on(s1)
    acl.create_box(Box(box_id="open", visibility=Visibility.PUBLIC, departments=None), "o")
    acl.create_box(Box(box_id="closed", visibility=Visibility.PUBLIC, departments=frozenset()), "o")
    s1.close()
    s2 = SqliteStorage(tmp_path / "d")
    again = acl_on(s2)
    assert again.get_box("open").departments is None and again.get_box("closed").departments == frozenset()
    s2.close()


def test_create_box_is_atomic_with_its_owner(storage):
    class Broken(SqliteMemberships):
        def set_role(self, box_id, user_id, role):
            raise sqlite3.OperationalError("disk I/O error")

    acl = acl_on(storage, Broken(storage))
    with pytest.raises(sqlite3.OperationalError):
        acl.create_box(Box(box_id="b", visibility=Visibility.PUBLIC), "own")
    assert acl.get_box("b") is None and storage.load_boxes() == []  # ไม่มีกล่องที่ไม่มี owner ค้าง


# ---------- transaction ของสมาชิก: 2 thread พร้อมกัน ----------


class SlowMembers(SqliteMemberships):
    """อ่านรายชื่อสมาชิกแล้วหน่วงก่อนคืน — เปิดช่องให้อีก thread แทรกระหว่าง "ตรวจ owner" กับ "เขียน" """

    def members(self, box_id):
        out = super().members(box_id)
        time.sleep(0.2)
        return out


class SlowInMemory(InMemoryMemberships):
    def members(self, box_id):
        out = super().members(box_id)
        time.sleep(0.2)
        return out


def race(acl: AclStore, actions) -> list:
    barrier = threading.Barrier(len(actions))
    results: list = [None] * len(actions)

    def run(i, fn):
        barrier.wait()
        try:
            fn()
            results[i] = "ok"
        except Exception as e:  # noqa: BLE001 — เก็บชนิดไว้ตรวจ
            results[i] = type(e).__name__

    threads = [threading.Thread(target=run, args=(i, fn)) for i, fn in enumerate(actions)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    return results


def two_owner_box(acl: AclStore) -> None:
    acl.create_box(Box(box_id="b", visibility=Visibility.PRIVATE), "A")
    acl.set_member("b", "B", BoxRole.OWNER)


@pytest.mark.parametrize("backend", ["sqlite", "memory"])
def test_concurrent_owner_removals_cannot_orphan_the_box(storage, backend):
    members = SlowMembers(storage) if backend == "sqlite" else SlowInMemory()
    acl = acl_on(storage, members) if backend == "sqlite" else AclStore(memberships=members)
    two_owner_box(acl)
    results = race(acl, [lambda: acl.remove_member("b", "A"), lambda: acl.remove_member("b", "B")])
    assert sorted(results) == ["AclLastOwnerError", "ok"]
    assert list(acl.members("b").values()) == [BoxRole.OWNER]


def test_concurrent_demote_and_remove_cannot_orphan_the_box(storage):
    acl = acl_on(storage, SlowMembers(storage))
    two_owner_box(acl)
    results = race(acl, [lambda: acl.set_member("b", "A", BoxRole.VIEWER), lambda: acl.remove_member("b", "B")])
    assert sorted(results) == ["AclLastOwnerError", "ok"]
    assert BoxRole.OWNER in acl.members("b").values()


def test_race_test_detects_the_bug_without_a_transaction(storage, monkeypatch):
    """ตัวควบคุม: ถ้าไม่มี transaction ครอบ test เดียวกันต้องเห็นกล่องไม่มี owner (แปลว่า test ด้านบนจับปัญหานี้ได้จริง)"""
    acl = acl_on(storage, SlowMembers(storage))
    two_owner_box(acl)
    monkeypatch.setattr(acl, "_member_tx", lambda box_id: nullcontext())
    results = race(acl, [lambda: acl.remove_member("b", "A"), lambda: acl.remove_member("b", "B")])
    assert results == ["ok", "ok"] and acl.members("b") == {}


def test_last_owner_rules_hold_on_sqlite(storage):
    acl = acl_on(storage)
    acl.create_box(Box(box_id="b", visibility=Visibility.PUBLIC), "A")
    with pytest.raises(AclLastOwnerError):
        acl.remove_member("b", "A")
    with pytest.raises(AclLastOwnerError):
        acl.set_member("b", "A", BoxRole.ADMIN)


def test_removed_member_is_denied_from_storage_immediately(storage):
    acl = acl_on(storage)
    acl.create_box(Box(box_id="b", visibility=Visibility.PUBLIC), "A")
    acl.register_file(FileAcl(file_id="f", box_id="b"))
    acl.set_member("b", "u", BoxRole.VIEWER)
    old = acl.principal_for("u")
    assert acl.can_access(old, "f")
    acl.remove_member("b", "u")
    assert not acl.can_access(old, "f")
    assert not acl.can_access(Principal(user_id="u", box_ids=frozenset({"b"})), "f")


# ---------- ทะเบียนไฟล์ / jobs ----------


def seed_file(storage: SqliteStorage, acl: AclStore, file_id: str, content_hash: str = "h") -> None:
    acl.register_file(FileAcl(file_id=file_id, box_id="b"))
    storage.insert_file(FileRecord(file_id=file_id, box_id="b", filename=f"{file_id}.docx", content_hash=content_hash,
                                   size_bytes=1, status="queued", submitted_by="A"))


def test_same_content_in_same_box_is_rejected_by_the_database(storage):
    acl = acl_on(storage)
    acl.create_box(Box(box_id="b", visibility=Visibility.PUBLIC), "A")
    seed_file(storage, acl, "f1", "same")
    with pytest.raises(sqlite3.IntegrityError):
        seed_file(storage, acl, "f2", "same")


def test_jobs_are_claimed_fifo_and_only_once(storage):
    acl = acl_on(storage)
    acl.create_box(Box(box_id="b", visibility=Visibility.PUBLIC), "A")
    for i in range(6):
        seed_file(storage, acl, f"f{i}", f"h{i}")
        storage.insert_job(JobRecord(job_id=f"j{i}", file_id=f"f{i}", box_id="b", submitted_by="A", status="queued"))
    claimed: list[str] = []
    lock = threading.Lock()

    def worker():
        while True:
            job = storage.claim_next_job()
            if job is None:
                return
            with lock:
                claimed.append(job.job_id)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert sorted(claimed) == [f"j{i}" for i in range(6)]  # ไม่มีงานไหนถูกหยิบสองครั้ง
    assert storage.count_jobs("extracting") == 6

    storage2 = SqliteStorage(storage.data_dir)  # ลำดับ FIFO ด้วย thread เดียว
    for i in range(3):
        storage2.update_job(f"j{i}", status="queued")
    assert [storage2.claim_next_job().job_id for _ in range(3)] == ["j0", "j1", "j2"]
    storage2.close()


def test_requeue_interrupted_jobs(storage):
    acl = acl_on(storage)
    acl.create_box(Box(box_id="b", visibility=Visibility.PUBLIC), "A")
    for i, status in enumerate(["extracting", "ocr", "indexing", "done", "failed", "queued"]):
        seed_file(storage, acl, f"f{i}", f"h{i}")
        storage.insert_job(JobRecord(job_id=f"j{i}", file_id=f"f{i}", box_id="b", submitted_by="A", status=status,
                                     pages_done=3))
        storage.update_file(f"f{i}", status=status)
    assert storage.requeue_interrupted() == ["j0", "j1", "j2"]
    assert [storage.get_job(f"j{i}").status for i in range(6)] == ["queued"] * 3 + ["done", "failed", "queued"]
    assert storage.get_job("j1").pages_done == 0 and storage.get_file("f1").status == "queued"


def test_job_json_fields_roundtrip(storage):
    acl = acl_on(storage)
    acl.create_box(Box(box_id="b", visibility=Visibility.PUBLIC), "A")
    seed_file(storage, acl, "f")
    storage.insert_job(JobRecord(job_id="j", file_id="f", box_id="b", submitted_by="A", status="queued"))
    gaps = [{"unit_index": 1, "label": "หน้า 2", "status": "failed", "error_code": None, "reason": "เหตุผล"}]
    storage.update_job("j", gaps=gaps, units={"ok": 1, "needs_ocr": 0, "failed": 1, "rejected": 0})
    job = storage.get_job("j")
    assert job.gaps == gaps and job.units["failed"] == 1
    with pytest.raises(KeyError):
        storage.update_job("j", nope=1)
