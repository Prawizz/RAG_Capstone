"""DELETE /v1/files/{file_id} + การลบต่อจนครบหลัง crash + การชนกับ worker (``app/file_deleter.py``)

ของจริงทุกชั้นผ่าน TestClient ยกเว้น OCR ปลอม (FakeBackend), hash embedding, Qdrant ในกระบวนการ — ดู ``Env`` ใน test_ingest_api
แคช OCR ของ test อยู่ใน tmp (conftest ตั้ง OCR_CACHE_DIR) — ไม่แตะ data/ocr_cache จริง
"""

from __future__ import annotations

import logging
import random
import threading
import time

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app import runtime
from app.extraction.ocr_cache import cache_dir
from app.file_deleter import ocr_cache_entries
from app.vectorstore import VectorStore
from tests.test_ingest_api import ADMIN, THAI_LONG, Env, docx_bytes, hdr, pdf_bytes, same_error


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.services.close()
    runtime.reset_gpu_state()


def delete(e: Env, file_id: str, user: str = "sysadmin", **h):
    return e.client.delete(f"/v1/files/{file_id}", headers={**hdr(user), **h})


def retrieve(e: Env, box_id: str = "b1", user: str = "sysadmin", query: str = THAI_LONG[:60]):
    r = e.client.post("/v1/retrieve", headers=hdr(user), json={"query": query, "box_id": box_id, "top_k": 10})
    assert r.status_code == 200, r.text
    return r.json()


def ids_in(body: dict) -> set[str]:
    return {r["file_id"] for r in body["results"]}


def ingest_done(e: Env, data: bytes, box_id: str = "b1", name: str = "เอกสาร.docx") -> dict:
    r = e.ingest(data, name=name, box_id=box_id)
    assert r.status_code == 202, r.text
    e.run()
    return r.json()


def db_row(e: Env, sql: str, *args):
    return e.services.storage._conn().execute(sql, args).fetchone()


# ---------- ลบสำเร็จ ----------


def test_delete_removes_everything_and_hides_the_file_immediately(env):
    env.box()
    keep = ingest_done(env, docx_bytes("ข้อความเอกสารที่ไม่ถูกลบ " * 20, "อีกหัวข้อ"))
    gone = ingest_done(env, docx_bytes())
    fid = gone["file_id"]
    storage = env.services.storage
    assert {keep["file_id"], fid} <= ids_in(retrieve(env)) and env.points(fid) > 0
    assert storage.original_path(fid).is_file() and storage.document_path(fid).is_file()

    r = delete(env, fid)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["file_id"] == fid and body["box_id"] == "b1" and body["status"] == "deleted"
    assert body["points_deleted"] > 0 and body["original_deleted"] and body["document_deleted"]
    assert body["cancelled_jobs"] == 0 and "ocr_cache" not in body and "ocr_cache_entries" not in body
    assert body["trace_id"] == r.headers["X-Trace-Id"]

    assert ids_in(retrieve(env)) == {keep["file_id"]}  # ค้นไม่เจอทันที (ไฟล์อื่นยังเจอ)
    assert env.points(fid) == 0 and env.points(keep["file_id"]) > 0
    assert not storage.original_path(fid).exists() and not storage.document_path(fid).exists()
    files = env.client.get("/v1/sources/b1/files", headers=ADMIN).json()["files"]
    assert [f["file_id"] for f in files] == [keep["file_id"]]
    [src] = env.client.get("/v1/sources", headers=ADMIN).json()["sources"]
    assert src["file_count"] == 1
    assert env.job(gone["job_id"]).status_code == 404  # งานยังอยู่เป็นประวัติ แต่ไม่เปิดทาง API
    # tombstone + ประวัติงาน + FileAcl ถูกลบ
    assert db_row(env, "SELECT status FROM files WHERE file_id = ?", fid)["status"] == "deleted"
    assert db_row(env, "SELECT status FROM jobs WHERE job_id = ?", gone["job_id"])["status"] == "done"
    assert db_row(env, "SELECT COUNT(*) FROM file_acls WHERE file_id = ?", fid)[0] == 0
    assert env.services.acl.get_file_acl(fid) is None


def test_second_delete_is_404_like_a_missing_file(env):
    env.box()
    fid = ingest_done(env, docx_bytes())["file_id"]
    assert delete(env, fid).status_code == 200
    again, missing = delete(env, fid), delete(env, "file_does_not_exist")
    assert again.status_code == 404 and same_error(again, missing)


def test_same_file_can_be_uploaded_again_after_delete(env):
    env.box()
    first = ingest_done(env, docx_bytes())
    assert delete(env, first["file_id"]).status_code == 200
    r = env.ingest(docx_bytes())
    assert r.status_code == 202 and r.json()["duplicate"] is False and r.json()["file_id"] != first["file_id"]
    env.run()
    assert ids_in(retrieve(env)) == {r.json()["file_id"]}


def test_marking_deleting_cuts_the_file_at_the_second_gate_before_points_are_gone(env):
    """ขั้นแรก (deleting) มีผลกับการค้นทันที แม้ point ยังอยู่ใน Qdrant (ด่าน 1 ผ่าน, ด่าน 2 ตัด)"""
    env.box()
    fid = ingest_done(env, docx_bytes())["file_id"]
    env.services.deleter.mark_deleting(fid)
    assert env.points(fid) > 0
    assert retrieve(env)["results"] == []
    assert env.client.get("/v1/sources/b1/files", headers=ADMIN).json()["files"] == []
    assert delete(env, fid).status_code == 404  # กำลังลบ = ไม่เห็นแล้ว


# ---------- แคช OCR ใช้ร่วมข้ามกล่อง ----------


def test_shared_content_in_another_box_keeps_the_ocr_cache_and_stays_searchable(env, caplog):
    """แคชเก็บไว้ / ถูกลบ: ผลอยู่ใน log เท่านั้น — response สองกรณีมีฟิลด์ชุดเดียวกัน (ไม่บอกว่ามีไฟล์เนื้อหาเดียวกันในกล่องอื่น)"""
    env.box("b1")
    env.box("b2")
    scan = pdf_bytes(["", "", "Text layer page " * 20])  # 2 หน้าแรกต้อง OCR
    a = ingest_done(env, scan, box_id="b1", name="สแกน.pdf")
    b = ingest_done(env, scan, box_id="b2", name="สแกน_สำเนา.pdf")
    content_hash = env.services.storage.get_file(a["file_id"]).content_hash
    cached = ocr_cache_entries(cache_dir(), content_hash)
    assert len(cached) >= 2

    with caplog.at_level(logging.INFO, logger="app.api.files"):
        first = delete(env, a["file_id"]).json()
    assert f"file_id={a['file_id']}" in caplog.text and f"ocr_cache=kept_shared ocr_cache_entries={len(cached)}" in caplog.text
    assert all(p.is_file() for p in cached)
    assert ids_in(retrieve(env, "b2")) == {b["file_id"]} and retrieve(env, "b1")["results"] == []

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="app.api.files"):
        second = delete(env, b["file_id"]).json()
    assert f"ocr_cache=deleted ocr_cache_entries={len(cached)}" in caplog.text
    assert set(first) == set(second)  # กรณีแคชถูกเก็บกับถูกลบ: ฟิลด์ชุดเดียวกัน
    assert {k: v for k, v in first.items() if k not in ("file_id", "box_id", "trace_id")} == \
        {k: v for k, v in second.items() if k not in ("file_id", "box_id", "trace_id")}
    assert not any("ocr" in k for k in first)
    assert ocr_cache_entries(cache_dir(), content_hash) == [] and not any(p.exists() for p in cached)


# ---------- สิทธิ์ ----------


def test_no_permission_is_404_exactly_like_a_missing_file(env):
    env.box("b1", departments=["finance", "legal"])
    env.box("other")
    open_file = ingest_done(env, docx_bytes())["file_id"]
    r = env.ingest(docx_bytes("ไฟล์เฉพาะฝ่ายกฎหมาย " * 20, "ลับ"), file_acl={"departments": ["legal"]})
    env.run()
    legal_only = r.json()["file_id"]
    assert env.member("viewer1", "viewer").status_code == 200
    assert env.member("admin1", "admin").status_code == 200
    assert env.member("other_admin", "admin", box_id="other").status_code == 200
    missing = delete(env, "file_does_not_exist")

    cases = [
        delete(env, open_file, "viewer1"),                          # สมาชิกแต่ไม่ใช่ owner/admin
        delete(env, open_file, "stranger"),                         # ไม่เป็นสมาชิก
        delete(env, open_file, "other_admin"),                      # admin ของกล่องอื่น
        delete(env, legal_only, "admin1"),                          # admin แต่ไม่เห็นไฟล์ (แผนก finance)
        delete(env, legal_only, "viewer1"),                         # viewer + ไม่เห็นไฟล์
    ]
    for r in cases:
        assert r.status_code == 404 and same_error(r, missing), r.text
    assert env.points(open_file) > 0 and env.points(legal_only) > 0  # ไม่มีอะไรถูกลบ
    ok = env.client.delete(f"/v1/files/{legal_only}", headers=hdr("admin1", departments="legal"))
    assert ok.status_code == 200


# ---------- งานที่ยังไม่เสร็จ ----------


def test_delete_while_queued_cancels_the_job_and_worker_skips_it(env):
    env.box()
    r = env.ingest(docx_bytes()).json()
    body = delete(env, r["file_id"]).json()
    assert body["cancelled_jobs"] == 1 and body["points_deleted"] == 0 and body["original_deleted"]
    assert body["document_deleted"] is False  # ยังไม่เคย extract
    assert env.run() == []  # worker ไม่ทำงานนั้น
    assert db_row(env, "SELECT status FROM jobs WHERE job_id = ?", r["job_id"])["status"] == "deleted"
    assert env.job(r["job_id"]).status_code == 404


def test_delete_while_in_progress_is_409_and_touches_nothing(env):
    env.box()
    r = env.ingest(docx_bytes()).json()
    job = env.services.storage.claim_next_job()  # worker หยิบไปแล้ว (extracting)
    for status in ("extracting", "ocr", "indexing"):
        env.services.storage.update_job(job.job_id, status=status)
        resp = delete(env, r["file_id"])
        assert resp.status_code == 409 and resp.json()["error_code"] == "conflict", resp.text
        assert status in resp.json()["message"]
    assert env.services.storage.get_file(r["file_id"]).status != "deleting"
    env.services.storage.update_job(job.job_id, status="extracting")
    env.services.worker._process(job)  # worker ทำต่อจนจบตามปกติ
    assert env.job(r["job_id"]).json()["status"] == "done"
    assert delete(env, r["file_id"]).status_code == 200


def test_queued_job_of_a_deleting_file_is_never_claimed(env):
    """ส่งซ้ำแทรกเข้ามาระหว่างลบ (job queued ใหม่ของไฟล์ deleting) -> worker ข้าม แล้วลบต่อจนครบตอนว่าง"""
    env.box()
    first = ingest_done(env, docx_bytes())
    fid = first["file_id"]
    storage = env.services.storage
    env.services.deleter.mark_deleting(fid)
    from app.storage import JobRecord
    storage.insert_job(JobRecord(job_id="job_sneaky", file_id=fid, box_id="b1", submitted_by="sysadmin",
                                 status="queued"))
    assert env.run() == []
    assert storage.get_job("job_sneaky").status == "deleting"
    env.services.worker.run_idle(force=True)  # worker ตอนว่าง -> FileDeleter.finish_pending
    assert storage.get_file(fid).status == "deleted" and env.points(fid) == 0
    assert storage.get_job("job_sneaky").status == "deleted"


def test_race_between_worker_claim_and_delete(env):
    """worker หยิบงาน vs DELETE พร้อมกันหลายรอบ: ไม่มีทางที่ลบสำเร็จแล้วยังเหลือ point / งานถูกทำต่อ
    (หน่วง worker แบบสุ่มมี seed 0-40 ms ให้ได้ทั้งสองลำดับรอบเส้นแบ่ง — ความถูกต้องจริงมาจาก BEGIN IMMEDIATE)"""
    env.box()
    outcomes = set()
    rng = random.Random(7)
    for i in range(16):
        jitter = rng.uniform(0, 0.04)
        r = env.ingest(docx_bytes(f"ข้อความรอบที่ {i} สำหรับทดสอบการชนกัน " * 15, f"หัวข้อ {i}")).json()
        barrier = threading.Barrier(2)
        got: dict = {}

        def work():
            barrier.wait()
            time.sleep(jitter)
            got["job"] = env.services.worker.process_next()

        def remove():
            barrier.wait()
            got["status"] = delete(env, r["file_id"]).status_code

        threads = [threading.Thread(target=work), threading.Thread(target=remove)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        file = env.services.storage.get_file(r["file_id"])
        job = env.services.storage.get_job(r["job_id"])
        if got["status"] == 200:
            assert file.status == "deleted" and env.points(r["file_id"]) == 0
            assert job.status in ("deleted", "done")  # ลบก่อนหยิบ (ยกเลิก) หรือหลังงานจบ
            assert got["job"] is None or job.status == "done"
            outcomes.add("deleted_before" if job.status == "deleted" else "deleted_after")
        else:
            assert got["status"] == 409 and job.status == "done" and file.status == "done"
            assert env.points(r["file_id"]) > 0
            outcomes.add("conflict")
    assert "deleted_before" in outcomes and len(outcomes) >= 2, outcomes


# ---------- ทำต่อหลัง crash ----------


def shared_store_env(tmp_path, client: QdrantClient) -> Env:
    holder: dict = {}

    def factory(acl):
        holder["store"] = VectorStore(client, acl, holder["env"].settings, supports_payload_indexes=False)
        return holder["store"]

    e = Env(tmp_path, store_factory=factory)
    holder["env"] = e
    e.services.stores.get()
    e.store = holder["store"]
    return e


def test_delete_resumes_after_a_crash_on_api_start(tmp_path, monkeypatch):
    client = QdrantClient(":memory:")  # "Qdrant" ตัวเดียวกันข้าม process
    e = shared_store_env(tmp_path, client)
    e.box()
    gone = ingest_done(e, docx_bytes())["file_id"]
    keep = ingest_done(e, docx_bytes("ไฟล์ที่ยังอยู่ " * 30, "คงไว้"))["file_id"]

    def crash(*_a, **_k):
        raise RuntimeError("process ตายกลางทาง")

    monkeypatch.setattr("app.file_deleter.ocr_cache_entries", crash)  # ตายหลังลบ point / ต้นฉบับ / ผล extract
    with pytest.raises(RuntimeError):
        e.services.deleter.delete(e.services.acl.principal_for("sysadmin", ["finance"], 5), gone)
    assert e.services.storage.get_file(gone).status == "deleting" and e.points(gone) == 0
    e.services.close()
    monkeypatch.undo()

    again = shared_store_env(tmp_path, client)  # process ใหม่บน DATA_DIR เดิม
    try:
        assert again.services.acl.is_withdrawn(gone)  # ซ่อนตั้งแต่ก่อน request แรก
        with TestClient(again.app):  # lifespan -> finish_pending
            assert again.services.storage.get_file(gone).status == "deleted"
            assert ids_in(retrieve(again)) == {keep}
            assert delete(again, gone).status_code == 404
            assert again.services.storage._conn().execute(
                "SELECT COUNT(*) FROM file_acls WHERE file_id = ?", (gone,)).fetchone()[0] == 0
    finally:
        again.services.close()


def test_vector_store_down_during_delete_is_503_and_finishes_later(tmp_path):
    e = Env(tmp_path)
    try:
        e.box()
        fid = ingest_done(e, docx_bytes())["file_id"]
        real = e.services.deleter._store
        e.services.deleter._store = lambda: (_ for _ in ()).throw(ConnectionError("qdrant down"))
        r = delete(e, fid)
        assert r.status_code == 503 and r.json()["error_code"] == "unavailable"
        assert "qdrant down" not in r.json()["message"]  # ไม่เผยรายละเอียดภายใน
        assert retrieve(e)["results"] == []  # ซ่อนแล้วแม้ยังลบไม่ครบ
        assert delete(e, fid).status_code == 404
        e.services.deleter._store = real
        e.services.worker.run_idle(force=True)
        assert e.services.storage.get_file(fid).status == "deleted" and e.points(fid) == 0
    finally:
        e.services.close()
