"""ingest / jobs / sources / boxes ผ่าน TestClient กับของจริงทุกชั้น ยกเว้น: OCR ปลอม (FakeBackend), hash embedding,
Qdrant ในกระบวนการ — worker เรียก ``process_next()`` ตรงๆ (ไม่ใช้ thread) ยกเว้น test ที่ตรวจ lifespan"""

from __future__ import annotations

import functools
import io
import json
import time

import pymupdf
import pytest
from docx import Document as DocxDocument
from fastapi.testclient import TestClient
from qdrant_client import models as qm

from app import runtime
from app.access import BoxRole
from app.api import readiness, routes_sources
from app.api.deps import get_settings
from app.api.main import create_app
from app.api.schemas import ComponentState, JobResponse
from app.api.services import build_services
from app.config import Settings
from app.embedding import HashEmbedder
from app.extraction.ocr import FakeBackend, OcrUnavailable
from app.storage import SqliteStorage
from app.vectorstore import VectorStore

KEY = "k"
THAI_LONG = "ข้อความภาษาไทยสำหรับทดสอบการอ่านเอกสารที่ยาวพอสมควรเพื่อให้ผ่านด่านตรวจคุณภาพ " * 4
BROKEN = "สวัสดี���� ทดสอบ"  # ผล OCR ที่ไม่ผ่านด่านตรวจ


def hdr(user: str, departments: str = "finance", level: int = 5) -> dict:
    return {"X-Service-Key": KEY, "X-User-Id": user, "X-User-Departments": departments,
            "X-User-Security-Level": str(level)}


ADMIN = hdr("sysadmin")


@functools.lru_cache(maxsize=None)
def docx_bytes(text: str = THAI_LONG, heading: str = "หัวข้อตัวอย่าง") -> bytes:
    """cache: python-docx ใส่เวลาสร้างในไฟล์ -> สร้างใหม่ข้ามวินาทีได้ไบต์ต่างกัน (ไฟล์ซ้ำจะไม่ซ้ำ)"""
    d = DocxDocument()
    d.add_heading(heading, level=1)
    d.add_paragraph(text)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def pdf_bytes(pages: list[str]) -> bytes:
    """'' = หน้าภาพล้วน (needs_ocr), ข้อความอื่น = หน้าที่มีชั้นข้อความ"""
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), False)
    pix.set_rect(pix.irect, (200, 200, 200))
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 100), text, fontname="helv", fontsize=12)
        else:
            page.insert_image(page.rect, pixmap=pix)
    return doc.tobytes()


class Env:
    def __init__(self, tmp_path, *, ocr=None, embedder=None, store_factory=None, admins=("sysadmin",)):
        self.settings = Settings(service_key=KEY, data_dir=tmp_path / "data", system_admin_users=frozenset(admins),
                                 embedding_test_mode=True)
        self.events: list[str] = []
        self.ocr = ocr if ocr is not None else (lambda: FakeBackend(THAI_LONG))
        self.store: VectorStore | None = None

        def make_store(acl):
            if store_factory is not None:
                return store_factory(acl)
            self.store = VectorStore.in_memory(acl, self.settings)
            return self.store

        self.services = build_services(
            self.settings, vector_store_factory=make_store,
            embedder_factory=embedder or (lambda: HashEmbedder(self.settings.vector_size)),
            ocr_backend_factory=lambda: self.ocr(), ocr_unloader=lambda: self.events.append("unload-ocr"))
        self.app = create_app()
        self.app.dependency_overrides[get_settings] = lambda: self.settings
        self.app.state.services = self.services
        self.client = TestClient(self.app)

    # ---- helpers ----
    def box(self, box_id="b1", visibility="private", departments=None, level=0, owner="sysadmin"):
        body = {"box_id": box_id, "name": f"กล่อง {box_id}", "visibility": visibility, "min_security_level": level}
        if departments is not None:
            body["departments"] = departments
        r = self.client.post("/v1/boxes", headers=hdr(owner), json=body)
        assert r.status_code == 201, r.text
        return r.json()

    def member(self, user, role, box_id="b1", by="sysadmin"):
        return self.client.put(f"/v1/boxes/{box_id}/members", headers=hdr(by), json={"user_id": user, "role": role})

    def ingest(self, data: bytes, name="เอกสาร_ตัวอย่าง.docx", box_id="b1", user="sysadmin", file_acl=None, **h):
        form = {"box_id": box_id}
        if file_acl is not None:
            form["file_acl"] = json.dumps(file_acl)
        return self.client.post("/v1/ingest", headers={**hdr(user), **h}, data=form, files={"file": (name, data)})

    def job(self, job_id, user="sysadmin"):
        return self.client.get(f"/v1/jobs/{job_id}", headers=hdr(user))

    def run(self) -> list[str]:
        done = []
        while (j := self.services.worker.process_next()) is not None:
            done.append(j)
        return done

    def points(self, file_id: str) -> int:
        flt = qm.Filter(must=[qm.FieldCondition(key="file_id", match=qm.MatchValue(value=file_id))])
        return sum(self.store.client.count(c, count_filter=flt, exact=True).count for c in self.store.collections)


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.services.close()
    runtime.reset_gpu_state()


def same_error(a, b) -> bool:
    ja, jb = a.json(), b.json()
    return a.status_code == b.status_code and (ja["error_code"], ja["message"]) == (jb["error_code"], jb["message"])


# ---------- boxes / members ----------


def test_create_box_by_system_admin_only(env):
    body = env.box("b1", departments=["finance"], level=1)
    assert body["owner"] == "sysadmin" and body["departments"] == ["finance"] and body["visibility"] == "private"
    assert env.services.acl.role_of("sysadmin", "b1") == BoxRole.OWNER
    r = env.client.post("/v1/boxes", headers=hdr("someone"), json={"box_id": "x", "name": "x", "visibility": "public"})
    assert r.status_code == 404 and r.json()["error_code"] == "not_found"
    dup = env.client.post("/v1/boxes", headers=ADMIN, json={"box_id": "b1", "name": "x", "visibility": "public"})
    assert dup.status_code == 422 and dup.json()["error_code"] == "invalid_request"
    assert env.services.acl.get_box("b1").visibility.value == "private"  # สร้างทับเปลี่ยน visibility ไม่ได้


def test_members_management_and_permissions(env):
    env.box()
    assert env.member("ad", "admin").status_code == 200
    assert env.member("v1", "viewer", by="ad").json()["role"] == "viewer"  # admin เพิ่ม viewer ได้
    assert env.member("x", "owner", by="ad").status_code == 404  # admin ให้ owner ไม่ได้
    assert env.client.delete("/v1/boxes/b1/members/sysadmin", headers=hdr("ad")).status_code == 404  # แตะ owner ไม่ได้
    assert env.member("v2", "viewer", by="v1").status_code == 404  # viewer จัดการไม่ได้
    not_member = env.member("v3", "viewer", by="stranger")
    unknown_box = env.member("v3", "viewer", box_id="nope", by="sysadmin")
    assert same_error(not_member, unknown_box) and not_member.status_code == 404

    last = env.member("sysadmin", "admin")  # ลด role owner คนสุดท้าย
    assert last.status_code == 422 and last.json()["error_code"] == "invalid_request"
    assert env.client.delete("/v1/boxes/b1/members/sysadmin", headers=ADMIN).status_code == 422
    assert env.client.delete("/v1/boxes/b1/members/ghost", headers=ADMIN).status_code == 404

    r = env.client.delete("/v1/boxes/b1/members/v1", headers=hdr("ad"))
    assert r.status_code == 200 and r.json() == {"box_id": "b1", "user_id": "v1", "removed": True,
                                                 "trace_id": r.headers["X-Trace-Id"]}
    reopened = SqliteStorage(env.settings.data_dir)  # อยู่ใน SQLite จริง
    assert reopened.memberships.members("b1") == {"sysadmin": BoxRole.OWNER, "ad": BoxRole.ADMIN}
    reopened.close()


# ---------- ingest: ปฏิเสธ ----------


def test_ingest_requires_owner_or_admin_and_hides_existence(env):
    env.box()
    env.member("v", "viewer")
    viewer = env.ingest(docx_bytes(), user="v")
    stranger = env.ingest(docx_bytes(), user="stranger")
    unknown = env.ingest(docx_bytes(), box_id="nope")
    assert viewer.status_code == stranger.status_code == unknown.status_code == 404
    assert same_error(viewer, stranger) and same_error(stranger, unknown)


@pytest.mark.parametrize("data, name", [
    (b"just some text, not a document", "notes.docx"),  # นามสกุลหลอก
    (b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, "scan.png"),  # ภาพเดี่ยวไม่อยู่ในรายการที่รองรับ
    ("zip", "a.docx"),  # ZIP ที่ไม่ใช่เอกสาร Office (ZIP เสีย = corrupt รับเข้าคิวแล้วงานล้ม — ไม่ใช่ 415)
])
def test_unsupported_type_is_415(env, data, name):
    if data == "zip":
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("readme.txt", "hello")
        data = buf.getvalue()
    env.box()
    r = env.ingest(data, name=name)
    assert r.status_code == 415 and r.json()["error_code"] == "unsupported_type"
    assert env.services.storage.files_in_box("b1") == [] and list(env.services.storage.files_dir.iterdir()) == []


def test_file_acl_wider_than_box_is_acl_widening(env):
    env.box(departments=["finance"], level=2)
    wide = env.ingest(docx_bytes(), file_acl={"departments": ["sales"]})
    low = env.ingest(docx_bytes(), file_acl={"min_security_level": 1})
    assert wide.status_code == low.status_code == 422
    assert wide.json()["error_code"] == low.json()["error_code"] == "acl_widening"
    assert env.services.storage.files_in_box("b1") == []


def test_too_large_is_413(env):
    env.box()
    env.settings = env.settings.model_copy(update={"max_upload_bytes": 1000})
    r = env.ingest(docx_bytes())
    assert r.status_code == 413 and r.json()["error_code"] == "too_large"


# ---------- ingest -> worker -> jobs ----------


def test_docx_end_to_end_through_the_worker(env):
    env.box()
    r = env.ingest(docx_bytes(), name=r"C:\fakepath\รายงาน_ตัวอย่าง.docx")
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued" and body["duplicate"] is False and r.headers["X-Trace-Id"] == body["trace_id"]
    queued = env.job(body["job_id"]).json()
    assert queued["status"] == "queued" and queued["progress"] == {"pages_total": None, "pages_done": 0}

    assert env.run() == [body["job_id"]]
    job = JobResponse.model_validate(env.job(body["job_id"]).json())
    assert job.status == "done" and job.document_status == "done" and job.error is None and job.gaps == []
    assert job.filename == "รายงาน_ตัวอย่าง.docx"  # ตัด path ของ client ทิ้ง
    assert job.progress.pages_total == job.progress.pages_done == job.units.ok >= 1
    fid = body["file_id"]
    assert env.points(fid) > 0
    storage = env.services.storage
    assert storage.original_path(fid).read_bytes() == docx_bytes()
    saved = json.loads(storage.document_path(fid).read_text(encoding="utf-8"))
    assert saved["file_id"] == fid and saved["filename"] == "รายงาน_ตัวอย่าง.docx" and saved["units"]

    files = env.client.get("/v1/sources/b1/files", headers=ADMIN).json()
    assert [(f["file_id"], f["status"], f["detected_type"], f["gaps"]) for f in files["files"]] == [(fid, "done", "docx", 0)]
    assert files["files"][0]["ingested_at"] and files["next_cursor"] is None


def test_scan_pdf_goes_through_ocr_with_progress_and_gaps(tmp_path):
    seen: list[tuple[str, int]] = []
    holder = {}

    def backend():
        def text(n):  # ระหว่าง OCR อ่านสถานะงานจากที่เก็บจริง
            job = holder["env"].services.storage.get_job(holder["job"])
            seen.append((job.status, job.pages_done))
            return BROKEN if n == 1 else THAI_LONG
        return FakeBackend(text)

    e = Env(tmp_path, ocr=backend)
    holder["env"] = e
    try:
        e.box()
        r = e.ingest(pdf_bytes(["", "", "Text layer page " * 20]), name="สแกน_ตัวอย่าง.pdf")
        holder["job"] = r.json()["job_id"]
        e.run()
        job = e.job(holder["job"]).json()
        assert seen == [("ocr", 1), ("ocr", 2)]  # หน้าที่ 3 มีชั้นข้อความ: นับเสร็จตั้งแต่ก่อน OCR
        assert job["status"] == "done_with_gaps" and job["document_status"] == "done_with_gaps"
        assert job["progress"] == {"pages_total": 3, "pages_done": 3}
        assert job["units"] == {"ok": 2, "needs_ocr": 0, "failed": 1, "rejected": 0}
        [gap] = job["gaps"]
        assert gap["unit_index"] == 1 and gap["status"] == "failed"
        assert gap["reason"].startswith("OCR แล้วไม่ผ่านด่านตรวจ")
        files = e.client.get("/v1/sources/b1/files", headers=ADMIN).json()["files"]
        assert files[0]["gaps"] == 1 and files[0]["units_total"] == 3
    finally:
        e.services.close()


def test_ocr_unavailable_leaves_gaps_or_fails_transiently(tmp_path):
    def broken():
        raise OcrUnavailable("ไม่พบโปรแกรม tesseract")

    e = Env(tmp_path, ocr=broken)
    try:
        e.box()
        mixed = e.ingest(pdf_bytes(["Text layer page " * 20, ""]), name="ผสม_ตัวอย่าง.pdf").json()
        scan = e.ingest(pdf_bytes(["", ""]), name="สแกนล้วน_ตัวอย่าง.pdf").json()
        e.run()
        m = e.job(mixed["job_id"]).json()
        assert m["status"] == "done_with_gaps" and m["gaps"][0]["status"] == "needs_ocr"
        assert m["gaps"][0]["reason"] == "OCR ใช้ไม่ได้: ไม่พบโปรแกรม tesseract"
        s = e.job(scan["job_id"]).json()
        assert s["status"] == "failed" and s["error"]["code"] == "unavailable" and len(s["gaps"]) == 2
    finally:
        e.services.close()


def test_corrupt_file_fails_with_document_error_without_leaking_server_paths(env):
    env.box()
    r = env.ingest(b"%PDF-1.7\n this is not really a pdf", name="เสีย_ตัวอย่าง.pdf")
    assert r.status_code == 202  # ชนิดถูก (PDF) — ล้มตอน extract พร้อมเหตุผลที่ /jobs
    env.run()
    job = env.job(r.json()["job_id"]).json()
    assert job["status"] == "failed" and job["error"]["code"] == "corrupt" and job["document_status"] == "failed"
    assert str(env.settings.data_dir) not in job["error"]["message"] and "เสีย_ตัวอย่าง.pdf" in job["error"]["message"]
    assert env.store is None  # ล้มก่อนถึงขั้น index


def test_vector_store_down_fails_job_as_unavailable(tmp_path):
    def down(acl):
        raise ConnectionError("connection refused")

    e = Env(tmp_path, store_factory=down)
    try:
        e.box()
        r = e.ingest(docx_bytes()).json()
        e.run()
        job = e.job(r["job_id"]).json()
        assert job["status"] == "failed" and job["error"]["code"] == "unavailable"
        assert "vector store" in job["error"]["message"]
    finally:
        e.services.close()


def test_jobs_run_fifo(env):
    env.box()
    ids = [env.ingest(docx_bytes(f"{THAI_LONG} ฉบับที่ {i}")).json()["job_id"] for i in range(3)]
    assert env.run() == ids


# ---------- ไฟล์ซ้ำ ----------


def test_duplicate_of_done_and_in_progress_returns_the_same_job(env):
    env.box()
    first = env.ingest(docx_bytes()).json()
    again = env.ingest(docx_bytes(), name="ชื่ออื่น.docx")
    assert again.status_code == 200 and again.json()["duplicate"] is True
    assert (again.json()["job_id"], again.json()["status"]) == (first["job_id"], "queued")  # ยังไม่ได้ทำ
    env.run()
    done = env.ingest(docx_bytes())
    assert (done.status_code, done.json()["job_id"], done.json()["status"]) == (200, first["job_id"], "done")
    other_box = env.box("b2")
    assert env.ingest(docx_bytes(), box_id="b2").status_code == 202  # กล่องอื่น = ไม่ซ้ำ
    assert other_box


@pytest.mark.parametrize("code, retry", [
    ("unavailable", True), ("internal", True), ("conversion_timeout", True),
    ("encrypted", False), ("corrupt", False), ("unsupported_type", False), ("too_large", False), ("empty", False),
])
def test_duplicate_of_failed_job_requeues_only_transient_errors(env, code, retry):
    env.box()
    first = env.ingest(docx_bytes()).json()
    env.services.storage.update_job(first["job_id"], status="failed", error_code=code, error_message="x")
    env.services.storage.update_file(first["file_id"], status="failed")
    r = env.ingest(docx_bytes())
    body = r.json()
    assert r.status_code == 200 and body["duplicate"] is True and body["file_id"] == first["file_id"]
    if retry:
        assert body["status"] == "queued" and body["job_id"] != first["job_id"]
        assert env.run() == [body["job_id"]]
        assert env.job(body["job_id"]).json()["status"] == "done"
        assert env.job(first["job_id"]).json()["status"] == "failed"  # งานเดิมคงประวัติไว้
    else:
        assert (body["status"], body["job_id"]) == ("failed", first["job_id"])
        assert env.run() == []


def test_requeue_after_real_transient_failure(tmp_path):
    state = {"down": True}

    def store(acl):
        if state["down"]:
            raise ConnectionError("qdrant down")
        return VectorStore.in_memory(acl, Settings(embedding_test_mode=True))

    e = Env(tmp_path, store_factory=store)
    try:
        e.box()
        first = e.ingest(docx_bytes()).json()
        e.run()
        assert e.job(first["job_id"]).json()["error"]["code"] == "unavailable"
        state["down"] = False
        again = e.ingest(docx_bytes()).json()
        assert again["status"] == "queued" and again["job_id"] != first["job_id"]
        e.run()
        assert e.job(again["job_id"]).json()["status"] == "done"
    finally:
        e.services.close()


# ---------- jobs: ใครเห็น ----------


def test_job_visibility(env):
    env.box()
    env.member("ad", "admin")
    env.member("ad2", "admin")
    env.member("v", "viewer")
    job_id = env.ingest(docx_bytes(), user="ad").json()["job_id"]
    assert env.job(job_id, "ad").status_code == 200  # ผู้ส่ง
    assert env.job(job_id, "ad2").status_code == 200  # admin คนอื่นของกล่อง
    assert env.job(job_id, "sysadmin").status_code == 200  # owner
    viewer, stranger, missing = env.job(job_id, "v"), env.job(job_id, "stranger"), env.job("job_nope", "sysadmin")
    assert viewer.status_code == stranger.status_code == missing.status_code == 404
    assert same_error(viewer, missing)
    env.member("ad", "viewer")  # ผู้ส่งยังเป็นสมาชิก (viewer) -> ยังเห็นงานของตัวเอง
    assert env.job(job_id, "ad").status_code == 200
    env.client.delete("/v1/boxes/b1/members/ad", headers=ADMIN)  # ถูกถอดจากกล่อง -> ไม่เห็นแล้ว
    assert env.job(job_id, "ad").status_code == 404


# ---------- sources ----------


def test_sources_lists_member_boxes_with_acl_filtered_counts(env):
    env.box("b1", departments=["finance", "hr"])
    env.box("b2", visibility="public")
    env.box("b3")
    env.member("v", "viewer", "b1")
    env.member("v", "admin", "b2")
    env.ingest(docx_bytes(), box_id="b1")
    env.ingest(docx_bytes(f"{THAI_LONG} ลับ"), box_id="b1", file_acl={"min_security_level": 9})
    env.ingest(docx_bytes(f"{THAI_LONG} hr"), box_id="b1", file_acl={"departments": ["hr"]})
    env.run()
    r = env.client.get("/v1/sources", headers=hdr("v", "finance", 5)).json()
    got = {s["box_id"]: (s["role"], s["visibility"], s["file_count"], s["name"]) for s in r["sources"]}
    assert got == {"b1": ("viewer", "private", 1, "กล่อง b1"), "b2": ("admin", "public", 0, "กล่อง b2")}
    b1 = next(s for s in r["sources"] if s["box_id"] == "b1")
    assert b1["updated_at"] is not None and r["sources"][1]["updated_at"] is None
    assert env.client.get("/v1/sources", headers=hdr("stranger")).json()["sources"] == []


def test_files_pagination_with_acl_filter(env, monkeypatch):
    monkeypatch.setattr(routes_sources, "FILES_PAGE_SIZE", 2)
    monkeypatch.setattr(routes_sources, "_SCAN_BATCH", 3)
    env.box(departments=["finance", "hr"])
    env.member("v", "viewer")
    ids = []
    for i in range(6):
        acl = {"departments": ["hr"]} if i == 2 else None  # ไฟล์ที่ 3 v (finance) ไม่เห็น
        ids.append(env.ingest(docx_bytes(f"{THAI_LONG} {i}"), file_acl=acl).json()["file_id"])
    seen, cursor, pages = [], None, 0
    while True:
        params = {"cursor": cursor} if cursor else {}
        r = env.client.get("/v1/sources/b1/files", headers=hdr("v", "finance"), params=params).json()
        pages += 1
        seen += [f["file_id"] for f in r["files"]]
        cursor = r["next_cursor"]
        if cursor is None:
            break
    assert seen == [f for i, f in enumerate(ids) if i != 2] and pages == 3
    assert env.client.get("/v1/sources/b1/files", headers=hdr("v"), params={"cursor": "%%%"}).status_code == 422
    not_member = env.client.get("/v1/sources/b1/files", headers=hdr("stranger"))
    unknown = env.client.get("/v1/sources/nope/files", headers=hdr("stranger"))
    assert not_member.status_code == 404 and same_error(not_member, unknown)


# ---------- GPU / VRAM / restart ----------


class GpuEmbedder(HashEmbedder):
    device = "cuda"

    def __init__(self, events):
        super().__init__(1024)
        self.events = events

    def release(self):
        self.events.append("release-embedder")


def test_ocr_and_embedding_never_share_the_gpu(tmp_path):
    events: list[str] = []
    e = Env(tmp_path, embedder=lambda: (events.append("load-embedder"), GpuEmbedder(events))[1])
    e.events = events
    e.services.worker._ocr_unloader = lambda: events.append("unload-ocr")
    e.services.worker._ocr_backend_factory = lambda: (events.append("load-ocr"), FakeBackend(THAI_LONG))[1]
    try:
        e.box()
        e.ingest(docx_bytes())
        e.ingest(pdf_bytes(["", "Text layer page " * 20]), name="สแกน.pdf")
        e.run()
        assert events == [
            "unload-ocr", "load-embedder",  # งาน 1 (ไม่ต้อง OCR): ไม่มี OCR ค้างก่อนโหลด embedding
            "release-embedder", "load-ocr", "unload-ocr",  # งาน 2: ปล่อย embedding ก่อน OCR, ปล่อย OCR หลังเสร็จ
            "unload-ocr", "load-embedder",
        ]
    finally:
        e.services.close()


class AcceleratorError(RuntimeError):
    """ชื่อเดียวกับ torch.AcceleratorError (is_fatal_cuda ดูชื่อชนิด)"""


def test_fatal_cuda_fails_job_pauses_worker_and_reports_down(tmp_path):
    e = Env(tmp_path, ocr=lambda: FakeBackend([AcceleratorError("CUDA error: an illegal memory access was encountered")]))
    try:
        e.box()
        bad = e.ingest(pdf_bytes(["", ""]), name="สแกน.pdf").json()
        nxt = e.ingest(docx_bytes()).json()
        assert e.services.worker.process_next() == bad["job_id"]
        job = e.job(bad["job_id"]).json()
        assert job["status"] == "failed" and job["error"]["code"] == "unavailable"
        assert "CUDA" in job["error"]["message"]
        assert runtime.gpu_broken_reason() is not None
        assert e.services.worker.process_next() is None  # ไม่ใช้ GPU ต่อเงียบๆ: งานถัดไปรออยู่ในคิว
        assert e.job(nxt["job_id"]).json()["status"] == "queued"
        rd = readiness.probe(Settings(), gpu=True, qdrant=lambda _s: ComponentState.OK)
        assert rd.status.value == "not_ready"
        assert (rd.components.embedding, rd.components.ocr) == (ComponentState.DOWN, ComponentState.DOWN)
        assert any("GPU ใช้ต่อไม่ได้" in n for n in rd.runtime.notes)
        cpu = readiness.probe(Settings(embedding_device="cpu", ocr_backend="tesseract-local"), gpu=True,
                              qdrant=lambda _s: ComponentState.OK, tesseract=lambda: (True, ""))
        assert cpu.components.embedding == ComponentState.OK and cpu.components.ocr == ComponentState.OK
    finally:
        e.services.close()
        runtime.reset_gpu_state()


def test_interrupted_jobs_are_requeued_on_restart(tmp_path):
    e = Env(tmp_path)
    e.box()
    first = e.ingest(docx_bytes()).json()
    e.services.storage.claim_next_job()  # process ตายระหว่าง extracting
    e.services.storage.update_job(first["job_id"], status="ocr", pages_done=2)
    e.services.close()

    again = Env(tmp_path)  # process ใหม่บน DATA_DIR เดิม
    try:
        with TestClient(again.app):  # lifespan: worker.start() -> คืนคิว -> thread ทำงาน
            deadline = time.time() + 30
            while time.time() < deadline:
                status = again.job(first["job_id"]).json()["status"]
                if status in ("done", "done_with_gaps", "failed"):
                    break
                time.sleep(0.1)
        assert status == "done"
    finally:
        again.services.close()
