"""OCR ใช้ไม่ได้ = หน้ายังเป็น needs_ocr (ไม่ใช่ failed) + ส่งไฟล์ซ้ำเมื่อ OCR พร้อมแล้วเข้าคิวใหม่

backend ปลอม ``GpuOcr`` (ชื่อ/พารามิเตอร์คงที่ -> key ของแคช OCR ตรงกันระหว่างตอน "ไม่มี GPU" กับ "มี GPU" เหมือน Typhoon จริง)
โยน ``OcrUnavailable`` ตามลำดับ call ที่กำหนด — หน้าละ 1 call (ภาพเทาเรียบ ไม่ตัดหน้า) ตามลำดับหน้า
"""

from __future__ import annotations

import pytest

from app import runtime
from app.extraction.ocr import FakeBackend, OcrUnavailable
from tests.test_ingest_api import THAI_LONG, Env, pdf_bytes

NO_GPU = "ไม่พบ GPU CUDA — ทดสอบ"
TEXT_PAGE = "Text layer page " * 20
# OCR ได้ข้อความ แต่มีละตินกลางคำไทยหลายจุด -> ด่านตรวจ latin_in_thai -> failed (ไม่ใช่ needs_ocr)
LATIN_BROKEN = ("แกmผ่ถ่อหุ้น เน่Çอหา กxขyค ทดสoบระบบ ข้อมูlสำคัญ " * 6).strip()


class GpuOcr(FakeBackend):
    name = "gpu-ocr-test"
    model = "gpu-ocr"
    revision = "r1"

    def __init__(self, unavailable_calls=(), text=THAI_LONG) -> None:
        super().__init__(lambda n: OcrUnavailable(NO_GPU) if n in unavailable_calls else text)

    def cache_params(self) -> dict:
        return {"long_side": 1800}  # คงที่ (FakeBackend ใส่ repr ของฟังก์ชัน -> key เปลี่ยนทุกตัว)


class Switchable:
    """factory ของ worker: เปลี่ยน backend ระหว่าง test ได้ ("ติดตั้ง GPU แล้ว")"""

    def __init__(self, backend: GpuOcr) -> None:
        self.backend = backend

    def __call__(self) -> GpuOcr:
        return self.backend


@pytest.fixture
def make_env(tmp_path):
    made = []

    def make(backend: GpuOcr):
        ocr = Switchable(backend)
        e = Env(tmp_path / f"e{len(made)}", ocr=ocr)
        e.ocr_switch = ocr
        made.append(e)
        e.box()
        return e

    yield make
    for e in made:
        e.services.close()
    runtime.reset_gpu_state()


def test_some_pages_without_ocr_are_needs_ocr_gaps_and_the_job_is_done_with_gaps(make_env):
    """(ก) ขาดบางหน้า -> done_with_gaps + gap needs_ocr พร้อมเหตุผล (หน้าอื่นใช้ได้ตามปกติ)"""
    e = make_env(GpuOcr(unavailable_calls={1}))  # หน้า 1 OCR ได้, หน้า 2 OCR ใช้ไม่ได้, หน้า 3 มีชั้นข้อความ
    r = e.ingest(pdf_bytes(["", "", TEXT_PAGE]), name="สแกน.pdf").json()
    e.run()
    job = e.job(r["job_id"]).json()
    assert job["status"] == "done_with_gaps" and job["error"] is None
    assert job["units"] == {"ok": 2, "needs_ocr": 1, "failed": 0, "rejected": 0}
    [gap] = job["gaps"]
    assert gap["unit_index"] == 1 and gap["status"] == "needs_ocr"
    assert gap["reason"] == f"OCR ใช้ไม่ได้: {NO_GPU}"
    assert e.points(r["file_id"]) > 0


def test_all_ocr_pages_unavailable_fails_as_unavailable_not_empty(make_env):
    """(ข) ทุกหน้าที่ต้อง OCR ขาด (ไม่มีหน้า ok) -> failed / unavailable (ชั่วคราว ส่งซ้ำได้)"""
    e = make_env(GpuOcr(unavailable_calls={0, 1}))
    r = e.ingest(pdf_bytes(["", ""]), name="สแกนล้วน.pdf").json()
    e.run()
    job = e.job(r["job_id"]).json()
    assert job["status"] == "failed" and job["error"]["code"] == "unavailable"
    assert "2 หน้าที่ต้อง OCR ยังไม่ได้ OCR" in job["error"]["message"] and NO_GPU in job["error"]["message"]
    assert [g["status"] for g in job["gaps"]] == ["needs_ocr", "needs_ocr"]
    assert all(g["reason"] == f"OCR ใช้ไม่ได้: {NO_GPU}" for g in job["gaps"])
    assert job["units"]["needs_ocr"] == 2 and job["units"]["failed"] == 0


def test_resend_after_gpu_is_back_requeues_and_finishes_without_duplicate_points(make_env):
    """(ค) done_with_gaps ที่มีหน้า needs_ocr -> ส่งไฟล์เดิมซ้ำหลัง "มี GPU" = เข้าคิวใหม่ (job ใหม่, file เดิม, ACL/ชื่อเดิม)
    -> done · หน้าที่ OCR แล้วได้จากแคช (OCR จริงแค่หน้าที่ขาด) · point ไม่ซ้ำซ้อน"""
    data = pdf_bytes(["", "", TEXT_PAGE])
    e = make_env(GpuOcr(unavailable_calls={1}))
    first = e.ingest(data, name="สแกน.pdf", file_acl={"departments": ["finance"]}).json()
    e.run()
    before = e.points(first["file_id"])
    assert e.job(first["job_id"]).json()["status"] == "done_with_gaps"

    gpu = GpuOcr()  # ติดตั้ง GPU แล้ว
    e.ocr_switch.backend = gpu
    again = e.ingest(data, name="ชื่อใหม่ที่ไม่ถูกใช้.pdf")
    assert again.status_code == 200
    body = again.json()
    assert body["duplicate"] is True and body["status"] == "queued"
    assert body["file_id"] == first["file_id"] and body["job_id"] != first["job_id"]
    assert e.run() == [body["job_id"]]
    job = e.job(body["job_id"]).json()
    assert job["status"] == "done" and job["gaps"] == [] and job["units"]["ok"] == 3
    assert gpu.calls == 1  # หน้า 1 จากแคช — OCR จริงเฉพาะหน้า 2 ที่ขาด
    assert job["filename"] == "สแกน.pdf"  # ชื่อเดิม
    acl = e.services.acl.get_file_acl(first["file_id"])
    assert acl.departments == frozenset({"finance"})  # ACL เดิม

    after = e.points(first["file_id"])
    fresh = make_env(GpuOcr())  # ไฟล์เดียวกันจากศูนย์ (ไม่มีรอบที่ขาด) = จำนวน point ที่ถูกต้อง
    ref = fresh.ingest(data, name="สแกน.pdf").json()
    fresh.run()
    assert after == fresh.points(ref["file_id"]) > before  # ลบ point เดิมก่อนเขียน ไม่ซ้ำซ้อน

    third = e.ingest(data).json()  # จบแล้ว (done) -> ส่งซ้ำได้งานเดิม
    assert third["job_id"] == body["job_id"] and third["status"] == "done"


def test_all_pages_unavailable_then_resend_also_requeues(make_env):
    data = pdf_bytes(["", ""])
    e = make_env(GpuOcr(unavailable_calls={0, 1}))
    first = e.ingest(data).json()
    e.run()
    e.ocr_switch.backend = GpuOcr()
    body = e.ingest(data).json()
    assert body["status"] == "queued" and body["job_id"] != first["job_id"]
    e.run()
    assert e.job(body["job_id"]).json()["status"] == "done"


def test_gate_failure_after_ocr_is_failed_and_resend_returns_the_same_job(make_env):
    """(ง) OCR แล้วไม่ผ่านด่านตรวจ (latin_in_thai) -> gap failed (ไม่ใช่ needs_ocr) · ส่งซ้ำ = งานเดิม ไม่เข้าคิวใหม่"""
    e = make_env(GpuOcr())
    e.ocr_switch.backend = FakeBackend(lambda n: [THAI_LONG, LATIN_BROKEN][n])
    data = pdf_bytes(["", "", TEXT_PAGE])
    first = e.ingest(data).json()
    e.run()
    job = e.job(first["job_id"]).json()
    assert job["status"] == "done_with_gaps"
    [gap] = job["gaps"]
    assert gap["status"] == "failed" and "OCR แล้วไม่ผ่านด่านตรวจ" in gap["reason"] and "latin_in_thai" in gap["reason"]
    again = e.ingest(data)
    assert again.status_code == 200
    body = again.json()
    assert body["duplicate"] is True and body["job_id"] == first["job_id"] and body["status"] == "done_with_gaps"
    assert e.run() == []  # ไม่มีงานใหม่
