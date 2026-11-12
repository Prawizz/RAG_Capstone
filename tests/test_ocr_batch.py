"""scripts/ocr_batch.py: เลือกหน้า, ข้ามแคช, log, OOM retry, error รายหน้า, KeepAwake (FakeBackend ไม่ต้องมี GPU)"""

import importlib.util
import json
from pathlib import Path

import pymupdf
import pytest

from app.extraction.ocr import FakeBackend, OcrUnavailable
from app.extraction.pipeline import extract_file
from app.models import UnitStatus

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("ocr_batch", ROOT / "scripts" / "ocr_batch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ob = _load()
TEXT = "ข้อความภาษาไทยสำหรับทดสอบการอ่านเอกสารที่ยาวพอสมควรเพื่อให้ผ่านด่านตรวจคุณภาพ " * 4


def make_pdf(path, pages):
    """pages: ข้อความ ('' = หน้าภาพล้วน -> needs_ocr:scan_like)"""
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), False)
    pix.set_rect(pix.irect, (200, 200, 200))
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 100), text, fontname="helv", fontsize=12)
        else:
            page.insert_image(page.rect, pixmap=pix)
    doc.save(str(path))
    return path


class Scripted(FakeBackend):
    """FakeBackend ที่ cache key คงที่ข้ามรอบ (ของจริงใช้ repr ของ texts) และโยน BaseException ได้ (เช่น KeyboardInterrupt)
    ``fail`` = {ลำดับ call: exception}; call อื่นคืน TEXT"""

    def __init__(self, fail=None):
        super().__init__(TEXT)
        self.fail = fail or {}

    def recognize(self, image_png):
        n = self.calls
        if n in self.fail:
            self.calls += 1
            raise self.fail[n]
        return super().recognize(image_png)


@pytest.fixture
def pdf(tmp_path):
    # หน้า 1 มีข้อความ (ok) · หน้า 2,3,4,5 เป็นภาพล้วน (needs_ocr)
    return make_pdf(tmp_path / "a.pdf", ["Hello text layer page " * 10, "", "", "", ""])


@pytest.fixture
def env(tmp_path):
    return ob.CheckedCache(tmp_path / "cache"), ob.BatchLog(tmp_path / "out" / "ocr_batch.log"), tmp_path / "summary.json"


def lines(log):
    return [l for l in log.path.read_text(encoding="utf-8").splitlines() if not l.startswith("#")]


# ---------- เลือกหน้า ----------

@pytest.mark.parametrize("spec,expected", [
    (None, None), ("", None), ("  ", None), ("5", {5}), ("1-3", {1, 2, 3}), ("1-3, 10,12-13", {1, 2, 3, 10, 12, 13}),
])
def test_parse_pages(spec, expected):
    assert ob.parse_pages(spec) == expected


@pytest.mark.parametrize("bad", ["0", "3-1", "a", "1-x", "-2"])
def test_parse_pages_rejects_bad(bad):
    with pytest.raises(ValueError):
        ob.parse_pages(bad)


def test_select_pages_sorted_deduped_filtered():
    assert ob.select_pages([9, 3, 3, 5, 1], None) == [1, 3, 5, 9]
    assert ob.select_pages([9, 3, 5, 1], {3, 4, 5, 6}) == [3, 5]
    assert ob.select_pages([1, 2], {7}) == []


def test_plan_uses_gate_and_only_needs_ocr_pages(pdf):
    plan = ob.plan_files([pdf], None)[0]
    assert plan["pages"] == [2, 3, 4, 5] and plan["needs_ocr_total"] == 4 and plan["total_pages"] == 5
    assert ob.plan_files([pdf], {3, 4, 99})[0]["pages"] == [3, 4]
    assert ob.plan_files([pdf], {3, 4, 99})[0]["needs_ocr_total"] == 4  # ยังรายงานจำนวนทั้งหมดของไฟล์


def test_plan_missing_or_non_pdf_file_is_recorded_not_raised(tmp_path):
    png = tmp_path / "x.png"
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 30), False)
    png.write_bytes(pix.tobytes("png"))
    plans = ob.plan_files([tmp_path / "missing.pdf", png], None)
    assert all(p["error"] and p["pages"] == [] for p in plans)


# ---------- รัน batch ----------

def test_batch_ocrs_every_needs_ocr_page_and_writes_cache_log_summary(pdf, env):
    cache, log, sp = env
    backend = FakeBackend(TEXT)
    s = ob.run_batch([pdf], backend, cache=cache, log=log, summary_path=sp, out=lambda *_: None)
    f = s["files"]["a.pdf"]
    assert (f["selected"], f["ok"], f["cached"], f["error"], f["not_reached"]) == (4, 4, 0, 0, 0)
    assert backend.calls == 4
    assert s["totals"] == {"planned": 4, "ok": 4, "cached": 0, "error": 0, "not_reached": 0}
    assert len(list(cache.dir.glob("*.json"))) == 4
    assert json.loads(sp.read_text(encoding="utf-8"))["files"]["a.pdf"]["ok"] == 4
    rows = [l.split("\t") for l in lines(log)]
    assert [r[2] for r in rows] == ["2", "3", "4", "5"]
    assert all(r[1] == "a.pdf" and r[6] == "miss" and r[7] == "ok" for r in rows)


def test_rerun_skips_cached_pages_without_calling_backend(pdf, env):
    cache, log, sp = env
    ob.run_batch([pdf], FakeBackend(TEXT), cache=cache, log=log, out=lambda *_: None)
    again = FakeBackend(TEXT)
    s = ob.run_batch([pdf], again, cache=cache, log=log, summary_path=sp, out=lambda *_: None)
    assert again.calls == 0
    f = s["files"]["a.pdf"]
    assert (f["ok"], f["cached"], f["error"]) == (0, 4, 0)
    last = [l.split("\t") for l in lines(log)][-4:]
    assert all(r[6] == "hit" and r[7] == "cached" for r in last)


def test_resume_after_interrupt_continues_from_cache(pdf, env):
    cache, log, sp = env
    boom = Scripted({2: KeyboardInterrupt()})  # Ctrl+C ตอนหน้าที่ 3
    s = ob.run_batch([pdf], boom, cache=cache, log=log, summary_path=sp, out=lambda *_: None)
    assert s["interrupted"] and s["files"]["a.pdf"]["ok"] == 2 and s["files"]["a.pdf"]["not_reached"] == 2
    assert json.loads(sp.read_text(encoding="utf-8"))["interrupted"] is True  # summary ถูกเขียนแม้ขัดจังหวะ
    ok = FakeBackend(TEXT)
    s2 = ob.run_batch([pdf], ok, cache=cache, log=log, out=lambda *_: None)
    assert ok.calls == 2 and s2["files"]["a.pdf"]["cached"] == 2 and s2["files"]["a.pdf"]["ok"] == 2


def test_cache_key_matches_pipeline_run_ocr(pdf, env):
    """batch ต้องเขียนแคชด้วย key เดียวกับ run_ocr (ห้ามเปลี่ยนพารามิเตอร์) -> run_ocr ตามหลังได้ hit ทั้งหมด"""
    from app.extraction.ocr import run_ocr

    cache, log, _ = env
    ob.run_batch([pdf], FakeBackend(TEXT), cache=cache, log=log, out=lambda *_: None)
    later = FakeBackend(TEXT)
    import os

    # run_ocr ใช้ OcrCache() ปริยาย -> ชี้ไปโฟลเดอร์เดียวกันผ่าน env
    os.environ["OCR_CACHE_DIR"] = str(cache.dir)
    try:
        out = run_ocr(extract_file(pdf), pdf, later)
    finally:
        del os.environ["OCR_CACHE_DIR"]
    assert later.calls == 0
    assert [u.signals.get("ocr:cache_hit") for u in out.units if u.method.value == "ocr"] == [1.0] * 4


def test_limit_counts_only_real_runs_not_cached(pdf, env):
    cache, log, _ = env
    ob.run_batch([pdf], FakeBackend(TEXT), cache=cache, log=log, wanted={2}, out=lambda *_: None)  # หน้า 2 มีแคชแล้ว
    b = FakeBackend(TEXT)
    s = ob.run_batch([pdf], b, cache=cache, log=log, limit=2, out=lambda *_: None)
    f = s["files"]["a.pdf"]
    assert b.calls == 2 and (f["cached"], f["ok"], f["not_reached"]) == (1, 2, 1)  # 2 (แคช) + 3,4 รันจริง · 5 ยังไม่ถึง


def test_pages_selection_limits_work(pdf, env):
    cache, log, _ = env
    b = FakeBackend(TEXT)
    s = ob.run_batch([pdf], b, cache=cache, log=log, wanted={3, 5, 1}, out=lambda *_: None)  # 1 ไม่ใช่ needs_ocr -> ไม่ถูกเลือก
    assert b.calls == 2 and s["files"]["a.pdf"]["selected"] == 2
    assert [l.split("\t")[2] for l in lines(log)] == ["3", "5"]


def test_page_error_is_logged_and_batch_continues(pdf, env):
    cache, log, _ = env
    backend = FakeBackend(lambda n: ValueError("โมเดลพัง") if n == 1 else TEXT)
    s = ob.run_batch([pdf], backend, cache=cache, log=log, out=lambda *_: None)
    f = s["files"]["a.pdf"]
    assert (f["ok"], f["error"]) == (3, 1) and f["error_pages"][0]["page"] == 3 and "ValueError" in f["error_pages"][0]["error"]
    assert backend.calls == 4
    row = [l.split("\t") for l in lines(log)][1]
    assert row[2] == "3" and row[7] == "error" and "โมเดลพัง" in row[8]


def test_failed_page_is_retried_on_next_run(pdf, env):
    cache, log, _ = env
    ob.run_batch([pdf], Scripted({0: ValueError("x")}), cache=cache, log=log, out=lambda *_: None)
    b = FakeBackend(TEXT)
    s = ob.run_batch([pdf], b, cache=cache, log=log, out=lambda *_: None)
    assert b.calls == 1 and s["files"]["a.pdf"]["ok"] == 1 and s["files"]["a.pdf"]["cached"] == 3


class OomError(RuntimeError):
    pass


OomError.__name__ = "OutOfMemoryError"


def test_oom_clears_memory_and_retries_page_once(pdf, env):
    cache, log, _ = env
    freed = []
    backend = FakeBackend(lambda n: OomError("CUDA out of memory") if n == 0 else TEXT)
    s = ob.run_batch([pdf], backend, cache=cache, log=log, out=lambda *_: None, free_memory=lambda: freed.append(1))
    f = s["files"]["a.pdf"]
    assert freed == [1] and (f["ok"], f["error"]) == (4, 0) and backend.calls == 5  # หน้า 2 รัน 2 ครั้ง


def test_oom_twice_on_same_page_is_error_then_continues(pdf, env):
    cache, log, _ = env
    freed = []
    backend = FakeBackend(lambda n: OomError("CUDA out of memory") if n in (0, 1) else TEXT)
    s = ob.run_batch([pdf], backend, cache=cache, log=log, out=lambda *_: None, free_memory=lambda: freed.append(1))
    f = s["files"]["a.pdf"]
    assert freed == [1] and (f["ok"], f["error"]) == (3, 1) and f["error_pages"][0]["page"] == 2
    assert "OOM" in f["error_pages"][0]["error"]


def test_backend_unavailable_aborts_batch_and_writes_summary(pdf, env):
    cache, log, sp = env
    backend = FakeBackend(lambda n: OcrUnavailable("ไม่พบ GPU"))
    s = ob.run_batch([pdf], backend, cache=cache, log=log, summary_path=sp, out=lambda *_: None)
    assert backend.calls == 1 and "OcrUnavailable" in s["aborted"] and s["files"]["a.pdf"]["not_reached"] == 4
    assert json.loads(sp.read_text(encoding="utf-8"))["aborted"]


def test_unwritable_cache_aborts(pdf, tmp_path):
    blocker = tmp_path / "cache"
    blocker.write_text("ไฟล์ ไม่ใช่โฟลเดอร์")  # mkdir ใต้ไฟล์ไม่ได้ -> put ล้มเหลวเงียบ
    log = ob.BatchLog(tmp_path / "l.log")
    s = ob.run_batch([pdf], FakeBackend(TEXT), cache=ob.CheckedCache(blocker), log=log, out=lambda *_: None)
    assert s["aborted"] and "แคช" in s["aborted"]


def test_files_processed_in_given_order_and_bad_file_does_not_stop_others(tmp_path, env):
    cache, log, _ = env
    a = make_pdf(tmp_path / "a.pdf", ["", ""])
    b = make_pdf(tmp_path / "b.pdf", [""])
    s = ob.run_batch([tmp_path / "gone.pdf", b, a], FakeBackend(TEXT), cache=cache, log=log, out=lambda *_: None)
    assert list(s["files"]) == ["gone.pdf", "b.pdf", "a.pdf"]
    assert s["files"]["gone.pdf"]["file_error"] and s["files"]["b.pdf"]["ok"] == 1 and s["files"]["a.pdf"]["ok"] == 2
    assert [l.split("\t")[1] for l in lines(log)] == ["b.pdf", "a.pdf", "a.pdf"]


def test_log_line_format_and_tab_safety(tmp_path):
    log = ob.BatchLog(tmp_path / "x.log")
    log.page({"time": "T", "file": "f.pdf", "page": 7, "seconds": 12.34, "ocr_seconds": 11.0, "tokens": 99, "hit": False,
              "status": "error", "error": "a\tb\nc"})
    log.page({"time": "T", "file": "f.pdf", "page": 8, "seconds": 0.5, "ocr_seconds": 30.0, "tokens": None, "hit": True,
              "status": "cached", "error": None})
    r1, r2 = [l.split("\t") for l in lines(log)]
    assert r1 == ["T", "f.pdf", "7", "12.3", "11.0", "99", "miss", "error", "a b c"]
    assert r2[5] == "-" and r2[6] == "hit" and r2[8] == ""


def test_progress_line_has_eta(pdf, env):
    cache, log, _ = env
    out = []
    ob.run_batch([pdf], FakeBackend(TEXT), cache=cache, log=log, out=out.append)
    prog = [o for o in out if o.startswith("[")]
    assert prog[0].startswith("[1/4] a.pdf p2 miss") and "ETA" in prog[0]
    assert "ETA ≤" in prog[-1] or "ETA" in prog[-1]


# ---------- กัน sleep ----------

def test_keep_awake_sets_and_restores_previous_state(monkeypatch):
    calls = []

    class K(ob.KeepAwake):
        def _set(self, flags):
            calls.append(flags)
            return 0x80000001 if len(calls) == 1 else 0  # สถานะเดิม: continuous + system_required

    monkeypatch.setattr(ob.sys, "platform", "win32")
    with K() as ka:
        assert ka.active
    assert calls == [0x80000001, 0x80000001] and not ka.active


def test_keep_awake_restores_to_clear_when_previous_was_empty(monkeypatch):
    calls = []

    class K(ob.KeepAwake):
        def _set(self, flags):
            calls.append(flags)
            return 0x80000000 if len(calls) == 1 else 0  # เดิมมีแค่ ES_CONTINUOUS (ไม่มีแฟล็กขอ)

    monkeypatch.setattr(ob.sys, "platform", "win32")
    with K():
        pass
    assert calls == [0x80000001, 0x80000000]  # คืน = ES_CONTINUOUS เปล่า ล้างที่เราตั้ง


def test_keep_awake_restores_on_exception(monkeypatch):
    calls = []

    class K(ob.KeepAwake):
        def _set(self, flags):
            calls.append(flags)
            return 0x80000000

    monkeypatch.setattr(ob.sys, "platform", "win32")
    with pytest.raises(RuntimeError):
        with K():
            raise RuntimeError
    assert len(calls) == 2


def test_keep_awake_noop_off_windows_or_disabled(monkeypatch):
    class K(ob.KeepAwake):
        def _set(self, flags):
            raise AssertionError("ไม่ควรเรียก")

    monkeypatch.setattr(ob.sys, "platform", "linux")
    with K() as ka:
        assert not ka.active
    monkeypatch.setattr(ob.sys, "platform", "win32")
    with K(enabled=False) as ka:
        assert not ka.active


# ---------- cooldown ----------

def _run(pdf, env, backend=None, **kw):
    cache, log, _ = env
    sleeps = []
    s = ob.run_batch([pdf], backend or FakeBackend(TEXT), cache=cache, log=log, out=lambda *_: None,
                     sleep=sleeps.append, **kw)
    return s, sleeps


def test_cooldown_between_real_runs_but_not_after_the_last(pdf, env):
    s, sleeps = _run(pdf, env, cooldown=7)
    assert s["files"]["a.pdf"]["ok"] == 4 and sleeps == [7, 7, 7]  # 4 หน้า -> พัก 3 ครั้ง


def test_default_cooldown_is_zero_and_never_sleeps(pdf, env):
    _, sleeps = _run(pdf, env)
    assert sleeps == []


def test_cached_pages_never_trigger_cooldown(pdf, env):
    _run(pdf, env)  # เติมแคชทุกหน้า
    s, sleeps = _run(pdf, env, cooldown=7)
    assert s["files"]["a.pdf"]["cached"] == 4 and sleeps == []


def test_no_cooldown_after_a_hit_only_between_real_runs(pdf, env):
    _run(pdf, env, wanted={2})  # หน้า 2 มีแคช
    s, sleeps = _run(pdf, env, cooldown=7)  # คิว: 2(hit) 3 4 5(รันจริง)
    assert s["files"]["a.pdf"]["cached"] == 1 and sleeps == [7, 7]  # ไม่พักหลัง hit · พักหลัง 3 และ 4 · ไม่พักหลัง 5


def test_error_page_also_cools_down(pdf, env):
    _, sleeps = _run(pdf, env, backend=Scripted({0: ValueError("x")}), cooldown=3)
    assert sleeps == [3, 3, 3]


def test_no_trailing_cooldown_when_limit_reached(pdf, env):
    s, sleeps = _run(pdf, env, cooldown=7, limit=2)
    assert s["files"]["a.pdf"]["ok"] == 2 and sleeps == [7]


def test_cooldown_spans_files(tmp_path, env):
    a = make_pdf(tmp_path / "a.pdf", ["", ""])
    b = make_pdf(tmp_path / "b.pdf", [""])
    cache, log, _ = env
    sleeps = []
    ob.run_batch([a, b], FakeBackend(TEXT), cache=cache, log=log, out=lambda *_: None, sleep=sleeps.append, cooldown=2)
    assert sleeps == [2, 2]  # a1 | a2 | b1(สุดท้าย)


def test_ctrl_c_during_cooldown_is_a_clean_interrupt(pdf, env):
    cache, log, sp = env

    def stop(_):
        raise KeyboardInterrupt

    s = ob.run_batch([pdf], FakeBackend(TEXT), cache=cache, log=log, summary_path=sp, out=lambda *_: None,
                     sleep=stop, cooldown=5)
    assert s["interrupted"] and s["files"]["a.pdf"]["ok"] == 1 and json.loads(sp.read_text(encoding="utf-8"))["interrupted"]


def test_cooldown_does_not_change_cache_keys(pdf, tmp_path):
    """cooldown ต้องไม่กระทบ cache key: รันด้วย cooldown แล้วรันอีกรอบโดยไม่มี -> hit ทั้งหมด"""
    cache = ob.CheckedCache(tmp_path / "c")
    log = ob.BatchLog(tmp_path / "l.log")
    ob.run_batch([pdf], FakeBackend(TEXT), cache=cache, log=log, out=lambda *_: None, sleep=lambda _: None, cooldown=9)
    again = FakeBackend(TEXT)
    ob.run_batch([pdf], again, cache=cache, log=log, out=lambda *_: None)
    assert again.calls == 0


def test_cli_rejects_negative_cooldown():
    with pytest.raises(SystemExit) as e:
        ob.main(["--cooldown", "-1"])
    assert e.value.code == 2


# ---------- ETA ----------

def test_estimate_eta_is_mean_of_actual_miss_times():
    assert ob.estimate_eta([], 10) is None
    assert ob.estimate_eta([10], 3) == 30
    assert ob.estimate_eta([10, 30], 3) == 60  # เฉลี่ย 20
    assert ob.estimate_eta([100, 20, 30, 50], 2) == 2 * 50  # ค่าเฉลี่ยของทุกหน้าที่ miss ไม่ตัดหน้าไหนทิ้ง
    assert ob.estimate_eta([10, 30], 0) == 0.0


def test_estimate_eta_adds_cooldown_between_remaining_pages_only():
    assert ob.estimate_eta([10], 3, cooldown=5) == 3 * 10 + 2 * 5
    assert ob.estimate_eta([10], 1, cooldown=5) == 10


def test_estimate_eta_changes_with_the_run_not_a_constant():
    assert ob.estimate_eta([10], 5) != ob.estimate_eta([10, 90], 5)


def test_progress_eta_follows_measured_miss_seconds(pdf, env, monkeypatch):
    seq = iter([10.0, 30.0, 20.0, 20.0])

    def fake_page(backend, path, content_hash, page_no, cache, **_):
        return {"time": "T", "page": page_no, "hit": False, "tokens": 1, "ocr_seconds": 1.0, "error": None,
                "status": "ok", "seconds": next(seq)}

    monkeypatch.setattr(ob, "ocr_one_page", fake_page)
    cache, log, _ = env
    out = []
    ob.run_batch([pdf], FakeBackend(TEXT), cache=cache, log=log, out=out.append, sleep=lambda _: None)
    etas = [o.rsplit("ETA", 1)[1].strip() for o in out if o.startswith("[")]
    # หลัง 1: 3 หน้า x 10 · หลัง 2: 2 x 20 · หลัง 3: 1 x 20 · หลัง 4: ไม่เหลือ
    assert etas == ["≤0m30s", "≤0m40s", "≤0m20s", "≤0m00s"]


def test_progress_eta_includes_cooldown(pdf, env, monkeypatch):
    monkeypatch.setattr(ob, "ocr_one_page", lambda b, p, h, n, c, **_: {
        "time": "T", "page": n, "hit": False, "tokens": 1, "ocr_seconds": 1.0, "error": None, "status": "ok", "seconds": 10.0})
    cache, log, _ = env
    out = []
    ob.run_batch([pdf], FakeBackend(TEXT), cache=cache, log=log, out=out.append, sleep=lambda _: None, cooldown=5)
    etas = [o.rsplit("ETA", 1)[1].strip() for o in out if o.startswith("[")]
    assert etas[0] == "≤0m40s"  # 3 หน้า x 10 + พัก 2 ช่วง x 5


def test_eta_not_shown_until_a_real_run_has_happened(pdf, env):
    cache, log, _ = env
    ob.run_batch([pdf], FakeBackend(TEXT), cache=cache, log=log, wanted={2}, out=lambda *_: None)
    out = []
    ob.run_batch([pdf], FakeBackend(TEXT), cache=cache, log=log, wanted={2, 3}, out=out.append)
    prog = [o for o in out if o.startswith("[")]
    assert prog[0].endswith("ETA -") and "ETA ≤" in prog[1]  # หน้า hit ไม่ใช่ข้อมูลเวลา OCR


# ---------- CUDA error ที่ context เสีย -> หยุด batch (exit 3) ----------

class AcceleratorError(RuntimeError):
    """ชื่อเดียวกับ torch.AcceleratorError (จำลอง ไม่ต้องมี GPU)"""


@pytest.mark.parametrize("exc", [
    AcceleratorError("CUDA error: an illegal memory access was encountered"),
    AcceleratorError("something opaque"),  # ชนิดเดียวก็พอ
    RuntimeError("CUDA error: device-side assert triggered"),
    RuntimeError("CUDA error: unspecified launch failure"),
    RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED when calling cublasGemmEx"),
])
def test_is_fatal_cuda_true(exc):
    assert ob.is_fatal_cuda(exc)


@pytest.mark.parametrize("exc", [
    OomError("CUDA out of memory. Tried to allocate 2.00 GiB"),
    AcceleratorError("CUDA error: out of memory"),  # OOM ที่โผล่เป็น AcceleratorError ยังเป็น OOM (ล้างแล้วลองใหม่)
    ValueError("โมเดลพัง"),
    RuntimeError("cannot identify image file"),
])
def test_is_fatal_cuda_false(exc):
    assert not ob.is_fatal_cuda(exc)


def test_fatal_cuda_stops_batch_immediately_records_page_and_writes_summary(pdf, env):
    cache, log, sp = env
    backend = Scripted({1: AcceleratorError("CUDA error: an illegal memory access was encountered")})  # หน้าที่ 2 ของคิว (หน้า 3)
    sleeps = []
    s = ob.run_batch([pdf], backend, cache=cache, log=log, summary_path=sp, out=lambda *_: None,
                     sleep=sleeps.append, cooldown=5)
    f = s["files"]["a.pdf"]
    assert backend.calls == 2  # ไม่ไปหน้า 4, 5
    assert s["fatal_cuda"] and "CudaFatal" in s["aborted"] and not s["interrupted"]
    assert (f["ok"], f["error"], f["not_reached"]) == (1, 1, 2)
    assert f["error_pages"][0]["page"] == 3 and "illegal memory access" in f["error_pages"][0]["error"]
    assert sleeps == [5]  # พักหลังหน้า 2 ปกติ แต่ไม่พักหลังหน้าที่ CUDA พัง
    row = [l.split("\t") for l in lines(log)][1]
    assert row[2] == "3" and row[7] == "error" and "illegal memory access" in row[8]
    assert json.loads(sp.read_text(encoding="utf-8"))["fatal_cuda"] is True  # summary ถูกเขียนก่อนออก


def test_fatal_cuda_is_not_retried_or_memory_freed(pdf, env):
    cache, log, _ = env
    freed = []
    backend = Scripted({0: AcceleratorError("CUDA error: an illegal memory access was encountered")})
    ob.run_batch([pdf], backend, cache=cache, log=log, out=lambda *_: None, free_memory=lambda: freed.append(1))
    assert backend.calls == 1 and freed == []


def test_fatal_cuda_after_oom_retry_also_stops(pdf, env):
    cache, log, _ = env
    backend = Scripted({0: OomError("CUDA out of memory"), 1: AcceleratorError("CUDA error: illegal memory access")})
    s = ob.run_batch([pdf], backend, cache=cache, log=log, out=lambda *_: None, free_memory=lambda: None)
    assert s["fatal_cuda"] and backend.calls == 2 and s["files"]["a.pdf"]["error"] == 1


def test_fatal_cuda_skips_remaining_files(tmp_path, env):
    cache, log, _ = env
    a = make_pdf(tmp_path / "a.pdf", [""])
    b = make_pdf(tmp_path / "b.pdf", [""])
    backend = Scripted({0: AcceleratorError("x")})
    s = ob.run_batch([a, b], backend, cache=cache, log=log, out=lambda *_: None)
    assert backend.calls == 1 and s["files"]["a.pdf"]["error"] == 1 and s["files"]["b.pdf"]["not_reached"] == 1


def test_ordinary_and_oom_page_errors_do_not_set_fatal_flag(pdf, env):
    cache, log, _ = env
    s = ob.run_batch([pdf], Scripted({0: ValueError("x"), 1: OomError("CUDA out of memory"), 2: OomError("CUDA out of memory")}),
                     cache=cache, log=log, out=lambda *_: None, free_memory=lambda: None)
    assert not s["fatal_cuda"] and s["aborted"] is None


def test_fatal_cuda_rerun_resumes_from_cache(pdf, env):
    cache, log, _ = env
    ob.run_batch([pdf], Scripted({2: AcceleratorError("CUDA error: illegal memory access")}), cache=cache, log=log,
                 out=lambda *_: None)  # หน้า 2,3 สำเร็จ · หน้า 4 พัง
    again = Scripted()
    s = ob.run_batch([pdf], again, cache=cache, log=log, out=lambda *_: None)
    assert again.calls == 2 and s["files"]["a.pdf"]["cached"] == 2 and s["files"]["a.pdf"]["ok"] == 2 and not s["fatal_cuda"]


def test_main_exit_codes(pdf, tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(ob, "DEFAULT_LOG", tmp_path / "out" / "b.log")
    monkeypatch.setattr(ob, "DEFAULT_SUMMARY", tmp_path / "out" / "s.json")
    base = ["--files", str(pdf), "--no-keep-awake"]

    monkeypatch.setattr(ob, "get_backend", lambda _n: Scripted({0: AcceleratorError("CUDA error: illegal memory access")}))
    assert ob.main(base) == 3
    assert json.loads((tmp_path / "out" / "s.json").read_text(encoding="utf-8"))["fatal_cuda"] is True

    monkeypatch.setattr(ob, "get_backend", lambda _n: Scripted({0: OcrUnavailable("ไม่พบ GPU")}))
    assert ob.main(base) == 2

    monkeypatch.setattr(ob, "get_backend", lambda _n: Scripted({0: KeyboardInterrupt()}))
    assert ob.main(base) == 130

    monkeypatch.setattr(ob, "get_backend", lambda _n: Scripted())
    assert ob.main(base) == 0
