"""เลือก device ของ embedding / OCR backend ตาม config + GPU (app/runtime.py) และการรายงานผ่าน /v1/health/ready

ไม่ใช้ GPU/โมเดล/Qdrant จริง: ระบุ ``cuda=`` เอง หรือแทน ``cuda_available`` / probe ด้วยของปลอม
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import runtime
from app.api import readiness
from app.api.deps import get_settings
from app.api.main import create_app
from app.api.routes_health import get_probe
from app.api.schemas import ComponentState, ReadyResponse
from app.config import Settings, build_settings
from app.runtime import DeviceUnavailable, resolve_embedding_device, resolve_ocr_backend

KEY = "k"


def records(caplog):
    return [(r.levelno, r.getMessage()) for r in caplog.records if r.name == "app.runtime"]


# ---------- embedding device ----------


@pytest.mark.parametrize("requested, gpu, device, dtype", [
    ("auto", True, "cuda", "bf16"),
    ("auto", False, "cpu", "fp32"),
    ("cpu", True, "cpu", "fp32"),  # ตั้ง cpu = cpu แม้มี GPU
    ("cpu", False, "cpu", "fp32"),
    ("cuda", True, "cuda", "bf16"),
])
def test_embedding_device_resolution(requested, gpu, device, dtype):
    c = resolve_embedding_device(requested, "bf16", cuda=gpu, quiet=True)
    assert (c.device, c.dtype) == (device, dtype)
    assert c.fallback == (requested == "auto" and not gpu)


def test_cpu_always_uses_fp32_whatever_the_gpu_dtype():
    assert resolve_embedding_device("cpu", "fp16", cuda=True, quiet=True).dtype == "fp32"
    assert resolve_embedding_device("cuda", "fp16", cuda=True, quiet=True).dtype == "fp16"


def test_cuda_without_gpu_is_an_error_not_a_silent_cpu_fallback(caplog):
    caplog.set_level(logging.INFO, logger="app.runtime")
    with pytest.raises(DeviceUnavailable):
        resolve_embedding_device("cuda", cuda=False)
    assert any(level == logging.ERROR and "cuda" in msg for level, msg in records(caplog))


def test_every_decision_is_logged_and_cpu_fallback_is_a_warning(caplog):
    caplog.set_level(logging.INFO, logger="app.runtime")
    resolve_embedding_device("auto", cuda=True)
    resolve_embedding_device("auto", cuda=False)
    resolve_embedding_device("cpu", cuda=True)
    got = records(caplog)
    assert got[0] == (logging.INFO, "embedding device: cuda (bf16) — auto: พบ GPU CUDA")
    assert got[1] == (logging.WARNING, "embedding device: cpu (fp32) — auto: ไม่พบ GPU CUDA -> cpu")
    assert got[2] == (logging.INFO, "embedding device: cpu (fp32) — ตั้ง cpu")


def test_quiet_does_not_log(caplog):
    caplog.set_level(logging.INFO, logger="app.runtime")
    resolve_embedding_device("auto", cuda=False, quiet=True)
    resolve_ocr_backend("auto", cuda=False, quiet=True)
    assert records(caplog) == []


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError):
        resolve_embedding_device("cuda:0", cuda=True)
    with pytest.raises(ValidationError):
        Settings(embedding_device="gpu")


def test_bge_embedder_with_cuda_but_no_gpu_fails_loudly(monkeypatch):
    """ไม่โหลดโมเดล: ตัดสินอุปกรณ์ก่อน แล้วล้มด้วย EmbeddingUnavailable (ไม่ถอยไป CPU เอง)"""
    from app.embedding import BGEM3Embedder, EmbeddingUnavailable

    monkeypatch.setattr(runtime, "cuda_available", lambda: False)
    with pytest.raises(EmbeddingUnavailable, match="ไม่พบ GPU"):
        BGEM3Embedder(device="cuda")


# ---------- OCR backend ----------


@pytest.mark.parametrize("requested, gpu, name, usable, fallback", [
    ("auto", True, "typhoon-local", True, False),
    ("auto", False, "tesseract-local", True, True),
    ("tesseract-local", True, "tesseract-local", True, False),  # ตั้งทับได้
    ("typhoon-local", True, "typhoon-local", True, False),
    ("typhoon-local", False, "typhoon-local", False, False),  # ไม่สลับเป็น tesseract เอง
])
def test_ocr_backend_resolution(requested, gpu, name, usable, fallback):
    c = resolve_ocr_backend(requested, cuda=gpu, quiet=True)
    assert (c.name, c.usable, c.fallback) == (name, usable, fallback)


def test_ocr_fallback_and_unusable_choice_are_warnings(caplog):
    caplog.set_level(logging.INFO, logger="app.runtime")
    resolve_ocr_backend("auto", cuda=True)
    resolve_ocr_backend("auto", cuda=False)
    resolve_ocr_backend("typhoon-local", cuda=False)
    assert [lvl for lvl, _ in records(caplog)] == [logging.INFO, logging.WARNING, logging.WARNING]
    assert "tesseract-local" in records(caplog)[1][1]


def test_ocr_backend_comes_from_settings_and_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OCR_BACKEND", "tesseract-local")
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    s = build_settings(tmp_path / "none.env")
    assert (s.ocr_backend, s.embedding_device) == ("tesseract-local", "cpu")
    monkeypatch.setenv("OCR_BACKEND", "cloud-ocr")
    with pytest.raises(ValidationError):
        build_settings(tmp_path / "none.env")
    assert Settings().ocr_backend == "auto"


def test_get_backend_without_name_follows_auto(monkeypatch, caplog):
    from app.extraction import ocr as ocr_mod
    from app.extraction import ocr_tesseract

    class FakeTesseract:
        name = "tesseract-local"

    monkeypatch.setattr(ocr_tesseract, "TesseractLocalBackend", FakeTesseract)
    monkeypatch.setattr(runtime, "cuda_available", lambda: False)
    caplog.set_level(logging.INFO, logger="app.runtime")
    assert ocr_mod.get_backend(settings=Settings()).name == "tesseract-local"
    assert records(caplog) == [(logging.WARNING, "OCR backend: tesseract-local — auto: ไม่พบ GPU CUDA -> tesseract-local (CPU)")]


# ---------- /v1/health/ready ----------


def fake_probe(gpu, *, qdrant=ComponentState.OK, tesseract=(True, "")):
    return lambda settings: readiness.probe(settings, gpu=gpu, qdrant=lambda _s: qdrant, tesseract=lambda: tesseract)


def ready(settings: Settings, probe):
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_probe] = lambda: probe
    r = TestClient(app).get("/v1/health/ready", headers={"X-Service-Key": KEY})
    ReadyResponse.model_validate(r.json())
    assert r.headers["X-Trace-Id"] == r.json()["trace_id"]
    return r.status_code, r.json()


def test_ready_on_gpu_machine():
    code, body = ready(Settings(service_key=KEY), fake_probe(True))
    assert code == 200 and body["status"] == "ready"
    assert body["runtime"] == {"gpu_available": True, "embedding_requested": "auto", "embedding_device": "cuda",
                               "embedding_dtype": "bf16", "ocr_requested": "auto", "ocr_backend": "typhoon-local",
                               "notes": []}


def test_ready_on_cpu_machine_reports_the_fallbacks():
    code, body = ready(Settings(service_key=KEY), fake_probe(False))
    assert code == 200 and body["status"] == "ready"
    rt = body["runtime"]
    assert (rt["embedding_device"], rt["embedding_dtype"], rt["ocr_backend"]) == ("cpu", "fp32", "tesseract-local")
    assert len(rt["notes"]) == 2 and all("ไม่พบ GPU" in n for n in rt["notes"])


def test_ready_degraded_when_forced_typhoon_has_no_gpu():
    code, body = ready(Settings(service_key=KEY, ocr_backend="typhoon-local"), fake_probe(False))
    assert code == 503 and body["status"] == "degraded" and body["components"]["ocr"] == "down"
    assert body["runtime"]["ocr_backend"] == "typhoon-local"


def test_ready_degraded_when_tesseract_is_missing():
    code, body = ready(Settings(service_key=KEY), fake_probe(False, tesseract=(False, "ไม่พบโปรแกรม tesseract")))
    assert code == 503 and body["status"] == "degraded"
    assert any("ไม่พบโปรแกรม tesseract" in n for n in body["runtime"]["notes"])


def test_ready_not_ready_when_cuda_is_forced_without_gpu():
    code, body = ready(Settings(service_key=KEY, embedding_device="cuda"), fake_probe(False))
    assert code == 503 and body["status"] == "not_ready" and body["components"]["embedding"] == "down"
    assert body["runtime"]["embedding_device"] is None


def test_ready_not_ready_when_qdrant_is_down():
    code, body = ready(Settings(service_key=KEY), fake_probe(True, qdrant=ComponentState.DOWN))
    assert code == 503 and body["status"] == "not_ready"


def test_ready_flags_hash_embedding_test_mode():
    code, body = ready(Settings(service_key=KEY, embedding_test_mode=True), fake_probe(True))
    assert code == 503 and body["components"]["embedding"] == "degraded"


def test_api_startup_logs_device_and_backend(monkeypatch, caplog):
    monkeypatch.setattr(runtime, "cuda_available", lambda: False)
    caplog.set_level(logging.INFO, logger="app.runtime")
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(service_key=KEY)
    with TestClient(app):
        pass
    msgs = [m for _, m in records(caplog)]
    assert any(m.startswith("embedding device: cpu (fp32)") for m in msgs)
    assert any(m.startswith("OCR backend: tesseract-local") for m in msgs)
