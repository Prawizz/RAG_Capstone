"""เลือกอุปกรณ์ของ embedding และ OCR backend ตามค่าตั้ง + สภาพเครื่อง — **ห้าม fallback เงียบ**

- ``EMBEDDING_DEVICE`` = auto | cpu | cuda
  auto = cuda ถ้ามี GPU ไม่งั้น cpu · บน CPU ใช้ fp32 เสมอ (dtype ตาม ``EMBEDDING_DTYPE`` ใช้เฉพาะบน GPU)
  cuda แต่ไม่มี GPU -> ``DeviceUnavailable`` (ไม่ถอยไป CPU เอง)
- ``OCR_BACKEND`` = auto | typhoon-local | tesseract-local
  auto = typhoon-local เมื่อมี GPU, tesseract-local เมื่อไม่มี · ตั้งทับได้ (ตั้ง typhoon-local บนเครื่องไม่มี GPU = ใช้ไม่ได้
  และ backend จะโยน ``OcrUnavailable`` ตอนโหลด — ไม่สลับเป็น tesseract ให้เอง)

ทุกครั้งที่ตัดสิน log ผ่าน logger ``app.runtime``: กรณีปกติ INFO, กรณีที่ได้ของที่ด้อยกว่าเพราะเครื่องไม่มี GPU
(auto -> cpu / auto -> tesseract) หรือค่าที่ตั้งไว้ใช้ไม่ได้ = WARNING (เห็นได้แม้ไม่ได้ตั้งค่า logging)
``/v1/health/ready`` รายงานผลการตัดสินเดียวกันนี้ (``runtime``)

ตรวจ GPU ด้วย ``torch.cuda.is_available()`` (เคารพ ``CUDA_VISIBLE_DEVICES``); ไม่มี torch = ไม่มี GPU
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("app.runtime")

DEVICE_CHOICES = ("auto", "cpu", "cuda")
OCR_CHOICES = ("auto", "typhoon-local", "tesseract-local")
OCR_GPU_BACKEND = "typhoon-local"
OCR_CPU_BACKEND = "tesseract-local"


class DeviceUnavailable(RuntimeError):
    """ขออุปกรณ์ที่เครื่องนี้ไม่มี (เช่น cuda บนเครื่องไม่มี GPU)"""


def cuda_available() -> bool:
    try:
        import torch
    except Exception:  # ไม่มี torch = ไม่มีทางใช้ GPU
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@dataclass(frozen=True)
class DeviceChoice:
    requested: str  # ค่าที่ตั้ง (auto | cpu | cuda)
    device: str  # cpu | cuda
    dtype: str  # fp32 | bf16 | fp16 — dtype ที่ใช้จริง
    reason: str

    @property
    def fallback(self) -> bool:
        """auto แต่ได้ CPU เพราะไม่มี GPU"""
        return self.requested == "auto" and self.device == "cpu"


@dataclass(frozen=True)
class BackendChoice:
    requested: str  # auto | typhoon-local | tesseract-local
    name: str
    usable: bool  # False = ตั้งไว้แต่เครื่องนี้ใช้ไม่ได้ (typhoon-local ไม่มี GPU)
    reason: str

    @property
    def fallback(self) -> bool:
        return self.requested == "auto" and self.name == OCR_CPU_BACKEND


def resolve_embedding_device(requested: str, gpu_dtype: str = "bf16", *, cuda: Optional[bool] = None,
                             quiet: bool = False) -> DeviceChoice:
    """อุปกรณ์ + dtype ที่ embedding จะใช้จริง (``quiet`` = ไม่ log ใช้ตอนรายงานซ้ำ เช่น health)"""
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"EMBEDDING_DEVICE ต้องเป็น {'|'.join(DEVICE_CHOICES)} ไม่ใช่ {requested!r}")
    has_gpu = cuda_available() if cuda is None else cuda
    if requested == "cuda" and not has_gpu:
        msg = "EMBEDDING_DEVICE=cuda แต่ไม่พบ GPU CUDA (ไม่ถอยไป CPU เอง — ตั้ง auto หรือ cpu)"
        log.error("embedding device: %s", msg)
        raise DeviceUnavailable(msg)
    if requested == "cuda" or (requested == "auto" and has_gpu):
        choice = DeviceChoice(requested, "cuda", gpu_dtype,
                              "ตั้ง cuda" if requested == "cuda" else "auto: พบ GPU CUDA")
    else:
        choice = DeviceChoice(requested, "cpu", "fp32",
                              "ตั้ง cpu" if requested == "cpu" else "auto: ไม่พบ GPU CUDA -> cpu")
    if not quiet:
        level = logging.WARNING if choice.fallback else logging.INFO
        log.log(level, "embedding device: %s (%s) — %s", choice.device, choice.dtype, choice.reason)
    return choice


def resolve_ocr_backend(requested: str = "auto", *, cuda: Optional[bool] = None, quiet: bool = False) -> BackendChoice:
    """OCR backend ที่จะใช้ (ไม่สร้าง backend — แค่ตัดสินชื่อ)"""
    if requested not in OCR_CHOICES:
        raise ValueError(f"OCR_BACKEND ต้องเป็น {'|'.join(OCR_CHOICES)} ไม่ใช่ {requested!r}")
    has_gpu = cuda_available() if cuda is None else cuda
    if requested == "auto":
        if has_gpu:
            choice = BackendChoice(requested, OCR_GPU_BACKEND, True, "auto: พบ GPU CUDA")
        else:
            choice = BackendChoice(requested, OCR_CPU_BACKEND, True, "auto: ไม่พบ GPU CUDA -> tesseract-local (CPU)")
    elif requested == OCR_GPU_BACKEND and not has_gpu:
        choice = BackendChoice(requested, requested, False,
                               "ตั้ง typhoon-local แต่ไม่พบ GPU CUDA — ใช้ไม่ได้ (ไม่สลับเป็น tesseract เอง)")
    else:
        choice = BackendChoice(requested, requested, True, f"ตั้ง {requested}")
    if not quiet:
        level = logging.WARNING if (choice.fallback or not choice.usable) else logging.INFO
        log.log(level, "OCR backend: %s — %s", choice.name, choice.reason)
    return choice


# ---------- CUDA error / สถานะ GPU ของ process ----------

# ชนิด/ข้อความของ CUDA error ที่ context เสียแล้ว (ไม่ใช่ OOM ที่ล้างแล้วลองใหม่ได้)
_FATAL_CUDA_TYPES = {"AcceleratorError"}
_FATAL_CUDA_MARKERS = (
    "cuda error", "illegal memory access", "device-side assert", "unspecified launch failure",
    "cublas_status", "cudnn_status", "cuda driver error", "cuda kernel errors might be asynchronously reported",
)

_gpu_lock = threading.Lock()
_gpu_broken: Optional[str] = None


def is_oom(e: BaseException) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()


def is_fatal_cuda(e: BaseException) -> bool:
    """CUDA error ที่ใช้ context ต่อไม่ได้: AcceleratorError หรือข้อความ CUDA error อื่นที่ไม่ใช่ OOM"""
    if is_oom(e):
        return False
    msg = str(e).lower()
    return type(e).__name__ in _FATAL_CUDA_TYPES or any(m in msg for m in _FATAL_CUDA_MARKERS)


def mark_gpu_broken(reason: str) -> None:
    """CUDA context ของ process นี้ใช้ต่อไม่ได้ — จำไว้จนกว่า process จะเริ่มใหม่ (ไม่มีทางกู้ใน process เดิม)
    ingest worker หยุดหยิบงาน และ ``/v1/health/ready`` รายงานส่วนที่ใช้ GPU เป็น down"""
    global _gpu_broken
    with _gpu_lock:
        if _gpu_broken is None:
            _gpu_broken = reason
            log.error("GPU ใช้ต่อไม่ได้จนกว่าจะเริ่ม process ใหม่: %s", reason)


def gpu_broken_reason() -> Optional[str]:
    return _gpu_broken


def reset_gpu_state() -> None:
    """ใช้ใน test เท่านั้น (process จริงล้างสถานะนี้ได้ด้วยการเริ่มใหม่เท่านั้น)"""
    global _gpu_broken
    with _gpu_lock:
        _gpu_broken = None


def configure_logging(level: int = logging.INFO) -> None:
    """ให้ log ของ ``app.*`` (INFO ขึ้นไป) ออก stderr เมื่อยังไม่มีใครตั้ง — เรียกจาก API / สคริปต์"""
    root = logging.getLogger("app")
    root.setLevel(level)
    if not root.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(h)
