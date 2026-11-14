"""backend OCR: Typhoon OCR 1.5 (2B) รันบนเครื่องผ่าน transformers — ต้องมี GPU CUDA

วิธีเรียกและ prompt ตาม model card ทุกอย่าง (https://huggingface.co/scb10x/typhoon-ocr1.5-2b):
``AutoModelForImageTextToText`` + ``AutoProcessor``, ภาพย่อ/ขยายให้ด้านยาว 1800 px (โมเดลฝึกที่ขนาดนี้),
``apply_chat_template`` ด้วย prompt เดียว, ``max_new_tokens=10000``, ``skip_special_tokens=True``
ที่ต่างจาก card: ปักเลข revision (MODEL_REVISION), ตรวจว่าโมเดลอยู่บน GPU ทั้งหมด และ **ปิดการสุ่ม** (``DO_SAMPLE = False``):
card เรียก generate() ตามค่าใน generation_config ของโมเดลซึ่งสุ่ม (temperature 0.7) ผล OCR เปลี่ยนทุกรอบ

ห้ามถอยไป CPU เงียบๆ: ไม่มี CUDA / โมเดลถูก offload ไป CPU -> ``OcrUnavailable``
โหลดโมเดลครั้งเดียวต่อ process (``_load``) ใช้ซ้ำทุกหน้า
dependency หนักอยู่ใน requirements-ocr.txt; import torch/transformers เมื่อโหลดจริงเท่านั้น
"""

import io
import threading
import time

from app.extraction.ocr import OcrResult, OcrUnavailable

MODEL_ID = "scb10x/typhoon-ocr1.5-2b"
MODEL_REVISION = "15b381a2d62569e6736f9c085859dff68e48608d"  # commit ของ HF repo ที่ทดสอบ (2026-06-11)
IMAGE_LONG_SIDE = 1800
MAX_NEW_TOKENS = 10000  # เพดานโทเคนที่สร้างต่อภาพ (กันลูปไม่รู้จบ); ค่าเดียวกับ model card
# ต่างจาก model card: card เรียก generate() ตามค่าใน generation_config ของโมเดล ซึ่งเป็นการสุ่ม
# (do_sample, temperature 0.7, top_k 20, top_p 0.8) ผล OCR จึงเปลี่ยนทุกรอบ เราต้องการผลที่ทำซ้ำได้ -> greedy
DO_SAMPLE = False

# prompt ตาม model card (ห้ามแก้เอง)
PROMPT = """Extract all text from the image.

Instructions:
- Only return the clean Markdown.
- Do not include any explanation or extra text.
- You must include all information on the page.

Formatting Rules:
- Tables: Render tables using <table>...</table> in clean HTML format.
- Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$).
- Images/Charts/Diagrams: Wrap any clearly defined visual areas (e.g. charts, diagrams, pictures) in:

<figure>
Describe the image's main elements (people, objects, text), note any contextual clues (place, event, culture), mention visible text and its meaning, provide deeper analysis when relevant (especially for financial charts, graphs, or documents), comment on style or architecture if relevant, then give a concise overall summary. Describe in Thai.
</figure>

- Page Numbers: Wrap page numbers in <page_number>...</page_number> (e.g., <page_number>14</page_number>).
- Checkboxes: Use ☐ for unchecked and ☑ for checked boxes."""

_lock = threading.Lock()
_loaded = None  # (model, processor)


def resize_if_needed(img, max_size: int = IMAGE_LONG_SIDE):
    """เหมือน ``resize_if_needed`` ใน model card (ด้านยาว -> max_size เมื่อมีด้านใดเกิน 300 px)"""
    from PIL import Image

    width, height = img.size
    if width > 300 or height > 300:
        if width >= height:
            scale = max_size / float(width)
            new_size = (max_size, int(height * scale))
        else:
            scale = max_size / float(height)
            new_size = (int(width * scale), max_size)
        return img.resize(new_size, Image.Resampling.LANCZOS)
    return img


def _load():
    global _loaded
    with _lock:
        if _loaded is not None:
            return _loaded
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:
            raise OcrUnavailable(f"ยังไม่ได้ติดตั้ง dependency ของ OCR (pip install -r requirements-ocr.txt): {e}") from e
        if not torch.cuda.is_available():
            raise OcrUnavailable("ไม่พบ GPU CUDA — Typhoon OCR รันบน CPU ไม่ได้ในระบบนี้ (ไม่ถอยไปใช้ CPU)")
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, dtype="auto", device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        except Exception as e:
            raise OcrUnavailable(f"โหลดโมเดล {MODEL_ID}@{MODEL_REVISION[:12]} ไม่สำเร็จ: {type(e).__name__}: {e}") from e
        devices = {str(d) for d in getattr(model, "hf_device_map", {"": "cuda"}).values()}
        if any(d in ("cpu", "disk") for d in devices):
            raise OcrUnavailable(f"โมเดลถูกวางบน {sorted(devices)} ไม่ใช่ GPU ทั้งหมด (VRAM ไม่พอ?) — ไม่ใช้ CPU")
        model.eval()
        _loaded = (model, processor)
        return _loaded


class TyphoonLocalBackend:
    name = "typhoon-local"
    model = MODEL_ID
    revision = MODEL_REVISION[:12]
    sends_data_offsite = False

    def __init__(self, do_sample: bool = DO_SAMPLE, max_new_tokens: int = MAX_NEW_TOKENS):
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens

    def cache_params(self) -> dict:
        return {"do_sample": self.do_sample, "max_new_tokens": self.max_new_tokens, "long_side": IMAGE_LONG_SIDE}

    def recognize(self, image_png: bytes) -> OcrResult:
        import torch
        from PIL import Image

        model, processor = _load()
        t0 = time.perf_counter()
        img = resize_if_needed(Image.open(io.BytesIO(image_png)).convert("RGB"))
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        try:
            with torch.inference_mode():
                if self.do_sample:  # ค่าสุ่มตาม generation_config ของโมเดล (เหมือน card)
                    generated = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                else:  # greedy: ปิดพารามิเตอร์การสุ่มด้วย เพื่อไม่ให้ transformers เตือนว่าไม่ได้ใช้
                    generated = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                               temperature=None, top_k=None, top_p=None)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        n_tokens = int(trimmed[0].shape[0])
        text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return OcrResult(text=text, backend=self.name, model=MODEL_ID, revision=self.revision,
                         seconds=time.perf_counter() - t0, tokens=n_tokens)


def unload() -> None:
    """ปล่อยโมเดล OCR ออกจาก VRAM (เรียกซ้ำได้; ยังไม่เคยโหลด = ไม่ทำอะไร) — ingest worker เรียกก่อนโหลด embedding
    (GPU อ้างอิง 8 GB: ห้ามมี OCR กับ BGE-M3 อยู่บน GPU พร้อมกัน) โหลดใหม่เองอัตโนมัติเมื่อ recognize ครั้งถัดไป"""
    global _loaded
    with _lock:
        if _loaded is None:
            return
        _loaded = None
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def peak_vram_bytes() -> int:
    import torch

    return int(torch.cuda.max_memory_allocated())
