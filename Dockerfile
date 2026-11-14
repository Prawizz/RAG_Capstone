# RAG API Service (contract v1) — image ของ API + ingest worker (process เดียว)
#
#   docker compose build api                              # TORCH_VARIANT=cpu (ค่าเริ่มต้น)
#   docker compose build --build-arg TORCH_VARIANT=cu128 api   # GPU (NVIDIA, CUDA 12.8) — ดู docker-compose.gpu.yml
#
# ในภาพมี: Python 3.10 + requirements.txt, torch ตาม TORCH_VARIANT, Tesseract (+ภาษาไทย), LibreOffice (แปลง doc/ppt/xls),
#          ฟอนต์ไทย (TLWG) ให้ LibreOffice/หน้า render ไม่เป็นกล่องสี่เหลี่ยม
# ไม่มีในภาพ: โมเดล (BGE-M3 / Typhoon OCR / reranker) — mount HF cache ที่ /hf (HF_HOME) เป็น volume,
#             ข้อมูล (DATA_DIR=/data: rag.db, files/, documents/, ocr_cache/) — volume, ค่าลับ (SERVICE_KEY ฯลฯ) — env ตอนรัน
FROM python:3.10-slim-bookworm

ARG TORCH_VARIANT=cpu
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data \
    OCR_CACHE_DIR=/data/ocr_cache \
    HF_HOME=/hf \
    TORCH_VARIANT=${TORCH_VARIANT}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-tha tesseract-ocr-eng \
        libreoffice-writer-nogui libreoffice-calc-nogui libreoffice-impress-nogui \
        fonts-thai-tlwg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# torch ก่อน requirements.txt (ไม่ให้ transformers ดึง torch จาก PyPI ที่เป็น CUDA ขนาดใหญ่)
# cu128: ใส่ torchvision + accelerate สำหรับ Typhoon OCR (requirements-ocr.txt) · cpu: torch อย่างเดียว (Typhoon รันบน CPU ไม่ได้
# — ใช้ได้เฉพาะแคช OCR เดิม; OCR หน้าใหม่บน CPU ใช้ tesseract-local)
RUN if [ "$TORCH_VARIANT" = "cu128" ]; then \
        pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
            "torch==${TORCH_VERSION}+cu128" "torchvision==${TORCHVISION_VERSION}+cu128" \
        && pip install accelerate==1.15.0 ; \
    elif [ "$TORCH_VARIANT" = "cpu" ]; then \
        pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}+cpu" ; \
    else \
        echo "TORCH_VARIANT ต้องเป็น cpu หรือ cu128 (ได้ $TORCH_VARIANT)" >&2 ; exit 1 ; \
    fi

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY contract ./contract

# ผู้ใช้ไม่ใช่ root · /data และ /hf เป็นของผู้ใช้นี้ (named volume ใหม่รับสิทธิ์จากโฟลเดอร์ในภาพ)
RUN useradd --create-home --uid 1000 rag \
    && mkdir -p /data/ocr_cache /hf \
    && chown -R rag:rag /data /hf /app
USER rag

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/v1/health/live', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
