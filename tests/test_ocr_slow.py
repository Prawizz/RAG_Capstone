"""Test ช้า: Typhoon OCR จริงบน GPU (ข้ามถ้าไม่มี torch / GPU CUDA / โมเดลในแคช / ไฟล์ PDF)
รันด้วย ``pytest -m slow tests/test_ocr_slow.py`` — ต้องไม่มี process อื่นใช้ GPU อยู่"""

import re
from pathlib import Path

import pytest

DATASETS = Path(__file__).parent.parent / "eval" / "datasets"
CPALL = DATASETS / "E_ONE_REPORT_CPALLT.PDF"

pytestmark = pytest.mark.slow


def _skip_unless_ready():
    if not CPALL.exists():
        pytest.skip("ไม่มี E_ONE_REPORT_CPALLT.PDF")
    try:
        import torch
    except ImportError:
        pytest.skip("ไม่ได้ติดตั้ง requirements-ocr.txt")
    if not torch.cuda.is_available():
        pytest.skip("ไม่มี GPU CUDA")
    try:
        from huggingface_hub import snapshot_download

        from app.extraction.ocr_typhoon import MODEL_ID, MODEL_REVISION

        snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    except Exception:
        pytest.skip("ไม่มีโมเดล typhoon-ocr1.5-2b ในแคช")


def test_cpall_442_financial_note_page_has_numbers():
    _skip_unless_ready()
    from app.extraction.ocr import run_ocr
    from app.extraction.ocr_typhoon import TyphoonLocalBackend
    from app.extraction.pipeline import extract_file
    from app.models import ExtractionMethod, UnitStatus

    doc = extract_file(CPALL, pages=(442, 442))
    assert doc.units[0].status == UnitStatus.NEEDS_OCR  # ด่านตรวจส่งมาเอง ไม่ต้องบังคับ
    out = run_ocr(doc, CPALL, TyphoonLocalBackend())
    u = out.units[0]
    assert u.status == UnitStatus.OK and u.method == ExtractionMethod.OCR
    assert "typhoon-ocr1.5-2b@" in u.reason
    assert len(re.findall(r"[0-9]", u.text)) >= 10  # จำนวนเงิน / วันที่ในงบการเงิน
    assert "งบการเงิน" in u.text
    assert u.signals["gate:thai_chars"] > 500
