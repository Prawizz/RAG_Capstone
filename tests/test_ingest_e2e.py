"""ครบเส้นของจริง (slow): API + worker thread + extract จริง + BGE-M3 จริง + Qdrant จริง (docker compose)

    docker compose up -d qdrant
    pytest -m slow tests/test_ingest_e2e.py

สร้างกล่อง -> ingest DOCX เล็กใน eval/datasets/files -> poll /v1/jobs จน done -> มี point ของไฟล์ใน Qdrant และ
/v1/sources/{box}/files เห็นไฟล์ · ใช้ collection ชื่อเฉพาะของ test และลบทิ้งตอนจบ · DATA_DIR = tmp
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from app.api.deps import get_settings
from app.api.main import create_app
from app.config import build_settings

pytestmark = pytest.mark.slow

DOCX = Path(__file__).resolve().parents[1] / "eval" / "datasets" / "files" / "thesis_manual_th.docx"
KEY = "e2e-key"
POLL_TIMEOUT_S = 900


def test_ingest_docx_end_to_end_with_real_qdrant_and_bge_m3(tmp_path):
    if not DOCX.is_file():
        pytest.skip(f"ไม่มีไฟล์ทดสอบ {DOCX}")
    base = build_settings()
    client = QdrantClient(url=base.qdrant_url, api_key=base.qdrant_api_key, timeout=5, check_compatibility=False)
    try:
        client.get_collections()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ไม่มี Qdrant ที่ {base.qdrant_url} (docker compose up -d qdrant): {e}")
    suffix = uuid.uuid4().hex[:8]
    settings = base.model_copy(update={
        "service_key": KEY, "data_dir": tmp_path / "data", "system_admin_users": frozenset({"admin"}),
        "embedding_test_mode": False,  # BGE-M3 จริง (conftest เปิดโหมด test ผ่าน env — ที่นี่ปิดชัดเจน)
        "collection_public": f"e2e_public_{suffix}", "collection_private": f"e2e_private_{suffix}",
    })
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    h = {"X-Service-Key": KEY, "X-User-Id": "admin", "X-User-Departments": "", "X-User-Security-Level": "0"}
    try:
        with TestClient(app) as api:  # lifespan: worker thread ทำงานจริง
            r = api.post("/v1/boxes", headers=h, json={"box_id": "e2e", "name": "กล่องทดสอบครบเส้น",
                                                       "visibility": "private"})
            assert r.status_code == 201, r.text
            r = api.post("/v1/ingest", headers=h, data={"box_id": "e2e"},
                         files={"file": (DOCX.name, DOCX.read_bytes())})
            assert r.status_code == 202, r.text
            job_id, file_id = r.json()["job_id"], r.json()["file_id"]

            t0, statuses, job = time.time(), [], None
            while time.time() - t0 < POLL_TIMEOUT_S:
                job = api.get(f"/v1/jobs/{job_id}", headers=h).json()
                if not statuses or statuses[-1] != job["status"]:
                    statuses.append(job["status"])
                if job["status"] in ("done", "done_with_gaps", "failed"):
                    break
                time.sleep(0.5)
            elapsed = time.time() - t0
            print(f"\nstatuses={statuses} elapsed={elapsed:.1f}s progress={job['progress']} units={job['units']}")
            assert job["status"] in ("done", "done_with_gaps"), job
            assert job["error"] is None and job["units"]["ok"] > 0

            flt = qm.Filter(must=[qm.FieldCondition(key="file_id", match=qm.MatchValue(value=file_id))])
            points = client.count(settings.collection_private, count_filter=flt, exact=True).count
            public = client.count(settings.collection_public, count_filter=flt, exact=True).count
            print(f"points: private={points} public={public}")
            assert points > 0 and public == 0  # กล่อง private -> collection private เท่านั้น

            files = api.get("/v1/sources/e2e/files", headers=h).json()["files"]
            assert [(f["file_id"], f["filename"], f["detected_type"]) for f in files] == [(file_id, DOCX.name, "docx")]
            assert files[0]["status"] == job["status"] and files[0]["ingested_at"]
            sources = api.get("/v1/sources", headers=h).json()["sources"]
            assert [(s["box_id"], s["file_count"], s["role"]) for s in sources] == [("e2e", 1, "owner")]
    finally:
        for name in (settings.collection_public, settings.collection_private):
            if client.collection_exists(name):
                client.delete_collection(name)
        client.close()
