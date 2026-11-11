"""[M7] Tests for FastAPI RAG Retrieval Endpoint (/v1/retrieve)"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.access import AclStore, Box, FileAcl, Principal, Visibility
from app.api.main import app
from app.api.routes_retrieve import get_acl_store, get_embed_service, get_vector_store_service
from app.chunking import make_chunks
from app.config import Settings
from app.embedding import HashEmbedder, SparseVector
from app.models import Document, ExtractionMethod, QualityReport, Unit, UnitStatus, UnitType
from app.vectorstore import IndexedChunk, Query, VectorStore


VECTOR_SIZE = 8
TEST_SETTINGS = Settings(
    vector_size=VECTOR_SIZE,
    collection_public="test_api_public",
    collection_private="test_api_private",
    search_mode="dense",
)
SPARSE_ONE = SparseVector((1,), (1.0,))


def dummy_vec(n: float = 1.0) -> list[float]:
    return [n] + [0.0] * (VECTOR_SIZE - 1)


from tests.test_vectorstore import make_doc as make_test_doc, make_unit as make_test_unit


class DummyTestEmbedder:
    """Mock embedder สำหรับทดสอบ API"""

    dense_size = VECTOR_SIZE

    def embed_query(self, text: str):
        from app.embedding import Embedded

        return Embedded(dense=dummy_vec(1.0), sparse=SPARSE_ONE)

    def embed_document(self, text: str):
        from app.embedding import Embedded

        return Embedded(dense=dummy_vec(1.0), sparse=SPARSE_ONE)


@pytest.fixture
def test_setup():
    acl = AclStore()
    acl.register_box(Box(box_id="public_box", visibility=Visibility.PUBLIC))
    acl.register_box(
        Box(box_id="hr_box", visibility=Visibility.PRIVATE, departments=frozenset({"hr"}))
    )
    acl.register_file(FileAcl(file_id="pub_doc.pdf", box_id="public_box"))
    acl.register_file(FileAcl(file_id="hr_doc.pdf", box_id="hr_box"))

    store = VectorStore.in_memory(acl, TEST_SETTINGS)

    # Index เอกสารตัวอย่างลง public_box
    doc_text = "ในปี 2568 บริษัทมีรายได้รวม 1,022,143 ล้านบาท เพิ่มขึ้นร้อยละ 11 จากปีก่อนหน้า"
    doc = make_test_doc("pub_doc.pdf", [make_test_unit(0, doc_text)])
    chunks = make_chunks(doc.units[0], target_chars=1000)
    indexed = [IndexedChunk(c, dummy_vec(1.0), SPARSE_ONE, SPARSE_ONE) for c in chunks]
    store.upsert_chunks(doc, indexed)

    embedder = DummyTestEmbedder()

    # Override dependencies
    app.dependency_overrides[get_acl_store] = lambda: acl
    app.dependency_overrides[get_vector_store_service] = lambda: store
    app.dependency_overrides[get_embed_service] = lambda: embedder

    client = TestClient(app)
    yield client, store, acl

    app.dependency_overrides.clear()


class TestRetrieveApi:
    def test_health_check(self, test_setup):
        client, _, _ = test_setup
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "service": "rag-api"}

    def test_retrieve_success_with_citations(self, test_setup):
        client, _, _ = test_setup
        payload = {
            "query": "ในปี 2568 บริษัทมีรายได้รวมเท่าไหร่",
            "principal": {
                "user_id": "somchai",
                "departments": ["sales"],
                "security_level": 1,
                "box_ids": ["public_box"],
            },
            "box_id": "public_box",
            "limit": 3,
            "enable_rerank": True,
            "enable_abstention": True,
        }

        resp = client.post("/v1/retrieve", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["is_abstain"] is False
        assert len(data["hits"]) > 0
        assert "S1" in data["citations"]
        assert "pub_doc.pdf" == data["hits"][0]["file_id"]
        assert "[S1]" in data["formatted_context"]

    def test_retrieve_unauthorized_box_abstains_cleanly(self, test_setup):
        client, _, _ = test_setup
        # ผู้ใช้ไม่ได้อยู่ในกล่อง hr_box (ไม่มีสิทธิ์)
        payload = {
            "query": "ข้อมูลเงินเดือนและสวัสดิการ",
            "principal": {
                "user_id": "somchai",
                "departments": ["sales"],
                "security_level": 1,
                "box_ids": ["public_box"],  # ไม่มี hr_box
            },
            "box_id": "hr_box",
            "limit": 3,
        }

        resp = client.post("/v1/retrieve", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Hard Rule 9: คืนผลลัพธ์ว่างเปล่าเสมอ ไม่ฟ้อง error ว่ากล่องมีหรือไม่มี
        assert data["is_abstain"] is True
        assert data["abstain_reason"] == "NO_EVIDENCE_FOUND"
        assert len(data["hits"]) == 0
        assert len(data["citations"]) == 0
        assert "denied_layer" not in data
        assert "dropped_by_recheck" not in data

    def test_retrieve_abstention_on_irrelevant_query(self, test_setup):
        client, _, _ = test_setup
        # คำถามที่ไม่มีความเกี่ยวข้องใดๆ กับเอกสารที่มี
        payload = {
            "query": "การทดลองนิวเคลียร์ฟิสิกส์ในหลุมดำอวกาศ",
            "principal": {
                "user_id": "somchai",
                "departments": ["sales"],
                "security_level": 1,
                "box_ids": ["public_box"],
            },
            "box_id": "public_box",
            "limit": 3,
            "enable_abstention": True,
        }

        resp = client.post("/v1/retrieve", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_abstain"] is True
        assert data["abstain_reason"] in ("INSUFFICIENT_EVIDENCE", "NO_EVIDENCE_FOUND")
