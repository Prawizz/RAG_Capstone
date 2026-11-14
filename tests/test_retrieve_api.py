"""POST /v1/retrieve ผ่าน TestClient — hash embedding + Qdrant ในกระบวนการ (ไม่ต้องมี docker / BGE-M3)

กล่องและสมาชิกสร้างผ่าน API จริง · เนื้อหา index ตรงผ่าน ``VectorStore.upsert_chunks`` (ingest ทั้งเส้นทดสอบใน
test_ingest_api.py) เพื่อคุมชนิด unit ครบ 4 แบบ, flags และสิทธิ์ระดับไฟล์ได้ตรงตัว
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.access import FileAcl
from app.api.deps import get_settings
from app.api.main import create_app
from app.api.routes_health import get_probe
from app.api import readiness
from app.api.schemas import ComponentState, RetrieveResponse
from app.api.services import build_services
from app.chunking import make_document_chunks
from app.config import Settings
from app.embedding import HashEmbedder, cosine_similarity
from app.models import Document, ExtractionMethod, QualityFlag, QualityReport, Unit, UnitStatus, UnitType
from app.thai.normalizer import normalize
from app.vectorstore import P_FLAGS, VectorStore, embed_chunks

KEY = "k"
ADMIN = "sysadmin"
Q = "เงินปันผลหุ้นละเท่าไร"


def hdr(user: str, departments: str = "finance", level: int = 5, trace: str | None = None) -> dict:
    h = {"X-Service-Key": KEY, "X-User-Id": user, "X-User-Departments": departments, "X-User-Security-Level": str(level)}
    if trace:
        h["X-Trace-Id"] = trace
    return h


def unit(index: int, text: str, type_: UnitType = UnitType.PAGE, position: dict | None = None, label: str | None = None,
         method: ExtractionMethod = ExtractionMethod.TEXT_LAYER, flags=()) -> Unit:
    position = position or {"page": index + 1}
    return Unit(index=index, type=type_, position=position, label=label or f"หน้า {index + 1}", status=UnitStatus.OK,
                method=method, text=text, text_hash=hashlib.sha256(text.encode()).hexdigest(),
                quality=QualityReport(flags=list(flags)))


class Env:
    def __init__(self, tmp_path, *, min_score: float = 0.0, query_embedder_factory=None, store_factory=None,
                 chunk_target_chars: int = 1000, reranker_factory=None, **settings_kw):
        self.settings = Settings(service_key=KEY, data_dir=tmp_path / "data", system_admin_users=frozenset({ADMIN}),
                                 embedding_test_mode=True, retrieve_min_score=min_score,
                                 chunk_target_chars=chunk_target_chars, **settings_kw)
        self.store: VectorStore | None = None

        def make_store(acl):
            if store_factory is not None:
                return store_factory(acl)
            self.store = VectorStore.in_memory(acl, self.settings)
            return self.store

        self.services = build_services(self.settings, vector_store_factory=make_store,
                                       embedder_factory=lambda: HashEmbedder(self.settings.vector_size),
                                       query_embedder_factory=query_embedder_factory,
                                       reranker_factory=reranker_factory)
        self.app = create_app()
        self.app.dependency_overrides[get_settings] = lambda: self.settings
        self.app.state.services = self.services
        self.client = TestClient(self.app)
        self.embedder = HashEmbedder(self.settings.vector_size)

    def box(self, box_id: str, visibility: str = "public", departments=None, level: int = 0) -> None:
        body = {"box_id": box_id, "name": f"กล่อง {box_id}", "visibility": visibility, "min_security_level": level}
        if departments is not None:
            body["departments"] = departments
        r = self.client.post("/v1/boxes", headers=hdr(ADMIN), json=body)
        assert r.status_code == 201, r.text

    def member(self, box_id: str, user: str, role: str = "viewer") -> None:
        r = self.client.put(f"/v1/boxes/{box_id}/members", headers=hdr(ADMIN), json={"user_id": user, "role": role})
        assert r.status_code == 200, r.text

    def index(self, file_id: str, box_id: str, units: list[Unit], filename: str = "เอกสาร.pdf",
              departments=None, level=None) -> Document:
        self.services.acl.register_file(FileAcl(file_id=file_id, box_id=box_id,
                                                departments=None if departments is None else frozenset(departments),
                                                min_security_level=level))
        doc = Document(file_id=file_id, content_hash=hashlib.sha256(file_id.encode()).hexdigest(), filename=filename,
                       detected_type="pdf", size_bytes=1, units=units, pipeline_version="t", normalizer_version="t")
        chunks = make_document_chunks(units, self.settings.chunk_target_chars)
        self.services.stores.get().upsert_chunks(doc, embed_chunks(chunks, self.embedder))
        return doc

    def retrieve(self, user: str = ADMIN, query: str = Q, box_id: str | None = "pub", top_k: int | None = None, **h):
        body: dict = {"query": query}
        if box_id is not None:
            body["box_id"] = box_id
        if top_k is not None:
            body["top_k"] = top_k
        return self.client.post("/v1/retrieve", headers=hdr(user, **h), json=body)


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.services.close()


TEXT_A = "บริษัทตัวอย่างจ่ายเงินปันผลหุ้นละ 1.00 บาท สำหรับผลการดำเนินงานปีตัวอย่าง"
TEXT_B = "นโยบายการจ่ายเงินปันผลของบริษัทตัวอย่าง ไม่น้อยกว่าร้อยละ 40 ของกำไรสุทธิ"
TEXT_C = "รายงานความยั่งยืนและการบริหารความเสี่ยงด้านสิ่งแวดล้อมของกลุ่มบริษัท"


def without_trace(body: dict) -> dict:
    return {k: v for k, v in body.items() if k != "trace_id"}


# ---------- source_required / ไม่มีสิทธิ์ = ไม่มีผล ----------


def test_without_box_id_is_source_required(env):
    r = env.retrieve(box_id=None, trace="t-sr")
    assert r.status_code == 200
    assert r.json() == {"status": "source_required", "box_id": None, "visibility": None, "top_score": None,
                        "results": [], "trace_id": "t-sr"}


def test_not_member_and_unknown_box_look_exactly_like_an_empty_member_box(env):
    env.box("hr", visibility="private")
    env.member("hr", "member")
    env.index("f_hr", "hr", [unit(0, TEXT_A)])  # กล่องมีเนื้อหาตรงคำถาม แต่ outsider ไม่เป็นสมาชิก
    env.box("empty", visibility="private")
    env.member("empty", "member")

    member_empty = env.retrieve("member", box_id="empty", trace="t1")
    outsider_on_empty = env.retrieve("outsider", box_id="empty", trace="t2")
    assert member_empty.status_code == outsider_on_empty.status_code == 200
    assert without_trace(member_empty.json()) == without_trace(outsider_on_empty.json())
    assert member_empty.json()["trace_id"] == "t1" and outsider_on_empty.json()["trace_id"] == "t2"
    assert without_trace(member_empty.json()) == {"status": "no_relevant_context", "box_id": "empty",
                                                  "visibility": None, "top_score": None, "results": []}

    assert env.retrieve("member", box_id="hr").json()["results"]  # สมาชิกเห็นเนื้อหา
    outsider_on_hr = env.retrieve("outsider", box_id="hr")
    assert outsider_on_hr.status_code == 200
    assert without_trace(outsider_on_hr.json()) == {**without_trace(member_empty.json()), "box_id": "hr"}

    ghost_before = env.retrieve("member", box_id="ghost")  # กล่องที่ไม่มีอยู่ ...
    env.box("ghost")
    env.member("ghost", "member")
    ghost_after = env.retrieve("member", box_id="ghost")  # ... เทียบกับกล่องเดียวกันที่มีอยู่แต่ว่าง
    assert ghost_before.status_code == ghost_after.status_code == 200
    assert without_trace(ghost_before.json()) == without_trace(ghost_after.json())


def test_removed_member_gets_the_empty_answer_immediately(env):
    env.box("pub")
    env.member("pub", "u")
    env.index("f1", "pub", [unit(0, TEXT_A)])
    assert env.retrieve("u").json()["results"]
    r = env.client.delete("/v1/boxes/pub/members/u", headers=hdr(ADMIN))
    assert r.status_code == 200
    body = env.retrieve("u").json()
    assert (body["results"], body["visibility"], body["top_score"]) == ([], None, None)


# ---------- ACL: แผนก / ระดับ / สิทธิ์ระดับไฟล์ · visibility ----------


def test_departments_levels_and_file_acl_filter_results(env):
    env.box("pub", departments=["finance", "marketing"], level=1)
    for u in ("mkt", "fin_low", "fin_high", "no_dept"):
        env.member("pub", u)
    env.index("f_all", "pub", [unit(0, TEXT_A)])
    env.index("f_fin", "pub", [unit(0, TEXT_B)], departments=["finance"])  # แคบลงเหลือแผนกเดียว
    env.index("f_lvl", "pub", [unit(0, TEXT_A + " (ฉบับลับ)")], level=4)  # ระดับสูงกว่ากล่อง

    def files(user, departments, level):
        r = env.retrieve(user, departments=departments, level=level, top_k=10)
        assert r.status_code == 200
        return sorted({x["file_id"] for x in r.json()["results"]})

    assert files("mkt", "marketing", 5) == ["f_all", "f_lvl"]
    assert files("fin_low", "finance", 1) == ["f_all", "f_fin"]
    assert files("fin_high", "finance", 4) == ["f_all", "f_fin", "f_lvl"]
    assert files("fin_high", "finance", 0) == []  # ต่ำกว่าระดับของกล่อง = ไม่เห็นอะไรเลย
    assert files("no_dept", "", 5) == []  # กล่องจำกัดแผนก ผู้ใช้ไม่สังกัดแผนกใด
    empty = env.retrieve("no_dept", departments="", level=5).json()
    assert (empty["visibility"], empty["top_score"]) == (None, None)


def test_visibility_comes_from_the_box(env):
    env.box("pub")
    env.box("priv", visibility="private")
    env.index("f_pub", "pub", [unit(0, TEXT_A)])
    env.index("f_priv", "priv", [unit(0, TEXT_A)])
    assert env.retrieve(box_id="pub").json()["visibility"] == "public"
    priv = env.retrieve(box_id="priv").json()
    assert priv["visibility"] == "private" and {r["file_id"] for r in priv["results"]} == {"f_priv"}


# ---------- รูปแบบผล: citation 4 ชนิด, chunk_id, extraction ----------


def test_citations_for_all_four_unit_types(env):
    env.box("pub")
    env.index("f_pdf", "pub", [unit(46, TEXT_A, label="หน้า 47", method=ExtractionMethod.OCR,
                                    flags=[QualityFlag.OCR_INCOMPLETE])], filename="รายงาน_ตัวอย่าง.pdf")
    env.index("f_ppt", "pub", [unit(2, TEXT_A + " สไลด์", UnitType.SLIDE, {"slide": 3}, "สไลด์ 3",
                                    ExtractionMethod.OFFICE_XML)], filename="สไลด์_ตัวอย่าง.pptx")
    env.index("f_doc", "pub", [unit(1, TEXT_A + " หมวด", UnitType.SECTION,
                                    {"heading_path": ["หมวด 1", "ข้อ 1.1"], "paragraphs": [3, 7]}, "หมวด 1 › ข้อ 1.1",
                                    ExtractionMethod.CONVERTED)], filename="นโยบาย_ตัวอย่าง.doc")
    env.index("f_xls", "pub", [unit(4, TEXT_A + " ตาราง", UnitType.SHEET_REGION, {"sheet": "ปันผล", "range": "A1:F40"},
                                    "ปันผล A1:F40", ExtractionMethod.OFFICE_XML)], filename="ตาราง_ตัวอย่าง.xlsx")
    body = env.retrieve(top_k=10).json()
    by_file = {r["file_id"]: r for r in body["results"]}
    assert set(by_file) == {"f_pdf", "f_ppt", "f_doc", "f_xls"}

    pdf = by_file["f_pdf"]
    assert pdf["chunk_id"] == "f_pdf:u0046:c000" and pdf["filename"] == "รายงาน_ตัวอย่าง.pdf"
    assert pdf["citation"] == {"unit_type": "page", "page": 47, "slide": None, "sheet": None, "cell_range": None,
                               "heading_path": None, "label": "หน้า 47", "char_start": 0, "char_end": len(TEXT_A)}
    assert pdf["extraction"] == {"method": "ocr", "flags": ["ocr_incomplete"]}
    assert pdf["text"] == TEXT_A

    ppt = by_file["f_ppt"]["citation"]
    assert (ppt["unit_type"], ppt["slide"], ppt["label"], ppt["page"]) == ("slide", 3, "สไลด์ 3", None)
    doc = by_file["f_doc"]
    assert (doc["citation"]["unit_type"], doc["citation"]["heading_path"]) == ("section", ["หมวด 1", "ข้อ 1.1"])
    assert doc["extraction"] == {"method": "converted", "flags": []}
    xls = by_file["f_xls"]["citation"]
    assert (xls["unit_type"], xls["sheet"], xls["cell_range"], xls["label"]) == ("sheet_region", "ปันผล", "A1:F40",
                                                                                "ปันผล A1:F40")
    for r in body["results"]:
        assert re.fullmatch(r"[A-Za-z0-9_]+:u\d{4}:c\d{3}", r["chunk_id"])


def test_text_is_the_chunk_without_context_line_and_offsets_point_into_the_unit(tmp_path):
    e = Env(tmp_path, chunk_target_chars=120)
    try:
        e.box("pub")
        sentences = [f"ประโยคที่ {i} ของตารางเงินปันผลรายไตรมาส ตัวเลขตัวอย่าง {i}.50 บาทต่อหุ้น" for i in range(8)]
        text = "[แถวแรกของตาราง: ไตรมาส | เงินปันผล]\n" + "\n".join(sentences)
        u = unit(0, text, UnitType.SHEET_REGION, {"sheet": "ปันผล", "range": "A1:B9"}, "ปันผล A1:B9")
        e.index("f", "pub", [u])
        results = e.retrieve(top_k=20).json()["results"]
        assert len(results) > 1
        chunk_ids = [r["chunk_id"] for r in results]
        assert len(set(chunk_ids)) == len(chunk_ids)
        for r in results:
            c = r["citation"]
            assert r["text"] == text[c["char_start"]:c["char_end"]]
        later = [r for r in results if not r["chunk_id"].endswith("c000")]
        assert later and all("[แถวแรกของตาราง" not in r["text"] for r in later)
    finally:
        e.services.close()


def test_unit_flags_are_written_to_the_payload(env):
    env.box("pub")
    env.index("f", "pub", [unit(0, TEXT_A, flags=[QualityFlag.OCR_INCOMPLETE, QualityFlag.LOW_TEXT])])
    points, _ = env.store.client.scroll(env.settings.collection_public, with_payload=True)
    assert [p.payload[P_FLAGS] for p in points] == [["ocr_incomplete", "low_text"]]


# ---------- คะแนน ----------


def test_rank_follows_search_order_and_score_is_dense_cosine(env):
    env.box("pub")
    env.index("f1", "pub", [unit(0, TEXT_A), unit(1, TEXT_B), unit(2, TEXT_C)])
    body = env.retrieve(top_k=3).json()
    RetrieveResponse.model_validate(body)
    q = env.embedder.embed_query(normalize(Q)).dense
    texts = {"f1:u0000:c000": TEXT_A, "f1:u0001:c000": TEXT_B, "f1:u0002:c000": TEXT_C}
    for r in body["results"]:
        expected = cosine_similarity(q, env.embedder.embed_query(texts[r["chunk_id"]]).dense)
        assert r["score"] == pytest.approx(expected, abs=1e-4)
    assert [r["rank"] for r in body["results"]] == [1, 2, 3]
    assert body["top_score"] == max(r["score"] for r in body["results"])

    from app.api.routes_retrieve import chunk_id

    store_order = env.store.search(env.services.query_embedder.embed(normalize(Q)),
                                   env.services.acl.principal_for(ADMIN), "pub", limit=3,
                                   mode=env.settings.search_mode)
    assert [r["chunk_id"] for r in body["results"]] == [chunk_id(h.file_id, h.unit_index, h.chunk_index)
                                                        for h in store_order.hits]  # ลำดับ = RRF ของ Qdrant


def test_below_min_score_is_no_relevant_context_but_results_are_kept(tmp_path):
    for min_score, status in ((0.99, "no_relevant_context"), (-1.0, "ok")):
        e = Env(tmp_path / str(min_score), min_score=min_score)
        try:
            e.box("pub", visibility="private")
            e.index("f1", "pub", [unit(0, TEXT_A), unit(1, TEXT_C)])
            body = e.retrieve(top_k=5).json()
            assert body["status"] == status
            assert body["results"] and body["visibility"] == "private"
            assert body["top_score"] == max(r["score"] for r in body["results"])
            assert "dropped_by_recheck" not in str(body)
        finally:
            e.services.close()


def test_query_is_normalized_before_embedding(tmp_path):
    seen: list[str] = []

    class Recording(HashEmbedder):
        def embed_query(self, text):
            seen.append(text)
            return super().embed_query(text)

    e = Env(tmp_path, query_embedder_factory=lambda: Recording())
    try:
        e.box("pub")
        raw = "  เงินปันผลปี ๒๕๖๘​ ทํางาน  "
        assert e.retrieve(query=raw).status_code == 200
        assert seen == [normalize(raw)] and "2568" in seen[0] and "ทำงาน" in seen[0]
    finally:
        e.services.close()


# ---------- 503 ----------


class DownClient:
    def __getattr__(self, name):
        def fail(*a, **k):
            raise ConnectionError("connection refused")
        return fail


def test_qdrant_down_is_503_unavailable(env):
    env.box("pub")
    env.index("f1", "pub", [unit(0, TEXT_A)])
    env.store.client = DownClient()
    r = env.retrieve(trace="t-down")
    assert r.status_code == 503
    assert r.json()["error_code"] == "unavailable" and r.json()["trace_id"] == "t-down"
    assert set(r.json()) == {"error_code", "message", "trace_id"}


def test_qdrant_unreachable_at_first_use_is_503_and_recovers(tmp_path):
    state = {"down": True}

    def store(acl):
        if state["down"]:
            raise ConnectionError("connection refused")
        return VectorStore.in_memory(acl, Settings(embedding_test_mode=True))

    e = Env(tmp_path, store_factory=store)
    try:
        e.box("pub")
        r = e.retrieve()
        assert r.status_code == 503 and r.json()["error_code"] == "unavailable"
        state["down"] = False
        assert e.retrieve().status_code == 200
    finally:
        e.services.close()


def test_query_embedder_that_cannot_load_is_503_and_ready_reports_down(tmp_path):
    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError("โหลดโมเดลไม่ได้ (ทดสอบ)")

    e = Env(tmp_path, query_embedder_factory=broken)
    try:
        e.box("pub")
        for _ in range(3):
            r = e.retrieve()
            assert r.status_code == 503 and r.json()["error_code"] == "unavailable"
        assert len(calls) == 1  # โหลดครั้งเดียว ไม่ลองซ้ำทุก request
        assert e.retrieve(box_id=None).status_code == 200  # source_required ไม่ต้อง embed

        e.app.dependency_overrides[get_probe] = lambda: (
            lambda s: readiness.probe(s, gpu=True, qdrant=lambda _s: ComponentState.OK, tesseract=lambda: (True, "")))
        ready = e.client.get("/v1/health/ready", headers={"X-Service-Key": KEY})
        assert ready.status_code == 503
        body = ready.json()
        assert body["status"] == "not_ready" and body["components"]["embedding"] == "down"
        assert any("query embedding" in n and "โหลดโมเดลไม่ได้" in n for n in body["runtime"]["notes"])
    finally:
        e.services.close()


def test_query_embedding_failure_is_503_without_the_query_in_message_or_log(tmp_path, caplog):
    secret = "คำถามลับเฉพาะ test นี้ 12345"

    class Failing(HashEmbedder):
        def embed_query(self, text):
            raise RuntimeError(f"พังตอน embed: {text}")  # error ที่แนบข้อความคำถามมาด้วย ต้องไม่หลุดออกไป

    e = Env(tmp_path, query_embedder_factory=lambda: Failing())
    try:
        e.box("pub")
        with caplog.at_level(logging.DEBUG):
            r = e.retrieve(query=secret)
        assert r.status_code == 503 and r.json()["error_code"] == "unavailable"
        assert secret not in r.text and secret not in caplog.text
    finally:
        e.services.close()


# ---------- log ----------


def test_log_has_metrics_but_never_the_query_or_chunk_text(env, caplog):
    env.box("pub")
    env.member("pub", "u")
    env.index("f1", "pub", [unit(0, TEXT_A), unit(1, TEXT_B)])
    # ด่าน 2 ทิ้ง: สิทธิ์ไฟล์แคบลงหลัง index โดยไม่ refresh payload (ด่าน 1 ยังผ่าน)
    env.services.acl.register_file(FileAcl(file_id="f1", box_id="pub", departments=frozenset({"legal"})))
    env.index("f2", "pub", [unit(0, TEXT_C)])
    query = "เงินปันผลหุ้นละเท่าไร ข้อความคำถามเฉพาะ"
    with caplog.at_level(logging.DEBUG):
        r = env.retrieve("u", query=query, trace="t-log", top_k=5)
    assert r.status_code == 200
    body = r.json()
    assert {x["file_id"] for x in body["results"]} == {"f2"}
    assert "dropped_by_recheck" not in r.text
    lines = [rec.getMessage() for rec in caplog.records if rec.name == "app.api.retrieve"]
    assert len(lines) == 1
    line = lines[0]
    for key in ("trace_id=t-log", "box_id=pub", "results=1", "dropped_by_recheck=2", f"query_chars={len(query)}",
                "embed_ms=", "search_ms=", "map_ms=", f"top_score={body['top_score']:.4f}"):
        assert key in line, (key, line)
    for text in (query, TEXT_A, TEXT_B, TEXT_C, "เงินปันผล"):
        assert text not in caplog.text


# ---------- query embedder: โหลดใน lifespan ครั้งเดียว + lock ----------


def test_query_embedder_loads_once_in_lifespan(tmp_path):
    loads = []

    def factory():
        loads.append(1)
        return HashEmbedder()

    e = Env(tmp_path, query_embedder_factory=factory)
    try:
        e.box("pub")
        with TestClient(e.app) as client:
            assert len(loads) == 1 and e.services.query_embedder.state == "ok"
            for _ in range(3):
                r = client.post("/v1/retrieve", headers=hdr(ADMIN), json={"query": Q, "box_id": "pub"})
                assert r.status_code == 200
        assert len(loads) == 1
    finally:
        e.services.close()


def test_query_inference_never_overlaps_across_threads(tmp_path):
    active = {"now": 0, "max": 0}
    guard = threading.Lock()

    class Slow(HashEmbedder):
        def embed_query(self, text):
            with guard:
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
            time.sleep(0.02)
            with guard:
                active["now"] -= 1
            return super().embed_query(text)

    e = Env(tmp_path, query_embedder_factory=lambda: Slow())
    try:
        e.box("pub")
        e.index("f1", "pub", [unit(0, TEXT_A)])
        codes: list[int] = []

        def call():
            codes.append(e.retrieve().status_code)

        threads = [threading.Thread(target=call) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert codes == [200] * 8 and active["max"] == 1
    finally:
        e.services.close()


def test_worker_and_retrieve_share_one_vector_store(tmp_path):
    """ไฟล์ที่ ingest ผ่าน worker ค้นเจอทาง /v1/retrieve (vector store ก้อนเดียวกัน)"""
    import io

    from docx import Document as DocxDocument

    e = Env(tmp_path)
    try:
        e.box("pub")
        d = DocxDocument()
        d.add_paragraph(TEXT_C)  # ส่วนต้นเอกสาร (ก่อนหัวข้อแรก)
        d.add_heading("นโยบายเงินปันผล", level=1)
        d.add_paragraph(TEXT_B * 3)
        buf = io.BytesIO()
        d.save(buf)
        r = e.client.post("/v1/ingest", headers=hdr(ADMIN), data={"box_id": "pub"},
                          files={"file": ("นโยบาย_ตัวอย่าง.docx", buf.getvalue())})
        assert r.status_code == 202, r.text
        while e.services.worker.process_next() is not None:
            pass
        body = e.retrieve(query="นโยบายการจ่ายเงินปันผล").json()
        assert body["status"] == "ok" and body["results"][0]["file_id"] == r.json()["file_id"]
        c = body["results"][0]["citation"]
        assert c["unit_type"] == "section" and c["heading_path"] == ["นโยบายเงินปันผล"]
        assert body["results"][0]["extraction"]["method"] == "office_xml"
    finally:
        e.services.close()


# ---------- Qdrant จริงจาก docker (pytest -m qdrant) ----------


@pytest.mark.qdrant
def test_retrieve_against_real_qdrant_server(tmp_path):
    """local mode ไม่ใช่โค้ดชุดเดียวกับ server: ยืนยันว่า with_vectors + RRF (prefetch) + payload flags ทำงานบน server จริง
    collection ชื่อเฉพาะของ test ลบทิ้งท้ายเสมอ"""
    from qdrant_client import QdrantClient

    base = Settings()
    try:
        probe_client = QdrantClient(url=base.qdrant_url, timeout=5, check_compatibility=False)
        probe_client.get_collections()
    except Exception as e:  # ไม่มี docker ขึ้นอยู่ -> ข้าม ไม่ใช่ fail
        pytest.skip(f"ไม่มี Qdrant ที่ {base.qdrant_url}: {e}")
    names = ("pytest_retrieve_public", "pytest_retrieve_private")
    for name in names:
        if probe_client.collection_exists(name):
            probe_client.delete_collection(name)

    def real(acl):
        s = Settings(collection_public=names[0], collection_private=names[1], embedding_test_mode=True)
        return VectorStore(QdrantClient(url=s.qdrant_url, timeout=int(s.qdrant_timeout)), acl, s)

    e = Env(tmp_path, store_factory=real)
    try:
        e.box("pub", visibility="private")
        e.index("f1", "pub", [unit(0, TEXT_A, flags=[QualityFlag.OCR_INCOMPLETE]), unit(1, TEXT_B), unit(2, TEXT_C)])
        body = e.retrieve(top_k=3).json()
        RetrieveResponse.model_validate(body)
        assert body["visibility"] == "private" and len(body["results"]) == 3
        q = e.embedder.embed_query(normalize(Q)).dense
        texts = {0: TEXT_A, 1: TEXT_B, 2: TEXT_C}
        for r in body["results"]:
            u = int(r["chunk_id"].split(":u")[1][:4])
            assert r["score"] == pytest.approx(cosine_similarity(q, e.embedder.embed_query(texts[u]).dense), abs=1e-4)
        flags = {r["chunk_id"]: r["extraction"]["flags"] for r in body["results"]}
        assert flags["f1:u0000:c000"] == ["ocr_incomplete"] and flags["f1:u0001:c000"] == []
    finally:
        e.services.close()
        for name in names:
            if probe_client.collection_exists(name):
                probe_client.delete_collection(name)
        probe_client.close()
