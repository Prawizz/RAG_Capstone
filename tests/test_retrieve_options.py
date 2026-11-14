"""ส่วนเสริมของ /v1/retrieve (ปิดเป็นค่าเริ่มต้น): RERANKER (cross-encoder) · SALIENCE_RERANK / QUERY_EXPANSION (โมดูลของเพื่อน)

cross-encoder จริงไม่โหลดใน test (2 GB) — ใช้ scorer ปลอมที่บันทึกสิ่งที่ได้รับ · โมดูลของเพื่อนเรียกของจริง (ไม่แก้ตรรกะ)
"""

from __future__ import annotations

import logging
import re

import pytest
from pydantic import ValidationError

from app.config import Settings, build_settings
from app.embedding import HashEmbedder
from app.query_expansion import expand_query
from app.rerank import rerank_hits
from app.thai.normalizer import normalize
from app.vectorstore import embed_query
from tests.test_retrieve_api import ADMIN, Env, unit

TEXTS = [f"เอกสารหน้า {i} เรื่องเงินปันผลและผลประกอบการ ปี 256{i % 10} รายละเอียดหมายเลข {i}" for i in range(25)]


class FakeScorer:
    def __init__(self, scores=None, fail: bool = False) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.scores = scores
        self.fail = fail

    def score(self, query, passages):
        if self.fail:
            raise RuntimeError(f"boom {query}")
        self.calls.append((query, list(passages)))
        if self.scores is not None:
            return self.scores(passages)
        return [int(re.search(r"หมายเลข (\d+)", p).group(1)) / 100 for p in passages]  # เลขมาก = คะแนนสูง

    def release(self):
        pass


def make(tmp_path, **kw) -> Env:
    e = Env(tmp_path, **kw)
    e.box("pub")
    e.index("f1", "pub", [unit(i, t) for i, t in enumerate(TEXTS)])
    return e


def search_hits(e: Env, query: str, limit: int):
    q = embed_query(normalize(query), HashEmbedder(e.settings.vector_size))
    principal = e.services.acl.principal_for(ADMIN, ["finance"], 5)
    return list(e.services.stores.get().search(q, principal, "pub", limit=limit, mode=e.settings.search_mode,
                                               with_dense=True).hits)


def unit_of(result: dict) -> int:
    return int(result["chunk_id"].split(":u")[1][:4])


def test_defaults_are_off_and_log_says_so(tmp_path, caplog):
    s = Settings()
    assert (s.reranker, s.reranker_device, s.query_expansion, s.salience_rerank) == ("off", "cpu", False, False)
    e = make(tmp_path)
    try:
        with caplog.at_level(logging.INFO, logger="app.api.retrieve"):
            body = e.retrieve(top_k=5).json()
        assert [unit_of(r) for r in body["results"]] == [h.unit_index for h in search_hits(e, "เงินปันผลหุ้นละเท่าไร", 5)]
        line = next(r.getMessage() for r in caplog.records if "retrieve trace_id" in r.getMessage())
        assert "rerank_top=- extras=-" in line
        assert e.services.reranker.state == "off"
    finally:
        e.services.close()


def test_cross_encoder_reranks_the_top_20_and_keeps_dense_cosine_as_score(tmp_path, caplog):
    scorer = FakeScorer()
    e = make(tmp_path, reranker="bge-v2-m3", reranker_factory=lambda: scorer)
    try:
        query = "  เงินปันผล  หุ้นละ ๕ บาท "
        with caplog.at_level(logging.INFO, logger="app.api.retrieve"):
            body = e.retrieve(query=query, top_k=5).json()
        [(got_query, passages)] = scorer.calls
        assert got_query == normalize(query)
        candidates = search_hits(e, query, 20)
        assert len(passages) == 20 == len(candidates)
        assert passages == [f"{h.context}\n{h.text}" if h.context else h.text for h in candidates]
        expected = sorted(candidates, key=lambda h: -h.unit_index)[:5]  # FakeScorer: เลขมาก = คะแนนสูง
        assert [unit_of(r) for r in body["results"]] == [h.unit_index for h in expected]
        assert [r["rank"] for r in body["results"]] == [1, 2, 3, 4, 5]
        q = embed_query(normalize(query), HashEmbedder(e.settings.vector_size)).dense
        from app.api.routes_retrieve import dense_cosine
        for r, h in zip(body["results"], expected):
            assert r["score"] == pytest.approx(round(dense_cosine(q, h.dense), 4))  # score ความหมายเดิม
        assert body["top_score"] == max(r["score"] for r in body["results"])
        line = next(r.getMessage() for r in caplog.records if "retrieve trace_id" in r.getMessage())
        assert f"rerank_top={max(expected, key=lambda h: h.unit_index).unit_index / 100:.4f}" in line
        assert "extras=reranker:bge-v2-m3" in line and query.strip() not in line
    finally:
        e.services.close()


def test_rerank_ties_keep_the_rrf_order(tmp_path):
    e = make(tmp_path, reranker="bge-v2-m3", reranker_factory=lambda: FakeScorer(scores=lambda ps: [0.5] * len(ps)))
    try:
        body = e.retrieve(top_k=7).json()
        assert [unit_of(r) for r in body["results"]] == [h.unit_index for h in search_hits(e, "เงินปันผลหุ้นละเท่าไร", 20)[:7]]
    finally:
        e.services.close()


def test_reranker_that_cannot_load_is_503_and_not_ready(tmp_path):
    loads = []

    def broken():
        loads.append(1)
        raise OSError("ไม่มีไฟล์โมเดล")

    e = make(tmp_path, reranker="bge-v2-m3", reranker_factory=broken)
    try:
        for _ in range(2):
            r = e.retrieve()
            assert r.status_code == 503 and r.json()["error_code"] == "unavailable"
            assert r.json()["message"] == "reranker ใช้ไม่ได้ชั่วคราว"
        assert loads == [1]  # ไม่ลองโหลดซ้ำทุก request
        from app.api import readiness
        ready = readiness.with_reranker(readiness.probe(e.settings, gpu=False, qdrant=lambda s: "ok",
                                                        tesseract=lambda: (True, "")), e.services.reranker)
        assert ready.status.value == "not_ready" and any("reranker" in n and "ไม่มีไฟล์โมเดล" in n for n in ready.notes)
    finally:
        e.services.close()


def test_reranker_inference_error_is_503_without_leaking_the_query(tmp_path, caplog):
    e = make(tmp_path, reranker="bge-v2-m3", reranker_factory=lambda: FakeScorer(fail=True))
    try:
        with caplog.at_level(logging.INFO):
            r = e.retrieve(query="คำถามลับเฉพาะ")
        assert r.status_code == 503
        assert "คำถามลับเฉพาะ" not in r.text and "คำถามลับเฉพาะ" not in caplog.text
    finally:
        e.services.close()


def test_two_rerankers_at_once_is_a_config_error(monkeypatch):
    with pytest.raises(ValidationError, match="พร้อมกันไม่ได้"):
        Settings(reranker="bge-v2-m3", salience_rerank=True)
    for k, v in {"RERANKER": "bge-v2-m3", "RERANKER_DEVICE": "cuda", "QUERY_EXPANSION": "1"}.items():
        monkeypatch.setenv(k, v)
    s = build_settings(env_file="does-not-exist.env")
    assert (s.reranker, s.reranker_device, s.query_expansion, s.salience_rerank) == ("bge-v2-m3", "cuda", True, False)
    with pytest.raises(ValidationError):
        Settings(reranker="colbert")


def test_salience_rerank_uses_the_friends_module_unchanged(tmp_path):
    e = make(tmp_path, salience_rerank=True)
    try:
        query = "เงินปันผล ปี 2563 หมายเลข 13"
        body = e.retrieve(query=query, top_k=5).json()
        expected = [r.hit.unit_index for r in rerank_hits(query, search_hits(e, query, 20))[:5]]
        assert [unit_of(r) for r in body["results"]] == expected
    finally:
        e.services.close()


def test_query_expansion_embeds_the_expanded_text(tmp_path):
    seen: list[str] = []

    class Recording(HashEmbedder):
        def embed_query(self, text):
            seen.append(text)
            return super().embed_query(text)

    e = make(tmp_path, query_expansion=True, query_embedder_factory=lambda: Recording(1024))
    try:
        query = "EBITDA ปี 2567"
        assert e.retrieve(query=query).status_code == 200
        assert seen == [normalize(expand_query(query).expanded_text)]
        assert "2024" in seen[0]  # พ.ศ. -> ค.ศ. จากโมดูลของเพื่อน
    finally:
        e.services.close()


def test_friend_modules_are_not_touched_when_off(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("ไม่ควรถูกเรียกเมื่อปิด")

    monkeypatch.setattr("app.api.routes_retrieve.expand_query", boom)
    monkeypatch.setattr("app.api.routes_retrieve.rerank_hits", boom)
    e = make(tmp_path)
    try:
        assert e.retrieve().status_code == 200
    finally:
        e.services.close()


# ---------- ตัววัด: ห้ามวัดโมดูลของเพื่อนบนชุด dev ----------


def test_eval_refuses_friend_modules_on_the_dev_set(tmp_path):
    from scripts.retrieval_api_eval import parse_api_log, refuse_friend_modules_on_dev

    log = tmp_path / "api.log"
    log.write_text(
        "INFO retrieve trace_id=t1 box_id=b status=ok results=1 top_score=0.5 dropped_by_recheck=0 top_k=10 "
        "query_chars=3 text_chars=9 embed_ms=50.0 search_ms=20.0 map_ms=1.0 rerank_ms=900.5 rerank_top=0.9123 "
        "extras=reranker:bge-v2-m3\n"
        "INFO retrieve trace_id=t2 box_id=b status=ok results=1 top_score=0.5 dropped_by_recheck=0 top_k=10 "
        "query_chars=3 text_chars=9 embed_ms=50.0 search_ms=20.0 map_ms=1.0 rerank_ms=3.0 rerank_top=0.4 "
        "extras=query_expansion,salience_rerank\n", encoding="utf-8")
    parsed = parse_api_log(log, {"t1", "t2"})
    assert parsed["t1"]["rerank_ms"] == 900.5 and parsed["t1"]["rerank_top"] == 0.9123
    refuse_friend_modules_on_dev(Settings(reranker="bge-v2-m3"), {"t1": parsed["t1"]})  # cross-encoder วัดบน dev ได้
    with pytest.raises(SystemExit, match="query_expansion"):
        refuse_friend_modules_on_dev(Settings(), parsed)
    with pytest.raises(SystemExit, match="salience_rerank"):
        refuse_friend_modules_on_dev(Settings(salience_rerank=True), None)


def test_non_finite_reranker_scores_are_an_error_not_a_ranking(tmp_path):
    e = make(tmp_path, reranker="bge-v2-m3",
             reranker_factory=lambda: FakeScorer(scores=lambda ps: [float("nan")] + [0.5] * (len(ps) - 1)))
    try:
        r = e.retrieve()
        assert r.status_code == 503 and r.json()["error_code"] == "unavailable"
    finally:
        e.services.close()
