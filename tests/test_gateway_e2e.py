"""mock Gateway + RAG API ตัวจริง (ACL จริง, Qdrant ในกระบวนการ, hash embedding) ด้วย persona ของ demo

ต่างจาก test_gateway_mock.py ที่ตอบจากตัวอย่าง contract: ที่นี่ฝั่ง RAG คือโค้ดจริง จึงพิสูจน์ว่าสองฝั่งเข้ากันและสิทธิ์ไม่รั่วตลอดทาง
(ไม่ผ่าน socket: ใช้ TestClient ของ RAG เป็นตัวส่ง HTTP — ยังเป็น request/response ตาม contract ทุกตัว)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gateway_mock.budget import Budget
from gateway_mock.llm import ExtractiveLLM, LLMRouter
from gateway_mock.main import create_app
from gateway_mock.rag_client import RagClient
from tests.test_retrieve_api import ADMIN, KEY, Env, hdr, unit

PUBLIC, PRIVATE = "annual-reports", "finance-private"
Q = "เงินปันผลหุ้นละเท่าไร"


def as_user(user_id: str) -> dict:
    return {"X-Demo-User": user_id}


@pytest.fixture
def stack(tmp_path):
    env = Env(tmp_path)
    env.box(PUBLIC, "public")
    env.box(PRIVATE, "private", departments=["finance"], level=2)
    for user, box, role in [("somchai.demo", PUBLIC, "viewer"), ("wanida.demo", PUBLIC, "viewer"),
                            ("wanida.demo", PRIVATE, "viewer"), ("preecha.demo", PRIVATE, "viewer"),
                            ("anan.demo", PRIVATE, "viewer"), ("malee.demo", PRIVATE, "admin")]:
        env.member(box, user, role)
    env.index("f_pub", PUBLIC, [unit(0, "บริษัทจ่ายเงินปันผลหุ้นละ 1.00 บาท สำหรับปีล่าสุด")], filename="รายงานประจำปี.pdf")
    env.index("f_priv", PRIVATE, [unit(6, "งบการเงินภายใน เงินปันผลที่เสนอหุ้นละ 2.50 บาท")], filename="งบภายใน.pdf")

    rag = RagClient("http://rag.test", KEY)
    rag._http = env.client  # TestClient เป็น httpx.Client — ใช้ส่ง request จริงเข้าแอป RAG
    internal, external = ExtractiveLLM("internal", "internal"), ExtractiveLLM("external", "external")
    gateway = TestClient(create_app(rag=rag, router=LLMRouter(internal, external), budget=Budget(100)))
    return env, gateway


def chat(gateway, user, box_id=PRIVATE, question=Q):
    return gateway.post("/chat", headers=as_user(user), json={"question": question, "box_id": box_id}).json()


def comparable(body: dict) -> dict:
    return {k: v for k, v in body.items() if k != "trace_id"}


def test_member_with_clearance_gets_an_internal_llm_answer_with_page_citation(stack):
    _, gateway = stack
    r = chat(gateway, "wanida.demo")
    assert r["status"] == "answered" and r["visibility"] == "private"
    assert r["llm"]["location"] == "internal"
    assert r["citations"][0]["label"] == "หน้า 7" and r["citations"][0]["filename"] == "งบภายใน.pdf"
    assert "2.50" in r["answer"]


def test_public_box_answer_may_use_external_llm(stack):
    _, gateway = stack
    r = chat(gateway, "somchai.demo", PUBLIC)
    assert r["status"] == "answered" and r["visibility"] == "public" and r["llm"]["location"] == "external"
    assert r["citations"][0]["file_id"] == "f_pub"


@pytest.mark.parametrize("user", ["somchai.demo", "preecha.demo", "anan.demo"])
def test_users_without_access_get_exactly_what_a_nonexistent_box_gives(stack, user):
    """somchai = ไม่ใช่สมาชิก · preecha = ระดับต่ำเกิน · anan = แผนกไม่ตรง — ทั้งหมดต้องแยกไม่ออกจากกล่องที่ไม่มี"""
    _, gateway = stack
    denied = chat(gateway, user, PRIVATE)
    missing = chat(gateway, user, "no-such-box")
    assert denied["status"] == "not_found" and denied["answer"] is None and denied["citations"] == []
    assert denied["visibility"] is None and denied["llm"] is None
    assert comparable({**denied, "box_id": None}) == comparable({**missing, "box_id": None})


def test_private_text_never_reaches_an_external_llm(stack):
    env, gateway = stack
    ext_calls: list = []
    original = ExtractiveLLM.generate

    def spy(self, question, passages):
        if self.location == "external":
            ext_calls.append(passages)
        return original(self, question, passages)

    ExtractiveLLM.generate = spy  # type: ignore[method-assign]
    try:
        for user in ("wanida.demo", "malee.demo", "somchai.demo", "anan.demo"):
            chat(gateway, user, PRIVATE)
        assert ext_calls == []
        chat(gateway, "somchai.demo", PUBLIC)
        assert len(ext_calls) == 1  # ตัวสอดแนมทำงานจริง: public ใช้ external ได้
    finally:
        ExtractiveLLM.generate = original  # type: ignore[method-assign]


def test_choose_source_lists_only_boxes_the_user_belongs_to(stack):
    _, gateway = stack
    r = gateway.post("/chat", headers=as_user("somchai.demo"), json={"question": Q}).json()
    assert r["status"] == "choose_source" and [s["box_id"] for s in r["sources"]] == [PUBLIC]
    r = gateway.post("/chat", headers=as_user("wanida.demo"), json={"question": Q}).json()
    assert sorted(s["box_id"] for s in r["sources"]) == sorted([PUBLIC, PRIVATE])


def test_removing_a_member_takes_effect_on_the_very_next_question(stack):
    env, gateway = stack
    assert chat(gateway, "wanida.demo")["status"] == "answered"
    r = env.client.delete(f"/v1/boxes/{PRIVATE}/members/wanida.demo", headers=hdr(ADMIN))
    assert r.status_code == 200
    assert chat(gateway, "wanida.demo")["status"] == "not_found"


def test_rag_down_is_a_503_not_a_not_found(stack):
    env, gateway = stack

    def boom(*a, **k):
        raise RuntimeError("vector store ล่ม")

    env.services.stores.get().search = boom  # type: ignore[assignment]
    r = gateway.post("/chat", headers=as_user("wanida.demo"), json={"question": Q, "box_id": PRIVATE})
    assert r.status_code == 503 and r.json()["error_code"] == "unavailable"
    assert "ไม่พบ" not in r.json()["message"]
