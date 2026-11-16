"""mock Gateway: ทำตาม docs/API_CONTRACT.md ฝั่ง Gateway — ตัวอย่างตอบกลับของ RAG ใช้ไฟล์ใน contract/examples (ไม่ได้สร้างจากโค้ดเรา)"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway_mock.budget import Budget
from gateway_mock.llm import ExtractiveLLM, LLMRouter
from gateway_mock.main import create_app
from gateway_mock.rag_client import RagClient
from gateway_mock.users import DEMO_USERS

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "contract" / "examples"
KEY = "gw-test-key"
WANIDA = {"X-Demo-User": "wanida.demo"}


def example(name: str) -> dict:
    return json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))


class SpyLLM(ExtractiveLLM):
    def __init__(self, name: str, location: str) -> None:
        super().__init__(name, location)  # type: ignore[arg-type]
        self.calls = 0

    def generate(self, question, passages):
        self.calls += 1
        return super().generate(question, passages)


class FakeRag:
    """ตัวแทน RAG ที่ตอบจากตัวอย่างใน contract และจดทุก request ที่ Gateway ส่งมา"""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.retrieve_reply: tuple[int, dict] = (200, example("retrieve_ok"))
        self.sources_reply: tuple[int, dict] = (200, example("sources_list"))
        self.raise_error: Exception | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raise_error:
            raise self.raise_error
        status, body = self.retrieve_reply if request.url.path == "/v1/retrieve" else self.sources_reply
        return httpx.Response(status, json=body)


@pytest.fixture
def rag() -> FakeRag:
    return FakeRag()


@pytest.fixture
def internal() -> SpyLLM:
    return SpyLLM("internal-spy", "internal")


@pytest.fixture
def external() -> SpyLLM:
    return SpyLLM("external-spy", "external")


@pytest.fixture
def client(rag, internal, external) -> TestClient:
    app = create_app(rag=RagClient("http://rag.test", KEY, transport=httpx.MockTransport(rag)),
                     router=LLMRouter(internal, external), budget=Budget(50))
    return TestClient(app)


def ask(client, question="ปันผลหุ้นละเท่าไร", headers=WANIDA, **body):
    return client.post("/chat", headers=headers, json={"question": question, "box_id": "example-finance", **body})


def last_retrieve(rag: FakeRag) -> httpx.Request:
    return [r for r in rag.requests if r.url.path == "/v1/retrieve"][-1]


# ---------- identity: Gateway เป็นคนถือ ----------

def test_gateway_sends_identity_from_its_directory_not_from_the_caller(client, rag):
    r = ask(client, headers={**WANIDA, "X-User-Id": "malee.demo", "X-User-Security-Level": "5",
                             "X-User-Departments": "everything", "X-Service-Key": "attacker"})
    assert r.status_code == 200
    sent = last_retrieve(rag).headers
    assert sent["x-service-key"] == KEY
    assert (sent["x-user-id"], sent["x-user-departments"], sent["x-user-security-level"]) == ("wanida.demo", "finance", "3")


@pytest.mark.parametrize("headers", [{}, {"X-Demo-User": "nobody"}, {"X-Demo-User": ""}, {"x-user-id": "wanida.demo"}])
def test_unauthenticated_requests_get_401_and_never_reach_rag(client, rag, headers):
    r = ask(client, headers=headers)
    assert r.status_code == 401 and r.json()["error_code"] == "unauthorized"
    assert rag.requests == []


def test_trace_id_is_forwarded_and_echoed(client, rag):
    rag.retrieve_reply = (200, {**example("retrieve_ok"), "trace_id": "abc123"})
    ask(client, headers={**WANIDA, "X-Trace-Id": "abc123"})
    assert last_retrieve(rag).headers["x-trace-id"] == "abc123"
    assert ask(client, headers={**WANIDA, "X-Trace-Id": "abc123"}).json()["trace_id"] == "abc123"


def test_request_body_to_rag_matches_the_contract(client, rag):
    ask(client, top_k=7)
    assert json.loads(last_retrieve(rag).content) == {"query": "ปันผลหุ้นละเท่าไร", "box_id": "example-finance", "top_k": 7}
    client.post("/chat", headers=WANIDA, json={"question": "x"})
    assert "box_id" not in json.loads(last_retrieve(rag).content)  # ไม่ระบุกล่อง = ไม่ส่ง box_id ให้ RAG ตัดสิน source_required


# ---------- status: ok -> เลือก LLM ตาม visibility ----------

def test_private_box_is_answered_by_the_internal_llm_only(client, internal, external):
    r = ask(client).json()
    assert r["status"] == "answered" and r["visibility"] == "private"
    assert r["llm"] == {"name": "internal-spy", "location": "internal"}
    assert (internal.calls, external.calls) == (1, 0)


def test_public_box_may_use_the_external_llm(client, rag, internal, external):
    rag.retrieve_reply = (200, {**example("retrieve_ok"), "visibility": "public"})
    r = ask(client).json()
    assert r["llm"]["location"] == "external" and (internal.calls, external.calls) == (0, 1)


@pytest.mark.parametrize("visibility", [None, "", "PUBLIC", "internal", "unknown"])
def test_anything_not_exactly_public_fails_closed_to_internal(client, rag, internal, external, visibility):
    rag.retrieve_reply = (200, {**example("retrieve_ok"), "visibility": visibility})
    r = ask(client).json()
    assert r["llm"]["location"] == "internal" and external.calls == 0 and r["visibility"] is None


def test_router_rejects_swapped_backends():
    with pytest.raises(ValueError):
        LLMRouter(ExtractiveLLM("a", "external"), ExtractiveLLM("b", "external"))
    with pytest.raises(ValueError):
        LLMRouter(ExtractiveLLM("a", "internal"), ExtractiveLLM("b", "internal"))


def test_answer_carries_citations_from_the_results(client):
    r = ask(client).json()
    ok = example("retrieve_ok")["results"]
    assert [c["label"] for c in r["citations"]] == [x["citation"]["label"] for x in ok] == ["หน้า 3", "หมวดตัวอย่าง 1 › หัวข้อตัวอย่าง 1.1"]
    assert [c["file_id"] for c in r["citations"]] == [x["file_id"] for x in ok]
    assert [c["n"] for c in r["citations"]] == [1, 2] and "[1]" in r["answer"]


def test_ocr_incomplete_flag_is_surfaced_in_the_answer(client, rag):
    body = copy.deepcopy(example("retrieve_ok"))
    body["results"][0]["extraction"]["flags"] = ["ocr_incomplete"]
    rag.retrieve_reply = (200, body)
    r = ask(client).json()
    assert "OCR ที่อาจอ่านไม่ครบ" in r["answer"] and r["citations"][0]["flags"] == ["ocr_incomplete"]


# ---------- status อื่น ----------

def test_source_required_lists_only_the_users_own_boxes_and_skips_the_llm(client, rag, internal, external):
    rag.retrieve_reply = (200, example("retrieve_source_required"))
    r = client.post("/chat", headers=WANIDA, json={"question": "ปันผล"}).json()
    assert r["status"] == "choose_source" and r["answer"] is None and r["citations"] == []
    assert [s["box_id"] for s in r["sources"]] == [s["box_id"] for s in example("sources_list")["sources"]]
    assert (internal.calls, external.calls) == (0, 0)


def test_no_relevant_context_refuses_without_calling_an_llm(client, rag, internal, external):
    rag.retrieve_reply = (200, example("retrieve_no_relevant_context"))
    r = ask(client).json()
    assert r["status"] == "not_found" and r["answer"] is None and r["citations"] == []  # แม้ RAG ส่ง results คะแนนต่ำมา
    assert r["visibility"] is None and r["llm"] is None  # ตัวอย่างนี้ RAG ส่ง visibility=private มา — ต้องไม่หลุดไปกับ not_found
    assert (internal.calls, external.calls) == (0, 0)


def test_unknown_status_is_treated_like_no_relevant_context(client, rag, internal):
    rag.retrieve_reply = (200, {**example("retrieve_ok"), "status": "source_suggested"})
    r = ask(client).json()
    assert r["status"] == "not_found" and internal.calls == 0


def test_non_member_and_empty_box_are_indistinguishable_to_the_user(client, rag):
    """กฎข้อ 9: ผู้ใช้ต้องแยกไม่ได้ว่ากล่องไม่มี/ไม่มีสิทธิ์ หรือเป็นสมาชิกแต่กล่องว่าง"""
    rag.retrieve_reply = (200, example("retrieve_not_member_box"))
    not_member = ask(client).json()
    rag.retrieve_reply = (200, example("retrieve_member_empty_box"))
    empty = ask(client).json()
    for body in (not_member, empty):
        body.pop("trace_id")
    assert not_member == empty
    assert not_member["visibility"] is None and not_member["llm"] is None


# ---------- RAG ใช้ไม่ได้ ----------

@pytest.mark.parametrize("failure", [
    lambda rag: setattr(rag, "retrieve_reply", (503, example("error_unavailable"))),
    lambda rag: setattr(rag, "retrieve_reply", (500, example("error_internal"))),
    lambda rag: setattr(rag, "raise_error", httpx.ConnectError("refused")),
    lambda rag: setattr(rag, "raise_error", httpx.ReadTimeout("slow")),
])
def test_rag_outage_is_503_and_never_reported_as_not_found(client, rag, internal, external, failure):
    failure(rag)
    r = ask(client)
    assert r.status_code == 503 and r.json()["error_code"] == "unavailable"
    assert "ไม่พบ" not in r.json()["message"] and (internal.calls, external.calls) == (0, 0)


def test_rejected_service_key_is_a_gateway_config_error_not_the_users_fault(client, rag):
    rag.retrieve_reply = (401, example("error_unauthorized"))
    r = ask(client)
    assert r.status_code == 502 and r.json()["error_code"] == "upstream_misconfigured"


def test_rag_validation_error_is_passed_on_as_422(client, rag):
    rag.retrieve_reply = (422, example("error_invalid_request"))
    assert ask(client).status_code == 422


# ---------- งบประมาณ + validation + endpoint อื่น ----------

def test_budget_is_per_user_and_checked_before_calling_rag(rag, internal, external):
    app = create_app(rag=RagClient("http://rag.test", KEY, transport=httpx.MockTransport(rag)),
                     router=LLMRouter(internal, external), budget=Budget(2))
    c = TestClient(app)
    assert [ask(c).status_code for _ in range(2)] == [200, 200]
    before = len(rag.requests)
    r = ask(c)
    assert r.status_code == 429 and r.json()["error_code"] == "budget_exceeded" and len(rag.requests) == before
    assert ask(c, headers={"X-Demo-User": "malee.demo"}).status_code == 200  # คนอื่นยังใช้ได้
    assert c.get("/me", headers=WANIDA).json()["budget_used"] == 2


def test_budget_is_refunded_when_rag_is_down(client, rag):
    rag.raise_error = httpx.ConnectError("refused")
    assert ask(client).status_code == 503
    assert client.get("/me", headers=WANIDA).json()["budget_used"] == 0


@pytest.mark.parametrize("body", [{"question": ""}, {"question": "x" * 2001}, {"question": "x", "top_k": 0},
                                  {"question": "x", "top_k": 21}, {}])
def test_chat_validates_input(client, rag, body):
    r = client.post("/chat", headers=WANIDA, json=body)
    assert r.status_code == 422 and r.json()["error_code"] == "invalid_request" and rag.requests == []


def test_me_and_sources_endpoints(client, rag):
    me = client.get("/me", headers=WANIDA).json()
    assert me == {"user_id": "wanida.demo", "departments": "finance", "security_level": 3, "budget_used": 0}
    assert [s["box_id"] for s in client.get("/sources", headers=WANIDA).json()] == ["example-finance", "example-hr"]
    assert client.get("/sources").status_code == 401
    assert client.get("/health").json() == {"status": "ok"}


# ---------- กฎเหล็กข้อ 10 + ความสอดคล้องกับ demo ----------

def test_gateway_never_imports_the_rag_internals():
    """กฎข้อ 10: เรียก RAG ผ่าน API สาธารณะเท่านั้น"""
    pattern = re.compile(r"^\s*(from|import)\s+(app|scripts|tests)(\.|\s|$)", re.M)
    offenders = [p.name for p in (ROOT / "gateway_mock").glob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"import module ภายในของ RAG: {offenders}"


def test_demo_users_match_bootstrap_personas():
    spec = importlib.util.spec_from_file_location("bootstrap_demo", ROOT / "scripts" / "bootstrap_demo.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bootstrap_demo"] = mod  # dataclass ใน script ต้องหา module ของตัวเองเจอ
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("bootstrap_demo", None)
    assert {(u.user_id, u.departments, u.security_level) for u in DEMO_USERS} == \
           {(p.user_id, p.departments, p.level) for p in mod.PERSONAS}
