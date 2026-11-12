"""API contract v1: schema/ตัวอย่าง/OpenAPI + auth + แปลง header เป็น Principal (route ยังเป็น stub -> 501)"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.access import AclStore, Box, BoxRole, Principal, Visibility
from app.api import schemas as s
from app.api.deps import CurrentPrincipal, ServiceAuth, get_acl_store, get_settings
from app.api.main import create_app
from app.config import Settings

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contract"
KEY = "test-service-key"
USER = {"X-User-Id": "u1024", "X-User-Departments": "finance", "X-User-Security-Level": "2"}
AUTH = {"X-Service-Key": KEY, **USER}


@pytest.fixture
def acl() -> AclStore:
    store = AclStore()
    store.create_box(Box(box_id="finance-reports", visibility=Visibility.PRIVATE, name="รายงานการเงิน"), "owner1")
    store.set_member("finance-reports", "u1024", BoxRole.VIEWER)
    store.create_box(Box(box_id="other-box", visibility=Visibility.PUBLIC), "someone-else")
    return store


@pytest.fixture
def app(acl):
    application = create_app()
    application.dependency_overrides[get_settings] = lambda: Settings(service_key=KEY, max_upload_bytes=1024)
    application.dependency_overrides[get_acl_store] = lambda: acl

    @application.get("/_test/whoami", dependencies=[ServiceAuth])
    def whoami(principal: CurrentPrincipal):  # เปิดดู Principal ที่แปลงจาก header (เฉพาะ test)
        return {"user_id": principal.user_id, "departments": sorted(principal.departments),
                "security_level": principal.security_level, "box_ids": sorted(principal.box_ids)}

    @application.get("/_test/boom")
    def boom():
        raise RuntimeError("secret internal detail")

    return application


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def spec() -> dict:
    return create_app().openapi()


def assert_error(resp, status: int, code: str) -> dict:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error_code", "message", "trace_id"} and body["error_code"] == code
    assert resp.headers["X-Trace-Id"] == body["trace_id"]
    return body


# ---------------------------------------------------------------------------------------------
# ตัวอย่าง + OpenAPI (ไฟล์ใน contract/)
# ---------------------------------------------------------------------------------------------

def test_every_example_file_validates_against_its_schema():
    files = sorted((CONTRACT / "examples").glob("*.json"))
    assert {f.stem for f in files} == set(s.EXAMPLES)  # ไม่มีไฟล์ตกค้าง / ขาด
    for f in files:
        model, _payload = s.EXAMPLES[f.stem]
        model.model_validate(json.loads(f.read_text(encoding="utf-8")))


@pytest.mark.parametrize("name", ["retrieve_source_required", "retrieve_no_relevant_context", "ingest_duplicate",
                                  "ingest_not_found", "ingest_acl_widening", "jobs_not_found",
                                  "sources_files_not_found"])
def test_required_cases_are_exported(name):
    assert (CONTRACT / "examples" / f"{name}.json").is_file()


def test_contract_files_match_the_code():
    from scripts.export_openapi import main as export_main

    assert export_main(["--check"]) == 0, "รัน python scripts/export_openapi.py"


def test_openapi_has_no_forbidden_status(spec):
    codes = {code for ops in spec["paths"].values() for op in ops.values() for code in op["responses"]}
    assert "403" not in codes
    assert "403" not in json.dumps(spec, ensure_ascii=False)


def test_openapi_errors_all_use_error_response(spec):
    assert "HTTPValidationError" not in spec["components"]["schemas"]
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            for code, resp in op["responses"].items():
                if int(code) < 400 or (path == "/v1/health/ready" and code == "503"):
                    continue
                ref = resp["content"]["application/json"]["schema"]["$ref"]
                assert ref.endswith("/ErrorResponse"), (method, path, code)


def test_every_object_schema_has_examples_that_validate(spec):
    models = {name: getattr(s, name) for name in dir(s) if isinstance(getattr(s, name), type)}
    for name, schema in spec["components"]["schemas"].items():
        if "properties" not in schema:
            continue  # enum
        assert schema.get("examples"), f"{name} ไม่มี example"
        if name in models:
            for example in schema["examples"]:
                models[name].model_validate(example)


def test_auth_and_user_headers_on_every_endpoint_except_health(spec):
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            params = {p["name"] for p in op.get("parameters", [])}
            assert "X-Trace-Id" in params
            for resp in op["responses"].values():
                assert "X-Trace-Id" in resp["headers"]
            if path.startswith("/v1/health"):
                assert not params & {"X-User-Id", "X-User-Departments", "X-User-Security-Level"}
                assert ("security" in op) == (path == "/v1/health/ready")
            else:
                assert op["security"] == [{"ServiceKey": []}], (method, path)
                assert {"X-User-Id", "X-User-Departments", "X-User-Security-Level"} <= params, (method, path)


def test_endpoints_and_success_codes(spec):
    got = {(m.upper(), p): sorted(c for c in op["responses"] if c < "400")
           for p, ops in spec["paths"].items() for m, op in ops.items()}
    assert got == {
        ("GET", "/v1/health/live"): ["200"],
        ("GET", "/v1/health/ready"): ["200"],
        ("POST", "/v1/ingest"): ["200", "202"],
        ("GET", "/v1/jobs/{job_id}"): ["200"],
        ("GET", "/v1/sources"): ["200"],
        ("GET", "/v1/sources/{box_id}/files"): ["200"],
        ("POST", "/v1/retrieve"): ["200"],
        ("POST", "/v1/boxes"): ["201"],
        ("PUT", "/v1/boxes/{box_id}/members"): ["200"],
        ("DELETE", "/v1/boxes/{box_id}/members/{user_id}"): ["200"],
    }


# ---------------------------------------------------------------------------------------------
# กล่องที่ไม่เป็นสมาชิก == กล่องว่าง (กฎเหล็กข้อ 9) — ระดับ schema/ตัวอย่าง
# ---------------------------------------------------------------------------------------------

def test_not_member_example_equals_empty_member_example_except_trace_id():
    a = json.loads((CONTRACT / "examples" / "retrieve_member_empty_box.json").read_text(encoding="utf-8"))
    b = json.loads((CONTRACT / "examples" / "retrieve_not_member_box.json").read_text(encoding="utf-8"))
    assert a["trace_id"] != b["trace_id"]
    assert {k: v for k, v in a.items() if k != "trace_id"} == {k: v for k, v in b.items() if k != "trace_id"}


@pytest.mark.parametrize("leak", [{"visibility": "private"}, {"top_score": 0.0}])
def test_empty_result_cannot_reveal_box_properties(leak):
    """ไม่มีผล -> visibility/top_score ต้อง null: กล่องที่ไม่เห็นจึงตอบเหมือนกล่องว่างได้ทางเดียว"""
    with pytest.raises(ValidationError):
        s.RetrieveResponse.model_validate({**s.EX_RETRIEVE_EMPTY_MEMBER, **leak})


def test_retrieve_response_rules():
    s.RetrieveResponse.model_validate(s.EX_RETRIEVE_NO_RELEVANT)  # คะแนนต่ำยังคืน results ได้
    with pytest.raises(ValidationError):  # ok ต้องมีผล
        s.RetrieveResponse.model_validate({**s.EX_RETRIEVE_EMPTY_MEMBER, "status": "ok"})
    with pytest.raises(ValidationError):  # source_required ห้ามมี box_id
        s.RetrieveResponse.model_validate({**s.EX_RETRIEVE_SOURCE_REQUIRED, "box_id": "finance-reports"})
    with pytest.raises(ValidationError):  # top_score ต้องเท่าคะแนนสูงสุด
        s.RetrieveResponse.model_validate({**s.EX_RETRIEVE_OK, "top_score": 0.99})
    with pytest.raises(ValidationError):  # citation ของ page ต้องมี page
        s.Citation.model_validate({"unit_type": "page", "label": "หน้า 1", "char_start": 0, "char_end": 1})


@pytest.mark.parametrize("body", [
    {"query": ""}, {"query": "   "}, {"query": "ก" * 2001}, {"query": "q", "top_k": 0}, {"query": "q", "top_k": 21},
    {"query": "q", "box_ids": ["finance-reports"]},  # ห้ามรับ box_ids จาก Gateway
    {"query": "q", "principal": {"user_id": "u"}},
    {"query": "q", "box_id": "../etc"},
])
def test_retrieve_request_rejects(body):
    with pytest.raises(ValidationError):
        s.RetrieveRequest.model_validate(body)


def test_retrieve_request_defaults_and_limits():
    r = s.RetrieveRequest.model_validate({"query": "ก" * 2000})
    assert r.box_id is None and r.top_k == 5


def test_new_ingest_job_must_be_queued():
    with pytest.raises(ValidationError):
        s.IngestResponse.model_validate({**s.EX_INGEST_QUEUED, "status": "done"})


# ---------------------------------------------------------------------------------------------
# runtime: health, trace id, auth
# ---------------------------------------------------------------------------------------------

def test_live_works_without_auth_and_echoes_trace_id(client):
    r = client.get("/v1/health/live", headers={"X-Trace-Id": "gw-req-42"})
    assert r.status_code == 200 and r.json() == {"status": "ok", "trace_id": "gw-req-42"}
    assert r.headers["X-Trace-Id"] == "gw-req-42"


def test_trace_id_is_generated_when_missing_or_malformed(client):
    a = client.get("/v1/health/live").json()["trace_id"]
    b = client.get("/v1/health/live", headers={"X-Trace-Id": "bad id with spaces"}).json()["trace_id"]
    assert re.fullmatch(r"[0-9a-f]{32}", a) and re.fullmatch(r"[0-9a-f]{32}", b) and a != b


@pytest.mark.parametrize("key", [None, "", "wrong-key"])
def test_wrong_or_missing_service_key_is_401(client, key):
    headers = dict(USER)
    if key is not None:
        headers["X-Service-Key"] = key
    for method, path, kw in [("get", "/v1/sources", {}), ("post", "/v1/retrieve", {"json": {"query": "q"}}),
                             ("get", "/v1/jobs/job_1", {}), ("get", "/v1/health/ready", {})]:
        assert_error(getattr(client, method)(path, headers=headers, **kw), 401, "unauthorized")


def test_auth_is_checked_before_user_headers_and_body(client):
    """ผู้ที่ไม่ผ่าน service auth ไม่ได้รายละเอียด validation"""
    r = client.post("/v1/retrieve", headers={"X-Service-Key": "wrong"}, json={"query": ""})
    assert_error(r, 401, "unauthorized")


def test_unset_service_key_fails_closed(app, client):
    app.dependency_overrides[get_settings] = lambda: Settings(service_key=None)
    assert_error(client.get("/v1/sources", headers=AUTH), 401, "unauthorized")


def test_stub_endpoints_return_not_implemented(client):
    calls = [
        ("get", "/v1/sources", {}),
        ("get", "/v1/sources/finance-reports/files", {"params": {"cursor": "abc"}}),
        ("get", "/v1/jobs/job_1", {}),
        ("post", "/v1/retrieve", {"json": {"query": "เงินปันผล", "box_id": "finance-reports"}}),
        ("post", "/v1/boxes", {"json": s.EX_CREATE_BOX}),
        ("put", "/v1/boxes/finance-reports/members", {"json": s.EX_PUT_MEMBER}),
        ("delete", "/v1/boxes/finance-reports/members/u2048", {}),
        ("post", "/v1/ingest", {"data": {"box_id": "finance-reports"}, "files": {"file": ("a.pdf", b"%PDF-1.7")}}),
    ]
    for method, path, kw in calls:
        r = getattr(client, method)(path, headers={**AUTH, "X-Trace-Id": "t-1"}, **kw)
        body = assert_error(r, 501, "not_implemented")
        assert body["trace_id"] == "t-1", path


def test_unknown_path_is_not_found_in_error_format(client):
    assert_error(client.get("/v1/nope", headers=AUTH), 404, "not_found")


def test_unexpected_error_is_500_without_details(client):
    body = assert_error(client.get("/_test/boom"), 500, "internal")
    assert "secret" not in body["message"]


# ---------------------------------------------------------------------------------------------
# runtime: header -> Principal
# ---------------------------------------------------------------------------------------------

def whoami(client, **headers):
    return client.get("/_test/whoami", headers={**AUTH, **headers})


def test_headers_become_principal_with_boxes_from_acl_store(client):
    r = whoami(client, **{"X-User-Departments": " finance , ,audit ", "X-User-Box-Ids": "other-box"})
    assert r.status_code == 200
    assert r.json() == {"user_id": "u1024", "departments": ["audit", "finance"], "security_level": 2,
                        "box_ids": ["finance-reports"]}  # box_ids จากสมาชิกภาพ ไม่ใช่จาก header


def test_empty_departments_header_means_no_department(client):
    r = whoami(client, **{"X-User-Departments": ""})
    assert r.status_code == 200 and r.json()["departments"] == []


def test_non_member_user_has_no_boxes(client):
    r = whoami(client, **{"X-User-Id": "stranger"})
    assert r.status_code == 200 and r.json()["box_ids"] == []


@pytest.mark.parametrize("level", ["abc", "-1", "1.5", "", " ", "+2", "๒".encode("utf-8"), "1e3"])
def test_security_level_not_a_non_negative_integer_is_422(client, level):
    body = assert_error(whoami(client, **{"X-User-Security-Level": level}), 422, "invalid_request")
    assert "X-User-Security-Level" in body["message"]


@pytest.mark.parametrize("missing", ["X-User-Id", "X-User-Departments", "X-User-Security-Level"])
def test_missing_user_header_is_422(client, missing):
    headers = {k: v for k, v in AUTH.items() if k != missing}
    body = assert_error(client.get("/_test/whoami", headers=headers), 422, "invalid_request")
    assert missing in body["message"]


def test_blank_user_id_is_422(client):
    assert_error(whoami(client, **{"X-User-Id": "   "}), 422, "invalid_request")


def test_principal_reflects_membership_changes_on_next_request(client, acl):
    assert whoami(client).json()["box_ids"] == ["finance-reports"]
    acl.remove_member("finance-reports", "u1024")
    assert whoami(client).json()["box_ids"] == []


# ---------------------------------------------------------------------------------------------
# runtime: ingest (ตรวจก่อน stub)
# ---------------------------------------------------------------------------------------------

def test_ingest_declared_size_over_limit_is_413_before_reading(client):
    r = client.post("/v1/ingest", headers=AUTH, data={"box_id": "finance-reports"},
                    files={"file": ("big.pdf", b"x" * 4096)})
    assert_error(r, 413, "too_large")


def test_ingest_bad_file_acl_is_422(client):
    r = client.post("/v1/ingest", headers=AUTH, data={"box_id": "finance-reports", "file_acl": "{not json"},
                    files={"file": ("a.pdf", b"%PDF")})
    assert_error(r, 422, "invalid_request")
    r = client.post("/v1/ingest", headers=AUTH,
                    data={"box_id": "finance-reports", "file_acl": json.dumps({"min_security_level": -1})},
                    files={"file": ("a.pdf", b"%PDF")})
    assert_error(r, 422, "invalid_request")


def test_ingest_missing_file_or_box_is_422(client):
    assert_error(client.post("/v1/ingest", headers=AUTH, data={"box_id": "finance-reports"}), 422, "invalid_request")
    assert_error(client.post("/v1/ingest", headers=AUTH, files={"file": ("a.pdf", b"%PDF")}), 422, "invalid_request")
