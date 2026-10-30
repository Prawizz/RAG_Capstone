import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models import (
    POSITION_REQUIRED_KEYS,
    Document,
    DocumentStatus,
    Unit,
    UnitStatus,
    UnitType,
)

FIXTURES = Path(__file__).parent / "fixtures" / "units"


def load_unit(name: str) -> Unit:
    return Unit.model_validate_json((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def make_doc(units: list[Unit]) -> Document:
    return Document(
        file_id="f1",
        content_hash="0" * 64,
        filename="ตัวอย่าง.pdf",
        detected_type="pdf",
        size_bytes=1,
        units=units,
        pipeline_version="0",
        normalizer_version="0",
    )


def with_status(unit: Unit, status: UnitStatus) -> Unit:
    data = unit.model_dump()
    data["status"] = status
    data["reason"] = None if status == UnitStatus.OK else "ทดสอบ"
    return Unit.model_validate(data)


def test_fixture_exists_for_every_unit_type():
    assert {p.stem for p in FIXTURES.glob("*.json")} == {t.value for t in UnitType}


@pytest.mark.parametrize("unit_type", list(UnitType))
def test_fixtures_validate(unit_type):
    assert load_unit(unit_type.value).type == unit_type


@pytest.mark.parametrize("unit_type", list(UnitType))
def test_position_has_required_keys(unit_type):
    unit = load_unit(unit_type.value)
    assert POSITION_REQUIRED_KEYS[unit_type] <= unit.position.keys()


@pytest.mark.parametrize("unit_type", list(UnitType))
def test_position_missing_key_rejected(unit_type):
    data = load_unit(unit_type.value).model_dump()
    data["position"] = {}
    with pytest.raises(ValidationError):
        Unit.model_validate(data)


def test_non_ok_unit_requires_reason():
    data = load_unit("page").model_dump()
    data["status"] = UnitStatus.FAILED
    with pytest.raises(ValidationError):
        Unit.model_validate(data)


def test_document_status():
    ok = load_unit("page")
    bad = with_status(ok, UnitStatus.NEEDS_OCR)
    assert make_doc([ok, ok]).status == DocumentStatus.DONE
    assert make_doc([ok, bad]).status == DocumentStatus.DONE_WITH_GAPS
    assert make_doc([bad, bad]).status == DocumentStatus.FAILED
    assert make_doc([]).status == DocumentStatus.FAILED


def test_unit_summary_and_count():
    ok = load_unit("page")
    doc = make_doc([ok, with_status(ok, UnitStatus.REJECTED)])
    assert doc.unit_count == 2
    assert doc.unit_summary == {"ok": 1, "needs_ocr": 0, "failed": 0, "rejected": 1}


def test_json_roundtrip_keeps_thai_unescaped():
    doc = make_doc([load_unit(t.value) for t in UnitType])
    raw = doc.model_dump_json()
    assert "\\u0e" not in raw
    assert "สมชาย ใจดี" in raw
    assert Document.model_validate_json(raw) == doc
    assert json.loads(raw)["status"] == "done"
