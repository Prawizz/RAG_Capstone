import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models import (
    POSITION_REQUIRED_KEYS,
    DetectedType,
    Document,
    DocumentError,
    DocumentStatus,
    ErrorCode,
    QualityFlag,
    QualityReport,
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
        detected_type=DetectedType.PDF,
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


def test_error_forces_failed_status():
    doc = make_doc([load_unit("page")])
    assert doc.status == DocumentStatus.DONE
    doc.error = DocumentError(code=ErrorCode.ENCRYPTED, message="ไฟล์มีรหัสผ่าน")
    assert doc.status == DocumentStatus.FAILED
    raw = doc.model_dump_json()
    assert Document.model_validate_json(raw).error.code == ErrorCode.ENCRYPTED


def test_page_is_one_based_and_page_label_optional():
    data = load_unit("page").model_dump()
    assert data["position"]["page"] == 47 and data["position"]["page_label"] == "45"
    data["position"] = {"page": 0}
    with pytest.raises(ValidationError):
        Unit.model_validate(data)
    data["position"] = {"page": 3}  # ไม่มี page_label ก็ผ่าน
    Unit.model_validate(data)


def test_ok_unit_with_blank_text_requires_empty_flag():
    data = load_unit("page").model_dump()
    data.update(text="   ", blocks=[])
    with pytest.raises(ValidationError):
        Unit.model_validate(data)
    data["quality"]["flags"] = [QualityFlag.EMPTY]
    Unit.model_validate(data)


def test_section_heading_path_includes_own_heading():
    unit = load_unit("section")
    own = next(b for b in unit.blocks if b.type == "heading")
    assert unit.position["heading_path"][-1] == own.text


def test_detected_type_is_enum():
    assert {t.value for t in DetectedType} == {"pdf", "docx", "pptx", "xlsx", "doc", "ppt", "xls"}
    with pytest.raises(ValidationError):
        Document.model_validate({**make_doc([]).model_dump(), "detected_type": "txt"})


@pytest.mark.parametrize("unit_type", list(UnitType))
def test_text_hash_is_sha256_of_text(unit_type):
    unit = load_unit(unit_type.value)
    assert unit.text_hash == hashlib.sha256(unit.text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("code", list(ErrorCode))
def test_error_retriable_derived_from_code(code):
    expected = code in {ErrorCode.CONVERSION_TIMEOUT, ErrorCode.INTERNAL}
    err = DocumentError(code=code, message="x")
    assert err.retriable is expected
    assert DocumentError.model_validate_json(err.model_dump_json()) == err
    assert json.loads(err.model_dump_json())["retriable"] is expected


def test_score_is_optional_and_signals_default_empty():
    unit = load_unit("slide")
    assert unit.quality.score is None
    assert unit.signals == {}
    assert load_unit("page").signals.keys() >= {"char_count", "block_count", "image_coverage"}


def test_score_range_still_enforced():
    with pytest.raises(ValidationError):
        QualityReport(score=1.5)


def test_detected_type_none_requires_error():
    with pytest.raises(ValidationError):
        Document(
            file_id="f", content_hash="0" * 64, filename="x", size_bytes=1,
            pipeline_version="0", normalizer_version="0",
        )


def test_document_without_detected_type_with_error_ok():
    doc = Document(
        file_id="f", content_hash="0" * 64, filename="x.txt", size_bytes=1,
        pipeline_version="0", normalizer_version="0",
        error=DocumentError(code=ErrorCode.UNSUPPORTED_TYPE, message="x"),
    )
    assert doc.detected_type is None and doc.status == DocumentStatus.FAILED
    assert Document.model_validate_json(doc.model_dump_json()) == doc
