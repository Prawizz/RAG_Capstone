import importlib.util
import json
from pathlib import Path

import pytest

from app.models import Document, Unit, UnitType

ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "units"


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("extract_cli", ROOT / "scripts" / "extract_cli.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_json_keeps_thai_unescaped(cli, tmp_path, monkeypatch, capsys):
    units = [
        Unit.model_validate_json((FIXTURES / f"{t.value}.json").read_text(encoding="utf-8"))
        for t in UnitType
    ]
    fake = Document(
        file_id="f", content_hash="0" * 64, filename="คู่มือ.pdf", detected_type="pdf",
        size_bytes=1, units=units, pipeline_version="t", normalizer_version="t",
    )
    monkeypatch.setattr(cli, "extract_file", lambda path, pages=None: fake)
    out = tmp_path / "out.json"
    assert cli.main(["whatever.pdf", "--out", str(out)]) == 0
    raw = out.read_text(encoding="utf-8")
    assert "\\u0e" not in raw and "สมชาย ใจดี" in raw
    assert json.loads(raw)["filename"] == "คู่มือ.pdf"
    assert "คู่มือ.pdf" in capsys.readouterr().out


def test_cli_end_to_end_on_real_pdf(cli, tmp_path, capsys):
    import pymupdf

    pdf = tmp_path / "a.pdf"
    d = pymupdf.open()
    d.new_page().insert_text((72, 100), "hello")
    d.new_page()
    d.save(str(pdf))
    d.close()
    out = tmp_path / "o.json"
    assert cli.main([str(pdf), "--out", str(out), "--pages", "1-2"]) == 0
    text = capsys.readouterr().out
    assert "unit ต่อสถานะ" in text and "ok=2" in text and "เฉลี่ยต่อหน้า" in text
    assert "ด่านตรวจคุณภาพ:" in text and "ok:low_text=1" in text and "ok:empty=1" in text
    assert "ยังไม่ได้ประเมิน" not in text
    assert json.loads(out.read_text(encoding="utf-8"))["unit_count"] == 2


def test_cli_reports_error_and_nonzero_exit(cli, tmp_path, capsys):
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert cli.main([str(f)]) == 1
    assert "unsupported_type" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["0-5", "5-2", "a-b"])
def test_cli_bad_pages(cli, value):
    with pytest.raises(SystemExit):
        cli.main(["x.pdf", "--pages", value])
