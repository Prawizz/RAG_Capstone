import zipfile

import pytest

import app.extraction.detect as detect_mod
from app.extraction.detect import OLE_MAGIC, detect
from app.extraction.errors import ExtractionError
from app.models import DetectedType, ErrorCode


def make_zip(path, names):
    with zipfile.ZipFile(path, "w") as z:
        for n in names:
            z.writestr(n, "x")
    return path


def test_pdf_with_leading_junk(tmp_path):
    f = tmp_path / "junk.pdf"
    f.write_bytes(b"\r\n\x00junk header\n" * 10 + b"%PDF-1.4\n...")
    assert detect(f) == DetectedType.PDF


def test_pdf_marker_beyond_1024_bytes_is_not_pdf(tmp_path):
    f = tmp_path / "late.bin"
    f.write_bytes(b"x" * 2000 + b"%PDF-1.4")
    with pytest.raises(ExtractionError):
        detect(f)


def test_pdf(tmp_path):
    f = tmp_path / "noext"  # ห้ามดูนามสกุล
    f.write_bytes(b"%PDF-1.7\n...")
    assert detect(f) == DetectedType.PDF


@pytest.mark.parametrize(
    "inner,expected",
    [
        ("word/document.xml", DetectedType.DOCX),
        ("ppt/presentation.xml", DetectedType.PPTX),
        ("xl/workbook.xml", DetectedType.XLSX),
    ],
)
def test_office_zip(tmp_path, inner, expected):
    f = make_zip(tmp_path / "x.bin", ["[Content_Types].xml", inner])
    assert detect(f) == expected


def test_extension_is_ignored(tmp_path):
    f = make_zip(tmp_path / "report.pdf", ["word/document.xml"])
    assert detect(f) == DetectedType.DOCX


def test_plain_zip_unsupported(tmp_path):
    f = make_zip(tmp_path / "a.zip", ["hello.txt"])
    with pytest.raises(ExtractionError) as e:
        detect(f)
    assert e.value.code == ErrorCode.UNSUPPORTED_TYPE


def test_broken_zip_corrupt(tmp_path):
    f = tmp_path / "a.docx"
    f.write_bytes(b"PK\x03\x04" + b"\x00" * 10)
    with pytest.raises(ExtractionError) as e:
        detect(f)
    assert e.value.code == ErrorCode.CORRUPT


@pytest.mark.parametrize("data", [b"hello world", b"", b"\xff\xd8\xff\xe0jpeg"])
def test_garbage_unsupported(tmp_path, data):
    f = tmp_path / "a.pdf"
    f.write_bytes(data)
    with pytest.raises(ExtractionError) as e:
        detect(f)
    assert e.value.code == ErrorCode.UNSUPPORTED_TYPE


class FakeOle:
    streams: list[list[str]] = []

    def __init__(self, path):
        pass

    def listdir(self, streams=True, storages=False):
        return self.streams

    def close(self):
        pass


def ole_file(tmp_path, monkeypatch, streams):
    FakeOle.streams = streams
    monkeypatch.setattr(detect_mod.olefile, "OleFileIO", FakeOle)
    f = tmp_path / "x.bin"
    f.write_bytes(OLE_MAGIC + b"\x00" * 100)
    return f


@pytest.mark.parametrize(
    "streams,expected",
    [
        ([["WordDocument"], ["1Table"]], DetectedType.DOC),
        ([["PowerPoint Document"], ["Current User"]], DetectedType.PPT),
        ([["Workbook"]], DetectedType.XLS),
        ([["Book"]], DetectedType.XLS),
    ],
)
def test_ole_stream_mapping(tmp_path, monkeypatch, streams, expected):
    assert detect(ole_file(tmp_path, monkeypatch, streams)) == expected


@pytest.mark.parametrize("name", ["EncryptionInfo", "EncryptedPackage"])
def test_ole_encrypted(tmp_path, monkeypatch, name):
    f = ole_file(tmp_path, monkeypatch, [[name], ["DataSpaces", "Version"]])
    with pytest.raises(ExtractionError) as e:
        detect(f)
    assert e.value.code == ErrorCode.ENCRYPTED


def test_ole_unknown_streams(tmp_path, monkeypatch):
    with pytest.raises(ExtractionError) as e:
        detect(ole_file(tmp_path, monkeypatch, [["Something"]]))
    assert e.value.code == ErrorCode.UNSUPPORTED_TYPE


def test_ole_broken_is_corrupt(tmp_path):
    f = tmp_path / "x.doc"
    f.write_bytes(OLE_MAGIC + b"\x00" * 20)
    with pytest.raises(ExtractionError) as e:
        detect(f)
    assert e.value.code == ErrorCode.CORRUPT
