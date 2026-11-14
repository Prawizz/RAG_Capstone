"""ตัวตรวจ eval/datasets/questions.jsonl (eval/questions.py) + schema ที่คู่กัน — ไม่แตะไฟล์ questions.jsonl จริง"""

from __future__ import annotations

import json

import pytest

from eval.questions import DEFAULT_PATH, SCHEMA_PATH, TYPES, QuestionsError, load


def write(tmp_path, rows) -> object:
    p = tmp_path / "questions.jsonl"
    p.write_text("\n".join(r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                 encoding="utf-8")
    return p


def test_schema_file_agrees_with_the_validator():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert tuple(schema["properties"]["type"]["enum"]) == TYPES
    assert set(schema["required"]) == {"question", "type"}
    assert set(schema["properties"]) == {"id", "question", "type", "file", "pages", "note"}


def test_schema_examples_pass_the_validator(tmp_path):
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    rows = load(write(tmp_path, schema["examples"]))
    assert [r["type"] for r in rows] == ["question", "unanswerable", "chitchat"]
    assert rows[0]["pages"] == [152, 153] and rows[1]["pages"] == []


def test_empty_file_is_an_error_not_a_pass(tmp_path):
    with pytest.raises(QuestionsError, match="ยังไม่มีคำถาม"):
        load(write(tmp_path, [""]))


def test_the_real_file_is_still_empty_and_reported_as_such():
    """ไฟล์จริงยังว่าง (รอคนทำชุดคำถาม) — ถ้ามีคนเติมแล้ว ต้องผ่านตัวตรวจ"""
    if DEFAULT_PATH.read_text(encoding="utf-8").strip():
        assert load(DEFAULT_PATH)
    else:
        with pytest.raises(QuestionsError, match="ยังไม่มีคำถาม"):
            load(DEFAULT_PATH)


@pytest.mark.parametrize("row, message", [
    ({"question": "q", "type": "question", "pages": [1]}, "ต้องมี file"),
    ({"question": "q", "type": "question", "file": "a.pdf"}, "ต้องมี pages"),
    ({"question": "q", "type": "question", "file": "a.pdf", "pages": []}, "ต้องมี pages"),
    ({"question": "q", "type": "question", "file": "a.pdf", "pages": [0]}, ">= 1"),
    ({"question": "q", "type": "question", "file": "a.pdf", "pages": ["3"]}, ">= 1"),
    ({"question": "q", "type": "question", "file": "a.pdf", "pages": [3, 3]}, "ซ้ำ"),
    ({"question": "q", "type": "unanswerable", "file": "a.pdf"}, "ห้ามมี file"),
    ({"question": "", "type": "chitchat"}, "question ต้องเป็นข้อความ"),
    ({"question": "q", "type": "smalltalk"}, "type ต้องเป็น"),
    ({"question": "q", "type": "chitchat", "answer": "x"}, "ไม่รู้จักฟิลด์ 'answer'"),
    ("{not json", "JSON ไม่ถูกต้อง"),
])
def test_bad_rows_are_rejected_with_the_line_number(tmp_path, row, message):
    with pytest.raises(QuestionsError) as e:
        load(write(tmp_path, [{"question": "ok", "type": "chitchat"}, row]))
    assert any(message in m and "questions.jsonl:2" in m for m in e.value.errors), e.value.errors


def test_duplicate_ids_are_rejected(tmp_path):
    rows = [{"id": "a", "question": "x", "type": "chitchat"}, {"id": "a", "question": "y", "type": "chitchat"}]
    with pytest.raises(QuestionsError, match="ซ้ำกับบรรทัด 1"):
        load(write(tmp_path, rows))
