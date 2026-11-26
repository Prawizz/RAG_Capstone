"""scripts/questions_eval.py — ฟังก์ชันคำนวณล้วน (ไม่เรียก API)"""

from __future__ import annotations

import pytest

from scripts import questions_eval as qe


def res(rank: int, file: str, page: int, chunk: str | None = None, text: str = "t") -> dict:
    return {"rank": rank, "chunk_id": chunk or f"{file}:{page}:{rank}", "filename": file, "text": text,
            "citation": {"page": page, "label": f"หน้า {page}"}}


def test_auc_perfect_inverse_and_ties():
    assert qe.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert qe.auc([0.1], [0.9]) == 0.0
    assert qe.auc([0.5], [0.5]) == 0.5
    assert qe.auc([0.6, 0.4], [0.5]) == 0.5
    assert qe.auc([], [0.1]) is None


def test_stratified_split_halves_per_type_and_is_deterministic():
    rows = [{"id": f"q{i}", "type": "question"} for i in range(11)] + \
           [{"id": f"u{i}", "type": "unanswerable"} for i in range(4)] + [{"id": "c0", "type": "chitchat"}]
    cal, test = qe.stratified_split(rows, seed=1)
    assert cal.isdisjoint(test) and len(cal | test) == len(rows)
    assert sum(1 for x in cal if x.startswith("q")) == 5 and sum(1 for x in cal if x.startswith("u")) == 2
    assert "c0" in test  # floor(1/2) = 0 ไป calibrate
    assert qe.stratified_split(list(reversed(rows)), seed=1) == (cal, test)  # ไม่ขึ้นกับลำดับในไฟล์


def test_pick_threshold_respects_budget_including_ties():
    scores = [i / 100 for i in range(1, 101)]  # 0.01..1.00
    t = qe.pick_threshold(scores, 0.05)
    assert qe.reject_rate(scores, t) == (5, 100)
    tied = [0.3] * 10 + [0.5] * 90
    t = qe.pick_threshold(tied, 0.05)
    assert qe.reject_rate(tied, t)[0] <= 5  # เสมอกันที่ขอบ -> ปฏิเสธน้อยลง ไม่เกินงบ
    assert qe.pick_threshold([0.2], 0.05) == 0.2


def test_first_rank_compare():
    base = {"a": 1, "b": None, "c": 3, "d": 6}
    other = {"a": 2, "b": 4, "c": 3, "d": 5}
    c = qe.first_rank_compare(base, other)
    assert (c["better"], c["worse"], c["same"]) == (2, 1, 1)
    assert c["at5"] == {"gained": ["b", "d"], "lost": []}


def rows_and_runs():
    rows = [{"id": "q1", "type": "question", "file": "A.PDF", "pages": [5]},
            {"id": "q2", "type": "question", "file": "A.PDF", "pages": [7]},
            {"id": "u1", "type": "unanswerable"}]
    base = {"q1": {"results": [res(1, "A.PDF", 5)] + [res(i, "A.PDF", 50 + i) for i in range(2, 11)]},
            "q2": {"results": [res(1, "A.PDF", 8, "x"), res(2, "B.PDF", 8), res(3, "A.PDF", 9), res(4, "A.PDF", 10),
                               res(5, "A.PDF", 11), res(6, "A.PDF", 7)]},
            "u1": {"results": [res(1, "A.PDF", 1)]}}
    rr = {"q1": {"results": [res(1, "A.PDF", 5)]},
          "q2": {"results": [res(1, "A.PDF", 8, "y"), res(2, "A.PDF", 7), res(3, "A.PDF", 12)]},
          "u1": {"results": [res(1, "A.PDF", 1)]}}
    return rows, {"baseline_r1": base, "reranker": rr}


def test_build_candidates_pool_and_dedup_by_page():
    rows, runs = rows_and_runs()
    pool, items = qe.build_candidates(rows, runs)
    assert pool == ["q2"]  # q1 hit@1 ทั้งคู่ · u1 ไม่ใช่ question
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids))
    assert "q2|A.PDF|7" not in ids  # หน้าตาม truth ไม่ใช่ผู้สมัคร
    assert "q2|A.PDF|8" in ids and "q2|B.PDF|8" in ids  # file ต่างกัน = คนละรายการ
    assert "q2|A.PDF|12" in ids  # จาก reranker
    p8 = next(it for it in items if it["id"] == "q2|A.PDF|8")
    assert sorted(c["chunk_id"] for c in p8["chunks"]) == ["x", "y"]  # chunk ต่างกันในหน้าเดียวกัน รวมเป็นรายการเดียว
    assert {s["run"] for s in p8["seen"]} == {"baseline_r1", "reranker"}
    assert all("A.PDF|7" not in i for i in ids)


def test_lenient_ranks():
    rows, runs = rows_and_runs()
    j = {"q2|A.PDF|8": "no", "q2|B.PDF|8": "yes"}
    out = qe.lenient_ranks(rows, runs["baseline_r1"], j, ("yes",))
    assert out["q1"] == {"strict": True, "lenient": True, "unjudged": 0}
    assert out["q2"]["strict"] is False and out["q2"]["lenient"] is True
    out = qe.lenient_ranks(rows, runs["baseline_r1"], {"q2|A.PDF|8": "unsure"}, ("yes",))
    assert out["q2"] == {"strict": False, "lenient": False, "unjudged": 4}
    out = qe.lenient_ranks(rows, runs["baseline_r1"], {"q2|A.PDF|8": "unsure"}, ("yes", "unsure"))
    assert out["q2"]["lenient"] is True
    assert "u1" not in out


@pytest.mark.parametrize("note,tag", [("ต้อง OCR", "ต้อง OCR"), ("multi_hop", "multi_hop"), ("", "ทั่วไป"), (None, "ทั่วไป")])
def test_tag_of(note, tag):
    assert qe.tag_of({"note": note}) == tag


def test_expected_status_follows_the_mode_rule():
    off = {"results": [{}], "top_score": 0.6, "top_relevance": None}
    on = {"results": [{}], "top_score": 0.9, "top_relevance": 0.2}
    assert qe.expected_status(off, 0.55, 0.99) == "ok"
    assert qe.expected_status({**off, "top_score": 0.5}, 0.55, 0.0) == "no_relevant_context"
    assert qe.expected_status(on, 0.0, 0.55) == "no_relevant_context"  # dense สูงไม่ช่วยเมื่อ reranker เปิด
    assert qe.expected_status({**on, "top_relevance": 0.55}, 0.99, 0.55) == "ok"
    assert qe.expected_status({"results": [], "top_score": None, "top_relevance": None}, 0.0, 0.0) == "no_relevant_context"


def test_optimism_note_reports_cells_where_full_set_is_not_better():
    cell = lambda k, n: {"no_relevant_context": k, "n": n}  # noqa: E731
    good = {"c": {"full": {"question": cell(1, 100), "unanswerable": cell(8, 10), "chitchat": cell(5, 5)},
                  "test_half": {"question": cell(2, 50), "unanswerable": cell(3, 5), "chitchat": cell(2, 2)}}}
    assert "ไม่ได้ดีกว่า" not in qe.optimism_note(good)
    bad = {"c": {**good["c"], "full": {**good["c"]["full"], "question": cell(10, 100)}}}
    assert "c/question" in qe.optimism_note(bad)
