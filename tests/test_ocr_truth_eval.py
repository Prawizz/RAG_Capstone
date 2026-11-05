"""scripts/ocr_truth_eval.py: canon, ประกอบ truth, dedup, ตัดหมวดยกเว้น, CER (ข้อมูลสังเคราะห์ ไม่แตะแคช/ข้อมูลจริง)"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("ocr_truth_eval", ROOT / "scripts" / "ocr_truth_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def item(n, kind, pos, ty=None, te=None, bbox=(0, 0, 100, 20), image=0, region=None, audit=False, reason=None, auto=None):
    return {"id": f"F_p1_{n:03d}", "kind": kind, "pos": pos, "order": n,
            "typhoon": {"text": ty} if ty is not None else None,
            "tesseract": {"text": te, "image": image, "bbox": list(bbox), "conf": 90.0, "column": None} if te is not None else None,
            "region": region, "audit": audit, "audit_reason": reason,
            "auto_accept": (kind == "agree" and not audit) if auto is None else auto}


def page(items, figures=(), numbers=()):
    return {"file": "F.PDF", "page": 1, "items": items, "figures": list(figures), "typhoon_page_numbers": list(numbers)}


def verdict(text, v="edited"):
    return {"verdict": v, "text": text}


# ---------- canon ----------

def test_canon_subscript_and_superscript_digits(ev):
    assert ev.canon("H₂O ⁰¹²³⁴⁵⁶⁷⁸⁹ ₀₁₂₃₄₅₆₇₈₉") == "H2O01234567890123456789"


def test_canon_sara_am_decomposed(ev):
    assert ev.canon("กํา") == "กำ"
    assert ev.canon("ดําเนิน") == ev.canon("ดำเนิน")


def test_canon_strips_leading_bullets_but_keeps_inner_and_numbers(ev):
    assert ev.canon("• ข้อหนึ่ง") == "ข้อหนึ่ง"
    assert ev.canon("•ข้อหนึ่ง") == "ข้อหนึ่ง"
    assert ev.canon("+ H2 & CCS\n- สอง\n* สาม") == "H2&CCSสองสาม"
    assert ev.canon("a - b") == "a-b"  # ขีดกลางในบรรทัดไม่ใช่ bullet
    assert ev.canon("-5") == "-5"  # ติดตัวเลข = เครื่องหมายลบ


def test_canon_converts_curly_quotes_to_straight(ev):
    assert ev.canon("“Triple Transformation” ‘x’") == "\"TripleTransformation\"'x'"
    assert ev.canon("“a”") == ev.canon('"a"')


def test_eval_and_prepare_share_one_canon_function(ev):
    spec = importlib.util.spec_from_file_location("ocr_truth_prepare", ROOT / "scripts" / "ocr_truth_prepare.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    assert prep.canon is ev.canon


def test_canon_strips_markdown_and_all_whitespace(ev):
    assert ev.canon("# หัว **หนา** | ช่อง\t ต่อ\nไป") == "หัวหนาช่องต่อไป"


# ---------- CER ----------

def test_cer_synthetic(ev):
    assert ev.cer_stats("abcdefghij", "abcdefghij")["cer"] == 0
    s = ev.cer_stats("abcdefghij", "abcdefghix")  # แทนที่ 1 ตัวจาก 10
    assert s["distance"] == 1 and s["cer"] == pytest.approx(0.1) and s["accuracy"] == pytest.approx(0.9) and s["pass"]
    s = ev.cer_stats("abcdefghij", "abcdefgh")  # ขาด 2 ตัว
    assert s["distance"] == 2 and not s["pass"]
    assert ev.cer_stats("ab", "abcd")["cer"] == pytest.approx(1.0)  # ตัวเกินนับ (CER เกิน 1 ได้ถ้าเกินมาก)


def test_aggregate_cer_is_micro_average(ev):
    a, b = ev.cer_stats("a" * 10, "a" * 9), ev.cer_stats("b" * 90, "b" * 90)
    tot = ev.cer_stats_from_sums(100, 99, a["distance"] + b["distance"])
    assert tot["cer"] == pytest.approx(0.01)


def test_ngram_scores_ignore_order(ev):
    s = ev.ngram_scores(["abcdef", "ghijkl"], ["ghijkl", "abcdef"])
    assert s["recall"] == s["precision"] == s["f1"] == 1.0
    s = ev.ngram_scores(["abcdef"], ["abcdef", "zzzzzz"])
    assert s["recall"] == 1.0 and s["precision"] < 1.0


def test_binomial_upper_bound_zero_errors(ev):
    assert ev.upper_bound_binomial(0, 34) == pytest.approx(1 - 0.05 ** (1 / 34), abs=1e-6)
    assert ev.upper_bound_binomial(0, 0) is None


# ---------- ประกอบ truth ----------

def test_truth_assembled_in_pos_order_with_auto_accept_typhoon_and_judged_text(ev):
    items = [
        item(1, "tesseract_only", 9.5, te="ตก"),                        # อยู่ระหว่าง 2 กับ 3
        item(2, "agree", 0, ty="กขค", te="กขค"),                        # รับอัตโนมัติ = Typhoon
        item(3, "diff", 10, ty="งจฉ", te="งจก"),                        # คนแก้
        item(4, "agree", 20, ty="xyz", te="xyz", audit=True, reason="sample"),  # สุ่มตรวจ ใช้ข้อความที่คนตัดสิน
    ]
    judgments = {"F_p1_001": verdict("ตกจริง"), "F_p1_003": verdict("งจก"), "F_p1_004": verdict("xyz", "typhoon")}
    r = ev.evaluate_page(page(items), judgments, excluded={}, diagram_ranges={})
    assert r["truth_chars"] == len("กขค" + "ตกจริง" + "งจก" + "xyz")
    # truth = กขค|ตกจริง|งจก|xyz ; Typhoon hyp ไม่มีบรรทัด tesseract_only และ "งจฉ" ผิด 1 ตัว
    assert r["cer"]["typhoon"]["distance"] == 6 + 1
    assert r["cer"]["tesseract"]["distance"] == len("จริง")  # Tesseract อ่านบรรทัด tesseract_only ได้แค่ "ตก" ขาด "จริง"


def test_truth_order_differs_from_item_list_order(ev):
    # รายการให้มาสลับลำดับ แต่ประกอบตาม pos ได้ truth ต่อเนื่อง ถ้าเรียงผิด CER ไม่เป็น 0
    items = [item(2, "agree", 3, ty="DEF", te="DEF"), item(1, "agree", 0, ty="ABC", te="ABC")]
    r = ev.evaluate_page(page(items), {}, excluded={}, diagram_ranges={})
    assert r["cer"]["typhoon"]["distance"] == 0 and r["truth_chars"] == 6


def test_absent_judgment_is_empty_and_missed_line_is_counted(ev):
    items = [item(1, "agree", 0, ty="ABC", te="ABC"),
             item(2, "tesseract_only", 2.5, te="junk"),
             item(3, "tesseract_only", 2.6, te="อ่านตก")]
    judgments = {"F_p1_002": verdict("", "absent"), "F_p1_003": verdict("อ่านตก", "tesseract")}
    r = ev.evaluate_page(page(items), judgments, excluded={}, diagram_ranges={})
    assert [m["id"] for m in r["missed"]] == ["F_p1_003"]
    assert r["truth_chars"] == len("ABC" + "อ่านตก")
    assert r["fabricated"] == []


def test_fabricated_typhoon_only_counted(ev):
    items = [item(1, "agree", 0, ty="ABC", te="ABC"), item(2, "typhoon_only", 3, ty="แต่งขึ้น")]
    r = ev.evaluate_page(page(items), {"F_p1_002": verdict("", "absent")}, excluded={}, diagram_ranges={})
    assert [f["id"] for f in r["fabricated"]] == ["F_p1_002"]
    assert r["cer"]["typhoon"]["distance"] == len("แต่งขึ้น")  # ตัวอักษรเกินนับเป็นผิด


# ---------- dedup ----------

REGION = {"image": 0, "bbox": [0, 0, 10 ** 6, 100]}


def test_dedup_equal_keeps_typhoon_only_and_not_missed(ev):
    items = [item(1, "tesseract_only", -0.5, te="บรษท", bbox=(0, 10, 100, 40)),
             item(2, "typhoon_only", 0, ty="บริษัท", region=REGION)]
    judgments = {"F_p1_001": verdict("บริษัท"), "F_p1_002": verdict("บริษัท", "typhoon")}
    r = ev.evaluate_page(page(items), judgments, excluded={}, diagram_ranges={})
    (p,) = r["dedup_pairs"]
    assert (p["relation"], p["kept"], p["dropped"]) == ("equal", "F_p1_002", "F_p1_001")
    assert r["truth_chars"] == len("บริษัท")  # นับครั้งเดียว
    assert r["missed"] == []
    assert r["cer"]["typhoon"]["distance"] == 0


def test_dedup_equal_uses_canon_so_sara_am_forms_match(ev):
    items = [item(1, "tesseract_only", 0.5, te="x", bbox=(0, 10, 100, 40)), item(2, "typhoon_only", 1, ty="x", region=REGION)]
    judgments = {"F_p1_001": verdict("ดําเนิน"), "F_p1_002": verdict("ดำเนิน", "typhoon")}
    assert len(ev.find_dedup_pairs(items, judgments)) == 1


def test_dedup_containment_keeps_the_longer_and_drops_both_parts(ev):
    # tesseract_only บรรทัดยาว ครอบ typhoon_only สองชิ้น (เหมือน PTT 293 _003 ⊃ _001 + _012)
    items = [item(1, "typhoon_only", 0, ty="ส่วนที่ 2", region=None),
             item(2, "tesseract_only", 0.5, te="x", bbox=(0, 10, 100, 40)),
             item(3, "typhoon_only", 1, ty="คณะกรรมการ", region=REGION)]
    judgments = {"F_p1_001": verdict("ส่วนที่ 2", "typhoon"), "F_p1_002": verdict("ส่วนที่ 2 คณะกรรมการ", "tesseract"),
                 "F_p1_003": verdict("คณะกรรมการ", "typhoon")}
    r = ev.evaluate_page(page(items), judgments, excluded={}, diagram_ranges={})
    assert {(p["relation"], p["kept"], p["dropped"]) for p in r["dedup_pairs"]} == {
        ("tesseract_only_contains", "F_p1_002", "F_p1_001"), ("tesseract_only_contains", "F_p1_002", "F_p1_003")}
    assert r["truth_chars"] == len("ส่วนที่2คณะกรรมการ") and r["missed"] == []


def test_dedup_typhoon_only_longer_drops_tesseract_only(ev):
    items = [item(1, "tesseract_only", 0.5, te="x", bbox=(0, 10, 100, 40)), item(2, "typhoon_only", 1, ty="x", region=REGION)]
    judgments = {"F_p1_001": verdict("การดำเนิน"), "F_p1_002": verdict("การดำเนินงาน", "typhoon")}
    (p,) = ev.find_dedup_pairs(items, judgments)
    assert (p["relation"], p["dropped"]) == ("typhoon_only_contains", "F_p1_001")


def test_dedup_requires_same_image_and_overlapping_y_and_nonempty_text(ev):
    judgments = {"F_p1_001": verdict("บริษัท"), "F_p1_002": verdict("บริษัท", "typhoon")}
    far = item(1, "tesseract_only", 0.5, te="x", bbox=(0, 500, 100, 540))
    other_image = item(1, "tesseract_only", 0.5, te="x", bbox=(0, 10, 100, 40), image=1)
    near = item(1, "tesseract_only", 0.5, te="x", bbox=(0, 10, 100, 40))
    ty = item(2, "typhoon_only", 1, ty="x", region=REGION)
    assert ev.find_dedup_pairs([far, ty], judgments) == []
    assert ev.find_dedup_pairs([other_image, ty], judgments) == []
    assert len(ev.find_dedup_pairs([near, ty], judgments)) == 1
    empty = {"F_p1_001": verdict("", "absent"), "F_p1_002": verdict("บริษัท", "typhoon")}
    assert ev.find_dedup_pairs([near, ty], empty) == []  # ข้อความว่างไม่ใช่ข้อความย่อยของทุกอย่าง
    unrelated = {"F_p1_001": verdict("อื่น"), "F_p1_002": verdict("บริษัท", "typhoon")}
    assert ev.find_dedup_pairs([near, ty], unrelated) == []


# ---------- หมวดยกเว้น / ชุด (ข) ----------

def test_excluded_items_removed_from_truth_and_typhoon_hyp_and_counted_separately(ev):
    items = [item(1, "agree", 0, ty="ABC", te="ABC"),
             item(2, "typhoon_only", 3, ty="ลายเซ็น", region=None),    # ลายมือ: อยู่นอก scope
             item(3, "typhoon_only", 10, ty="หัวตารางใหม่", region=None)]  # Typhoon จัดหัวตารางใหม่ (absent แต่ห้ามนับแต่งขึ้น)
    judgments = {"F_p1_002": verdict("ลายเซ็น", "typhoon"), "F_p1_003": verdict("", "absent")}
    excluded = {"F_p1_002": "handwriting", "F_p1_003": "restructured"}
    r = ev.evaluate_page(page(items), judgments, excluded=excluded, diagram_ranges={})
    assert r["truth_chars"] == 3 and r["cer"]["typhoon"]["distance"] == 0
    assert r["fabricated"] == []
    assert set(r["excluded"]) == set(excluded) and r["excluded"]["F_p1_002"]["category"] == "handwriting"
    # ถ้าไม่ตัดออกจาก hypothesis จะเป็นตัวอักษรเกิน
    assert r["cer"]["typhoon_keep_excluded"]["distance"] == len("ลายเซ็น" + "หัวตารางใหม่")


def header_page():
    """หัวตารางที่ Typhoon จัดใหม่: Tesseract อ่านเป็น 3 บรรทัด, Typhoon รวมเป็นบรรทัดเดียวสลับลำดับ (typhoon_only ที่คนตัดสิน absent)"""
    items = [item(1, "agree", 0, ty="เนื้อหา", te="เนื้อหา"),
             item(2, "diff", 10, ty="ปี ABC ปี", te="ปี ABCปี"),
             item(3, "tesseract_only", 10.5, te="2568 2567"),
             item(4, "tesseract_only", 10.6, te="หมายเหตุ"),
             item(5, "typhoon_only", 11, ty="2567 หมายเหตุ ปี ABC 2568")]
    judgments = {"F_p1_002": verdict("ปี ABCปี", "tesseract"), "F_p1_003": verdict("2568 2567", "tesseract"),
                 "F_p1_004": verdict("หมายเหตุ", "tesseract"), "F_p1_005": verdict("", "absent")}
    return items, judgments, {f"F_p1_00{n}": "table_header_restructured" for n in (2, 3, 4, 5)}


def test_table_header_category_removed_from_truth_hypotheses_missed_fabricated_and_errors(ev):
    items, judgments, header = header_page()
    r = ev.evaluate_page(page(items), judgments, excluded={}, diagram_ranges={}, table_header=header)
    assert r["truth_chars"] == len("เนื้อหา") and r["truth_items"] == 1
    assert r["cer"]["typhoon"]["distance"] == 0 and r["cer"]["tesseract"]["distance"] == 0
    assert r["missed"] == [] and r["fabricated"] == [] and r["errors"] == []
    assert r["table_header"]["ids"] == sorted(header)


def test_table_header_scored_order_insensitively_like_diagram(ev):
    items, judgments, header = header_page()
    h = ev.evaluate_page(page(items), judgments, excluded={}, diagram_ranges={}, table_header=header)["table_header"]
    assert h["truth_lines"] == 3
    assert h["tesseract"]["recall"] == 1.0 and h["tesseract"]["precision"] == 1.0
    # Typhoon เรียงใหม่และรวมบรรทัด: n-gram ในบรรทัดเดิม ("2568 2567" ฯลฯ) ยังเจอได้เมื่ออยู่ในบรรทัดเดียวกัน แต่ไม่ลงโทษลำดับ
    assert 0 < h["typhoon"]["recall"] < 1.0 and h["typhoon"]["precision"] < 1.0
    # สลับลำดับบรรทัดของ hypothesis แล้วคะแนนต้องเท่าเดิม
    truth = ["ปี ABCปี", "2568 2567", "หมายเหตุ"]
    assert ev.ngram_scores(truth, truth[::-1]) == ev.ngram_scores(truth, truth)


def test_table_header_only_applies_to_listed_ids_and_missed_line_still_counted_elsewhere(ev):
    items, judgments, header = header_page()
    items.append(item(6, "tesseract_only", 20.5, te="หมายเหตุ"))
    judgments["F_p1_006"] = verdict("หมายเหตุ", "tesseract")  # ไม่อยู่ในหมวด = ยังนับเป็นอ่านตก
    r = ev.evaluate_page(page(items), judgments, excluded={}, diagram_ranges={}, table_header=header)
    assert [m["id"] for m in r["missed"]] == ["F_p1_006"]
    assert r["truth_chars"] == len("เนื้อหา" + "หมายเหตุ")


def test_default_table_header_ids_are_kbank87_142_to_145_and_184_stays_excluded(ev):
    assert set(ev.TABLE_HEADER_ITEMS) == {f"ONEREPORTKBANKT_p87_{n}" for n in (142, 143, 144, 145)}
    assert set(ev.TABLE_HEADER_ITEMS.values()) == {"table_header_restructured"}
    assert ev.EXCLUDED_ITEMS == {"ONEREPORTKBANKT_p87_184": "handwriting"}


def test_excluded_item_cannot_take_part_in_dedup(ev):
    items = [item(1, "tesseract_only", 0.5, te="x", bbox=(0, 10, 100, 40)), item(2, "typhoon_only", 1, ty="x", region=REGION)]
    judgments = {"F_p1_001": verdict("ชื่อ"), "F_p1_002": verdict("ชื่อ", "typhoon")}
    r = ev.evaluate_page(page(items), judgments, excluded={"F_p1_002": "handwriting"}, diagram_ranges={})
    assert r["dedup_pairs"] == []


def test_diagram_set_split_from_content_and_scored_separately(ev):
    items = [item(1, "agree", 0, ty="เนื้อหา", te="เนื้อหา"),
             item(2, "tesseract_only", 5.5, te="BOX A", bbox=(0, 200, 90, 230)),
             item(3, "tesseract_only", 5.6, te="junk", bbox=(0, 240, 90, 270)),
             item(4, "tesseract_only", 5.7, te="นอกช่วง", bbox=(0, 300, 90, 330))]
    judgments = {"F_p1_002": verdict("BOX A", "tesseract"), "F_p1_003": verdict("", "absent"), "F_p1_004": verdict("นอกช่วง", "tesseract")}
    r = ev.evaluate_page(page(items, figures=["BOX A BOX B"]), judgments, excluded={}, diagram_ranges={"F_p1": [(2, 3)]})
    assert r["diagram_ids"] == ["F_p1_002", "F_p1_003"]
    assert r["truth_chars"] == len("เนื้อหา" + "นอกช่วง")  # (ข) ไม่อยู่ในเนื้อความ
    assert [m["id"] for m in r["missed"]] == ["F_p1_004"]  # (ข) ไม่นับเป็นอ่านตก
    assert r["diagram"]["truth_lines"] == 1
    assert r["diagram"]["typhoon"]["recall"] == 1.0 and r["diagram"]["typhoon"]["precision"] < 1.0
    assert r["diagram"]["tesseract"]["recall"] == 1.0 and r["diagram"]["tesseract"]["precision"] < 1.0  # "junk" เป็นตัวเกิน
    assert r["cer"]["tesseract"]["distance"] == 0  # hypothesis ของ Tesseract ไม่มีบรรทัด (ข) จึงตรงกับ truth เนื้อความ


def test_diagram_lines_excluded_from_tesseract_hypothesis(ev):
    items = [item(1, "agree", 0, ty="AB", te="AB"), item(2, "tesseract_only", 1.5, te="DIAGRAM", bbox=(0, 200, 90, 230))]
    judgments = {"F_p1_002": verdict("DIAGRAM", "tesseract")}
    r = ev.evaluate_page(page(items), judgments, excluded={}, diagram_ranges={"F_p1": [(2, 2)]})
    assert r["cer"]["tesseract"]["distance"] == 0 and r["truth_chars"] == 2


# ---------- error ระดับคำ / agree / เลขหน้า ----------

def test_word_error_shows_typhoon_word_with_context(ev):
    errs = ev.word_errors(["x"], "ซึ่งรวมถึงวิธีการระบุระดับการเพิ่มขึ้น", "ซึ่งรวมถึงวิธีการระบุดับการเพิ่มขึ้น")
    assert any("ระบุดับ" in e["typhoon_context"] for e in errs)
    assert ev.word_errors(["x"], "วิธีการ ระบุ", "วิธีการระบุ") == []  # ต่างแค่ช่องว่าง


def test_adjacent_boundary_shift_is_not_reported_as_typhoon_error(ev):
    # "2568" อยู่รายการที่ 1 ของข้อความที่ถูก แต่ Typhoon ตัดไปอยู่รายการที่ 2 — รวมเป็นช่วงเดียวแล้วเท่ากัน
    items = [item(1, "diff", 0, ty="ประจำปี", te="x"), item(2, "diff", 10, ty="2568 ธนาคาร", te="x")]
    judgments = {"F_p1_001": verdict("ประจำปี 2568"), "F_p1_002": verdict("ธนาคาร")}
    assert ev.typhoon_errors(items, judgments, {}) == []
    judgments["F_p1_002"] = verdict("ธนาคารไทย")  # ถ้ามีผิดจริงต่อจากนั้นยังเห็น
    assert ev.typhoon_errors(items, judgments, {})


def test_agree_audit_counts_changed_text_only_in_sample(ev):
    items = [item(1, "agree", 0, ty="AAA", te="AAA", audit=True, reason="sample", auto=True),
             item(2, "agree", 5, ty="BBB", te="BBB", audit=True, reason="sample", auto=True),
             item(3, "agree", 10, ty="CCC", te="CCC", audit=True, reason="cut_mid", auto=False),
             item(4, "agree", 15, ty="DDD", te="DDD")]
    judgments = {"F_p1_001": verdict("AAA", "typhoon"), "F_p1_002": verdict("BBX"), "F_p1_003": verdict("CCX")}
    a = ev.evaluate_page(page(items), judgments, excluded={}, diagram_ranges={})["agree"]
    assert a["sampled"] == 2 and [c["id"] for c in a["sample_changed"]] == ["F_p1_002"]
    assert a["cut_mid"] == 1 and len(a["cut_mid_changed"]) == 1  # แยกจากกลุ่มสุ่ม ไม่ปนในอัตราผิด
    assert a["auto_accepted"] == 3  # agree ที่ auto_accept (สุ่มตรวจ 2 + ไม่ตรวจ 1)


def test_page_number_check_uses_expected_table(ev):
    p = page([item(1, "agree", 0, ty="A", te="A")], numbers=[41])
    p["file"], p["page"] = "ONEREPORTPTTT.PDF", 43
    assert ev.evaluate_page(p, {}, excluded={}, diagram_ranges={})["page_numbers"]["ok"]
    p["typhoon_page_numbers"] = [42]
    assert not ev.evaluate_page(p, {}, excluded={}, diagram_ranges={})["page_numbers"]["ok"]


def test_missing_judgment_for_judged_kind_is_an_error(ev):
    with pytest.raises(KeyError):
        ev.evaluate_page(page([item(1, "diff", 0, ty="A", te="B")]), {}, excluded={}, diagram_ranges={})
