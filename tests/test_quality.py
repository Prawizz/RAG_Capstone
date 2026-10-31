"""ด่านตรวจคุณภาพ (app/extraction/quality.py) ด้วย Unit สังเคราะห์"""

import hashlib

import pytest

from app.extraction.quality import (
    BROKEN_EVENTS_THRESHOLD,
    MIN_THAI_CHARS,
    apply_document_rule,
    apply_gate,
    dictionary_ratio,
    evaluate_unit,
    gate_key,
    gate_summary,
    latin_in_thai_count,
    thai_char_count,
)
from app.models import (
    ExtractionMethod,
    QualityFlag,
    QualityReport,
    Unit,
    UnitStatus,
    UnitType,
)

CLEAN = "ธนาคารมีนโยบายกำกับดูแลกิจการที่ดี "  # ข้อความไทยสะอาด


def thai_text(n: int) -> str:
    """ข้อความไทยที่มีอักษรไทยพอดี n ตัว (ซ้ำ CLEAN แล้วตัดที่ตัวพยัญชนะ)"""
    out = ""
    while thai_char_count(out) < n:
        out += CLEAN
    cut, count = [], 0
    for ch in out:
        if "ก" <= ch <= "๛":
            if count == n:
                break
            count += 1
        cut.append(ch)
    return "".join(cut)


def make_unit(text="", *, coverage=None, fffd=0, marks=0, tone=0, dangling=0, status=UnitStatus.OK,
              flags=None, page=1) -> Unit:
    signals = {"char_count": float(len(text.strip())), "block_count": 1.0}
    if coverage is not None:
        signals["image_coverage"] = coverage
    for name, n in [("fffd_remaining", fffd), ("invalid_mark_sequence", marks),
                    ("conflicting_tone_marks", tone), ("dangling_mark", dangling)]:
        if n:
            signals[f"norm_sig:{name}"] = float(n)
    return Unit(
        index=page - 1,
        type=UnitType.PAGE,
        position={"page": page},
        label=f"หน้า {page}",
        status=status,
        reason=None if status == UnitStatus.OK else "x",
        method=ExtractionMethod.TEXT_LAYER,
        text=text,
        text_hash=hashlib.sha256(text.encode()).hexdigest(),
        quality=QualityReport(flags=flags if flags is not None else ([] if text.strip() else [QualityFlag.EMPTY])),
        signals=signals,
    )


def flags_of(u: Unit) -> set:
    return {f.value for f in u.quality.flags}


# ---------- กฎ a–e ----------

def test_a_empty_text_without_big_image_keeps_empty_rule():
    for cov in (None, 0.1, 0.49):
        u = evaluate_unit(make_unit("", coverage=cov))
        assert u.status == UnitStatus.OK and flags_of(u) == {"empty"} and u.reason is None
        assert gate_key(u) == "ok:empty"


def test_b_before_a_empty_text_with_big_image_is_scan_like():
    u = evaluate_unit(make_unit("", coverage=0.6))
    assert u.status == UnitStatus.NEEDS_OCR and "scan_like" in flags_of(u) and "OCR" in u.reason
    assert gate_key(u) == "needs_ocr:scan_like"


@pytest.mark.parametrize("thai,coverage,expected", [
    (99, 0.5, "needs_ocr"),
    (99, 0.49, "ok"),
    (100, 0.9, "ok"),  # ข้อความพอ ภาพใหญ่ก็ไม่ต้อง OCR
    (99, None, "ok"),  # ไม่มี image_coverage (เช่น docx) ข้ามกฎ b
    (0, 0.5, "needs_ocr"),
])
def test_b_scan_like_edges(thai, coverage, expected):
    text = thai_text(thai) if thai else "page 1"
    u = evaluate_unit(make_unit(text, coverage=coverage))
    assert u.status.value == expected
    if expected == "needs_ocr":
        assert flags_of(u) == {"scan_like"} and "OCR" in u.reason and str(thai) in u.reason
    else:
        assert u.reason is None


@pytest.mark.parametrize("broken,expected", [(BROKEN_EVENTS_THRESHOLD - 1, "ok"), (BROKEN_EVENTS_THRESHOLD, "needs_ocr")])
def test_c_broken_events_edges(broken, expected):
    u = evaluate_unit(make_unit(thai_text(300), fffd=broken))
    assert u.status.value == expected
    assert u.signals["gate:broken_events"] == broken


def test_c_flags_and_reason_by_kind():
    u = evaluate_unit(make_unit(thai_text(300) + " เน่Çอหา", fffd=12, marks=20))
    assert u.status == UnitStatus.NEEDS_OCR
    assert flags_of(u) == {"replacement_char", "invalid_mark_sequence", "latin_in_thai"}
    assert u.reason == "พบจุดพัง 33 จุด (�: 12, สระซ้อน: 20, ละตินกลางคำไทย: 1)"
    assert u.signals["gate:broken_events"] == 33 and u.signals["gate:latin_in_thai"] == 1


def test_c_tone_and_dangling_use_invalid_mark_sequence_flag():
    u = evaluate_unit(make_unit(thai_text(300), tone=2, dangling=1))
    assert u.status == UnitStatus.NEEDS_OCR and flags_of(u) == {"invalid_mark_sequence"}
    assert "วรรณยุกต์ซ้อน: 2" in u.reason and "เครื่องหมายลอย: 1" in u.reason


def test_c_applies_even_when_text_is_short():
    u = evaluate_unit(make_unit("กขค", fffd=3))
    assert u.status == UnitStatus.NEEDS_OCR and flags_of(u) == {"replacement_char"}


def test_b_wins_over_c():
    u = evaluate_unit(make_unit("กขค", coverage=0.8, fffd=5))
    assert flags_of(u) == {"scan_like"}


@pytest.mark.parametrize("thai,expected_flags", [(MIN_THAI_CHARS - 1, {"low_text"}), (MIN_THAI_CHARS, set())])
def test_d_e_low_text_edge(thai, expected_flags):
    u = evaluate_unit(make_unit(thai_text(thai), coverage=0.1))
    assert u.status == UnitStatus.OK and flags_of(u) == expected_flags
    assert u.signals["gate:thai_chars"] == thai


def test_non_ok_units_are_not_touched():
    failed = make_unit("", status=UnitStatus.FAILED)
    assert evaluate_unit(failed) is failed


def test_score_stays_none_and_signals_are_flat():
    u = evaluate_unit(make_unit(thai_text(200), fffd=5))
    assert u.quality.score is None
    assert all(isinstance(v, float) for v in u.signals.values())
    assert {"gate:thai_chars", "gate:broken_events", "gate:latin_in_thai"} <= u.signals.keys()


# ---------- latin_in_thai ----------

@pytest.mark.parametrize("text", [
    "ปตท. (PTT) ประกาศผล",
    "รุ่น iPhone 15 ราคา 30,000 บาท",
    "บริษัท KBank จำกัด",
    "ราคา/PTT",
    "ข้อมูล: ABC",
    "Annual Report ประจำปี",
])
def test_correct_mixed_thai_english_is_not_latin_in_thai(text):
    assert latin_in_thai_count(text) == 0


@pytest.mark.parametrize("text,n", [
    ("เน่Çอหา", 1),
    ("แกmผ่ถ่อหุ้น", 1),
    ("กขAB", 1),
    ("ABกข", 1),
    ("กxขyค", 2),
])
def test_latin_glued_to_thai_is_counted_once_per_run(text, n):
    assert latin_in_thai_count(text) == n


# ---------- กฎระดับไฟล์ ----------

def doc_units(good: int, broken: int, thin: int = 0):
    units = []
    for i in range(broken):
        units.append(make_unit(thai_text(150), fffd=5, page=len(units) + 1))
    for i in range(good):
        units.append(make_unit(thai_text(150), page=len(units) + 1))
    for i in range(thin):
        units.append(make_unit(thai_text(20), page=len(units) + 1))
    return apply_gate(units)


def test_document_rule_triggers_over_half():
    out = doc_units(good=2, broken=3)  # พัง 3 จาก 5
    assert [u.status for u in out].count(UnitStatus.NEEDS_OCR) == 5
    swept = [u for u in out if QualityFlag.DOCUMENT_FONT_BROKEN in u.quality.flags]
    assert len(swept) == 2 and "3 จาก 5" in swept[0].reason
    assert gate_summary(out)["needs_ocr:document_font_broken"] == 2


def test_document_rule_exactly_half_does_not_trigger():
    out = doc_units(good=2, broken=2)
    assert [u.status for u in out].count(UnitStatus.OK) == 2
    assert not any(QualityFlag.DOCUMENT_FONT_BROKEN in u.quality.flags for u in out)


def test_document_rule_leaves_thin_units_alone():
    out = doc_units(good=1, broken=3, thin=2)
    assert [gate_key(u) for u in out].count("ok:low_text") == 2
    assert [gate_key(u) for u in out].count("needs_ocr:document_font_broken") == 1


def test_document_rule_ignores_scan_like_when_counting():
    scans = [make_unit(thai_text(20), coverage=0.9, page=i + 10) for i in range(10)]
    units = [make_unit(thai_text(150), page=1), make_unit(thai_text(150), page=2)]
    out = apply_gate(units + scans)
    assert [u.status for u in out[:2]] == [UnitStatus.OK, UnitStatus.OK]  # ภาพ 10 หน้าไม่นับเป็นหน้าพัง


def test_document_rule_without_rich_units_is_noop():
    units = [evaluate_unit(make_unit("กขค", fffd=4))]
    assert apply_document_rule(units) == units


# ---------- พจนานุกรม (วัดอย่างเดียว) ----------

def test_gate_does_not_compute_dictionary_ratio():
    u = evaluate_unit(make_unit(thai_text(300)))
    assert u.quality.dictionary_ratio is None


def test_dictionary_ratio_separates_gibberish_from_words():
    assert dictionary_ratio("ธนาคารมีนโยบายกำกับดูแลกิจการที่ดี") > 0.9
    assert dictionary_ratio("ธนคารมีนยบายกกำบดแลกจการทีดี" * 3) < dictionary_ratio("ธนาคารมีนโยบายกำกับดูแลกิจการที่ดี")
    assert dictionary_ratio("hello") is None


def test_dictionary_ratio_does_not_decide_status():
    # ข้อความสะอาดตามจุดพัง แต่ไม่ใช่คำในพจนานุกรม => ยัง ok (วัดอย่างเดียว)
    text = "กขคงจฉชซฌญ" * 15
    u = evaluate_unit(make_unit(text))
    assert u.status == UnitStatus.OK and dictionary_ratio(text) < 0.5


# ---------- mojibake ----------

from app.extraction.quality import mojibake_count  # noqa: E402


@pytest.mark.parametrize("text,n", [
    ("¢ŒÒ¾ÃÐ¾Ø·¸à¨ŒÒ", 14),  # KBANK หน้า 2
    ("abc ¡¢£ def", 3),
    ("ก™€‰ข", 3),  # อักขระพิเศษ cp1252
    ("¡¢ x £¤¥", 3),  # ท่อนสั้นกว่า 3 ไม่นับ
])
def test_mojibake_run_counts_characters(text, n):
    assert mojibake_count(text) == n


@pytest.mark.parametrize("text", [
    "ลิขสิทธิ์ © 2568 บริษัท",
    "สินค้าจดทะเบียน ® และ ™",
    "อุณหภูมิ 25° ราคา 3×4",
    "Société Générale",
    "ร้าน café ในกรุงเทพ",
    "© ® °",
    "ก×ข°ค©ง",
])
def test_correct_text_with_symbols_or_accents_is_not_mojibake(text):
    assert mojibake_count(text) == 0
    assert evaluate_unit(make_unit(thai_text(150) + " " + text)).signals["gate:broken_events"] == 0


def test_mojibake_counts_into_broken_events_flag_and_reason():
    u = evaluate_unit(make_unit(thai_text(150) + " ¢ŒÒ¾ÃÐ¾Ø·¸à¨ŒÒ"))
    assert u.status == UnitStatus.NEEDS_OCR and flags_of(u) == {"latin_in_thai"}
    assert u.signals["gate:mojibake"] == 14 and u.signals["gate:broken_events"] == 14
    assert "ข้อความเข้ารหัสเพี้ยน: 14" in u.reason


def test_mojibake_touching_thai_is_not_double_counted_as_latin_in_thai():
    u = evaluate_unit(make_unit(thai_text(150) + "กข¢ŒÒ¾ค"))
    assert u.signals["gate:mojibake"] == 4 and u.signals["gate:latin_in_thai"] == 0
