"""scripts/ocr_truth_prepare.py: จับคู่บรรทัด Typhoon กับ Tesseract แล้วแบ่งเป็น 4 แบบ (ข้อมูลสังเคราะห์)"""

import importlib.util
import io
import json
import re
from pathlib import Path

import pytest

from app.extraction.ocr import OcrResult

ROOT = Path(__file__).parent.parent
NL = chr(10)


@pytest.fixture(scope="module")
def prep():
    spec = importlib.util.spec_from_file_location("ocr_truth_prepare", ROOT / "scripts" / "ocr_truth_prepare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ty(text):
    return OcrResult(text=text, backend="typhoon-local", model="m", revision="r", seconds=1.0, tokens=10)


def te(lines, parts=None):
    """lines: list ของข้อความ หรือ (ข้อความ, conf) — วาง bbox ซ้อนลงมาทีละบรรทัด (ภาพ 1000x1000)"""
    out = []
    for i, ln in enumerate(lines):
        text, conf = ln if isinstance(ln, tuple) else (ln, 90.0)
        out.append({"text": text, "bbox": [50, 40 * i + 10, 900, 40 * i + 40], "conf": conf, "block": 1, "par": 1})
    extra = {"size": [1000, 1000], "lines": out}
    return OcrResult(text=NL.join(l["text"] for l in out), backend="tesseract-local", model="t", revision="v",
                     seconds=2.0, extra=extra if parts is None else {"parts": parts})


def kinds(data):
    return [(it["kind"], (it["typhoon"] or {}).get("text"), (it["tesseract"] or {}).get("text")) for it in data["items"]]


L1 = "ธนาคารกสิกรไทยรายงานประจำปีสองพันห้าร้อยหกสิบแปด"
L2 = "ข้าพเจ้าได้ประเมินความเสี่ยงโดยพิจารณาปัจจัยภายในและภายนอก"
L3 = "เงินให้สินเชื่อแก่ลูกหนี้คิดเป็นร้อยละห้าสิบสี่"


def test_exact_match_ignoring_markdown_and_spaces(prep):
    md = NL.join(["# " + L1, "", "* " + L2, "", L3])
    data = prep.build_items("f.PDF", 1, ty(md), te([L1.replace("ไทย", " ไทย"), L2, L3]))
    assert [it["kind"] for it in data["items"]] == ["agree", "agree", "agree"]
    assert data["counts"]["diff"] == 0 and data["counts"]["typhoon_only"] == 0


def test_one_character_difference_is_diff(prep):
    a = "ของแต่ละบริษัทให้เกิดประโยชน์สูงสุดพร้อมทั้งวางแนวทางการลงทุน"
    b = a.replace("แนว", "แผน")
    data = prep.build_items("f.PDF", 1, ty(b), te([a]))
    (it,) = data["items"]
    assert it["kind"] == "diff" and it["tesseract"]["text"] == a and 70 <= it["score"] < 100
    assert it["suggest"] in ("typhoon", "tesseract") and it["tesseract"]["bbox"] == [50, 10, 900, 40]


def test_typhoon_only_when_nothing_similar(prep):
    invented = "ข้อมูลเพิ่มเติมเกณฑ์การประเมินความเสี่ยงและตัวชี้วัดความเสี่ยง"
    data = prep.build_items("f.PDF", 1, ty(NL.join([L1, invented, L2])), te([L1, L2]))
    k = [it["kind"] for it in data["items"]]
    assert k == ["agree", "typhoon_only", "agree"]
    assert data["items"][1]["tesseract"] is None and data["counts"]["typhoon_only"] == 1


def test_typhoon_only_region_between_neighbours(prep):
    invented = "ข้อมูลเพิ่มเติมเกณฑ์การประเมินความเสี่ยงและตัวชี้วัดความเสี่ยง"
    data = prep.build_items("f.PDF", 1, ty(NL.join([L1, invented, L2])), te([L1, L2]))
    r = data["items"][1]["region"]
    assert r["image"] == 0 and r["bbox"][1] < 40 and r["bbox"][3] > 50  # ช่วงแนวตั้งคร่อมระหว่างสองบรรทัดข้างเคียง


def test_tesseract_only_line_missed_by_typhoon(prep):
    missed = "หมายเหตุประกอบงบการเงินรวมข้อสี่จุดสองจุดสอง"
    data = prep.build_items("f.PDF", 1, ty(NL.join([L1, L3])), te([L1, missed, L3]))
    k = [it["kind"] for it in data["items"]]
    assert k.count("tesseract_only") == 1 and data["counts"]["tesseract_only"] == 1
    t = next(it for it in data["items"] if it["kind"] == "tesseract_only")
    assert t["tesseract"]["text"] == missed and t["typhoon"] is None
    ends = [it["pos"] for it in data["items"] if it["kind"] == "agree"]
    assert ends[0] < t["pos"] < ends[1]  # อยู่ระหว่างบรรทัดแรกกับบรรทัดที่สองของ Typhoon ตามลำดับอ่าน


def test_order_independent_two_columns(prep):
    left = [L1, L2]
    right = [L3, "บริษัทและบริษัทย่อยประมาณการและบันทึกการด้อยค่าของเงินให้สินเชื่อ"]
    # Typhoon อ่านซ้ายทั้งคอลัมน์แล้วขวา; Tesseract อ่านสลับตามแถว
    ty_md = NL.join(left + right)
    tess = [left[0], right[0], left[1], right[1]]
    data = prep.build_items("f.PDF", 1, ty(ty_md), te(tess))
    assert [it["kind"] for it in data["items"]] == ["agree"] * 4


def test_wrapped_paragraph_is_judged_per_visual_line(prep):
    para = L1 + " " + L2 + " " + L3  # Typhoon เขียนเป็นบรรทัดเดียว
    data = prep.build_items("f.PDF", 1, ty(para), te([L1, L2, L3]))
    assert [it["kind"] for it in data["items"]] == ["agree"] * 3
    assert [it["tesseract"]["text"] for it in data["items"]] == [L1, L2, L3]
    assert [it["typhoon"]["text"] for it in data["items"]] == [L1, L2, L3]  # ตัดข้อความ Typhoon ตามบรรทัดในภาพ
    assert [it["tesseract"]["bbox"][1] for it in data["items"]] == [10, 50, 90]


def test_paragraph_continuing_across_columns_in_tesseract_order(prep):
    """Tesseract อ่านซ้ายทั้งคอลัมน์ก่อน (ย่อหน้าต่อกันข้ามคอลัมน์) — ไม่ต้องติดกันก็จับคู่ได้ และเรียงตาม Typhoon"""
    ty_md = L1 + " " + L2 + " " + L3
    data = prep.build_items("f.PDF", 1, ty(ty_md), te([L3, L1, L2]))  # ลำดับ Tesseract ต่างจาก Typhoon
    assert [it["kind"] for it in data["items"]] == ["agree"] * 3
    assert [it["tesseract"]["text"] for it in data["items"]] == [L1, L2, L3]  # เรียงตามตำแหน่งใน Typhoon


def test_diff_line_inside_long_paragraph_is_isolated(prep):
    bad = L2.replace("ความเสี่ยง", "ความเสียง")
    data = prep.build_items("f.PDF", 1, ty(L1 + " " + bad + " " + L3), te([L1, L2, L3]))
    assert [it["kind"] for it in data["items"]] == ["agree", "diff", "agree"]
    assert data["items"][1]["typhoon"]["text"] == bad and data["items"][1]["tesseract"]["text"] == L2


def test_junk_and_page_number_tesseract_lines_skipped(prep):
    res = ty("<page_number>158</page_number>" + NL + L1)
    data = prep.build_items("f.PDF", 1, res, te(["158", ("x", 90.0), ("ขยะจากกราฟิก", 10.0), L1]))
    assert [it["kind"] for it in data["items"]] == ["agree"]
    reasons = sorted(s["reason"] for s in data["skipped_tesseract"])
    assert len(reasons) == 3 and any("เลขหน้า" in r for r in reasons) and any("conf" in r for r in reasons)
    assert data["typhoon_page_numbers"] == [158]


def test_figure_text_is_kept_apart_and_never_truth(prep):
    md = L1 + NL + NL + "<figure>" + NL + "แผนภูมิแท่งแสดงรายได้" + NL + "</figure>"
    data = prep.build_items("f.PDF", 1, ty(md), te([L1]))
    assert data["figures"] == ["แผนภูมิแท่งแสดงรายได้"]
    assert all("แผนภูมิ" not in (it["typhoon"] or {}).get("text", "") for it in data["items"])
    assert data["counts"]["typhoon_only"] == 0


def test_table_rows_are_lines(prep):
    md = "<table><tr><td>" + L1 + "</td><td>1,000</td></tr><tr><td>" + L2 + "</td><td>2,000</td></tr></table>"
    data = prep.build_items("f.PDF", 1, ty(md), te([L1 + " 1,000", L2 + " 2,000"]))
    assert [it["kind"] for it in data["items"]] == ["agree", "agree"]


def test_audit_sample_is_ten_percent_with_fixed_seed(prep):
    lines = [f"บรรทัดทดสอบหมายเลข{'ก' * (i % 5)}{i:03d}ของเอกสารสำหรับการสุ่มตรวจ" for i in range(30)]
    md = NL.join(lines)
    d1 = prep.build_items("f.PDF", 7, ty(md), te(lines))
    d2 = prep.build_items("f.PDF", 7, ty(md), te(lines))
    a1 = [it["id"] for it in d1["items"] if it["audit"]]
    assert len(a1) == 3 and a1 == [it["id"] for it in d2["items"] if it["audit"]]
    assert d1["counts"]["agree"] == 27 and d1["counts"]["agree_audit"] == 3 and d1["counts"]["to_judge"] == 3
    d3 = prep.build_items("f.PDF", 8, ty(md), te(lines))  # หน้าอื่น -> เลือกชุดอื่นได้ แต่ยังได้ 10%
    assert sum(it["audit"] for it in d3["items"]) == 3


def test_at_least_one_audit_item_when_few_agree(prep):
    d = prep.build_items("f.PDF", 1, ty(L1), te([L1]))
    assert d["counts"]["agree_audit"] == 1 and d["counts"]["agree"] == 0


def test_counts_split_into_four_kinds(prep):
    invented = "ข้อมูลเพิ่มเติมเกณฑ์การประเมินความเสี่ยงและตัวชี้วัดความเสี่ยง"
    diff_a = "ของแต่ละบริษัทให้เกิดประโยชน์สูงสุดพร้อมทั้งวางแนวทางการลงทุน"
    missed = "หมายเหตุประกอบงบการเงินรวมข้อสี่จุดสองจุดสอง"
    d = prep.build_items("f.PDF", 1, ty(NL.join([L1, diff_a.replace("แนว", "แผน"), invented])), te([L1, diff_a, missed]))
    c = d["counts"]
    assert (c["agree"] + c["agree_audit"], c["diff"], c["typhoon_only"], c["tesseract_only"]) == (1, 1, 1, 1)
    assert c["to_judge"] == c["agree_audit"] + 3


def test_split_page_uses_parts_and_image_index(prep):
    p0 = {"size": [500, 500], "lines": [{"text": L1, "bbox": [10, 10, 400, 40], "conf": 90, "block": 1, "par": 1}]}
    p1 = {"size": [500, 500], "lines": [{"text": L3, "bbox": [20, 100, 450, 130], "conf": 90, "block": 1, "par": 1}]}
    d = prep.build_items("f.PDF", 1, ty(L1 + NL + NL + L3), te([], parts=[p0, p1]))
    assert [it["tesseract"]["image"] for it in d["items"]] == [0, 1]


def test_items_ids_unique_and_json_serialisable(prep):
    d = prep.build_items("A_B.PDF", 442, ty(NL.join([L1, L2])), te([L1, L2]))
    assert len({it["id"] for it in d["items"]}) == 2 and d["items"][0]["id"].startswith("A_B_p442_")
    json.dumps(d, ensure_ascii=False)


# ---------- ภาพครอป + HTML ----------

def png(w=1000, h=1000):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "PNG")
    return buf.getvalue()


def test_attach_images_crops_only_items_to_judge(prep):
    invented = "ข้อมูลเพิ่มเติมเกณฑ์การประเมินความเสี่ยงและตัวชี้วัดความเสี่ยง"
    lines = [f"บรรทัดทดสอบหมายเลข{i:03d}ของเอกสารสำหรับการสุ่มตรวจ" for i in range(12)]
    d = prep.build_items("f.PDF", 1, ty(NL.join(lines[:6] + [invented] + lines[6:])), te(lines))
    page = prep.attach_images(d, [png()])
    kinds_left = sorted(it["kind"] for it in page["items"])
    assert "typhoon_only" in kinds_left and sum(k == "agree" for k in kinds_left) == 1  # 11 ตรงกัน -> สุ่มตรวจ 1
    assert len(page["items"]) == d["counts"]["to_judge"]
    crops = [it["crop"] for it in page["items"]]
    assert all(c is None or c.startswith("data:image/jpeg;base64,") for c in crops)
    ty_only = next(it for it in page["items"] if it["kind"] == "typhoon_only")
    assert ty_only["crop"] is not None  # มีเพื่อนบ้านทั้งสองข้าง -> ครอปเป็นแถบ
    assert page["page_images"][0].startswith("data:image/jpeg")


def test_typhoon_only_without_neighbours_has_no_crop(prep):
    d = prep.build_items("f.PDF", 1, ty("ข้อมูลเพิ่มเติมเกณฑ์การประเมินความเสี่ยงและตัวชี้วัดความเสี่ยง"), te([L1]))
    page = prep.attach_images(d, [png()])
    it = next(i for i in page["items"] if i["kind"] == "typhoon_only")
    assert it["crop"] is None  # หน้าตรวจจะแสดงภาพหน้าเต็มแทน


def test_build_html_embeds_valid_json_and_escapes_script_end(prep):
    d = prep.build_items("f.PDF", 1, ty("</script><b>x</b> " + L1), te([L1]))
    html = prep.build_html([prep.attach_images(d, [png()])])
    assert "__DATA__" not in html
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    data = json.loads(m.group(1).replace("<\\/", "</"))
    assert data["total"] == len(data["pages"][0]["items"]) and data["dataset_id"]
    assert html.count("</script>") == 2  # เฉพาะแท็กปิดจริงของ data กับสคริปต์หลัก


def test_dataset_id_depends_on_item_ids(prep):
    p1 = prep.attach_images(prep.build_items("f.PDF", 1, ty(L1), te([L1])), [png()])
    p2 = prep.attach_images(prep.build_items("f.PDF", 2, ty(L1), te([L1])), [png()])
    get = lambda h: json.loads(re.search(r'type="application/json">(.*?)</script>', h, re.S).group(1))["dataset_id"]
    assert get(prep.build_html([p1])) != get(prep.build_html([p2]))


def test_review_page_works_in_a_fake_dom(prep, tmp_path):
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("ไม่มี node")
    invented = "ข้อมูลเพิ่มเติมเกณฑ์การประเมินความเสี่ยงและตัวชี้วัดความเสี่ยง"
    a = "ของแต่ละบริษัทให้เกิดประโยชน์สูงสุดพร้อมทั้งวางแนวทางการลงทุน"
    d = prep.build_items("f.PDF", 1, ty(NL.join([L1, a.replace("แนว", "แผน"), invented])), te([L1, a]))
    f = tmp_path / "review.html"
    f.write_text(prep.build_html([prep.attach_images(d, [png()])]), encoding="utf-8")
    r = subprocess.run([node, str(ROOT / "tests" / "ocr_review_smoke.js"), str(f)], capture_output=True, timeout=60)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    out = json.loads(r.stdout.decode("utf-8"))
    n = d["counts"]["to_judge"]
    assert out["countInitial"] == f"ทำแล้ว 0 / {n} รายการ" and out["firstCardHasKind"] and out["hasPageOptions"]
    assert out["countAfterOne"] == f"ทำแล้ว 1 / {n} รายการ" and out["countAfterTwo"] == f"ทำแล้ว 2 / {n} รายการ"
    assert out["download"]["done"] == 2 and out["download"]["total"] == n and out["download"]["verdicts"][0] == "absent"
    assert out["countAfterReload"] == f"ทำแล้ว 2 / {n} รายการ"  # ความคืบหน้าอยู่รอดหลังโหลดหน้าใหม่
    assert "รายการที่ 1 / " + str(n - 2) in out["posUndone"]


def test_similar_neighbouring_lines_are_not_cross_matched(prep):
    a = "4. การวิเคราะห์และคำอธิบายของฝ่ายจัดการ"
    b = "5. ข้อมูลทั่วไปและข้อมูลสำคัญอื่น"
    d = prep.build_items("f.PDF", 1, ty(NL.join([a, b])), te([a]))  # Tesseract อ่านตกบรรทัด b
    assert [it["kind"] for it in d["items"]] == ["agree", "typhoon_only"]
    assert d["items"][0]["tesseract"]["text"] == a


def test_best_run_does_not_swallow_unrelated_neighbour_line(prep):
    d = prep.build_items("f.PDF", 1, ty(L2), te([L1, L2, L3]))
    (it,) = [i for i in d["items"] if i["typhoon"]]
    assert it["tesseract"]["text"] == L2 and it["kind"] == "agree"
    assert {i["tesseract"]["text"] for i in d["items"] if i["kind"] == "tesseract_only"} == {L1, L3}


# ---------- กรอง fragment / duplicate (T06b ช่วงที่ 1 แก้) ----------

def te_boxes(rows, size=(1000, 1000)):
    """rows: (ข้อความ, bbox [x0,y0,x1,y1], conf, ภาพที่) — วาง bbox เองเพื่อทดสอบการทับกัน"""
    parts = [{"size": list(size), "lines": []}, {"size": list(size), "lines": []}]
    for text, bbox, conf, *img in rows:
        parts[img[0] if img else 0]["lines"].append({"text": text, "bbox": bbox, "conf": conf, "block": 1, "par": 1})
    return OcrResult(text="", backend="tesseract-local", model="t", revision="v", seconds=2.0, extra={"parts": parts})


MISSED = "หมายเหตุประกอบงบการเงินรวมข้อสี่จุดสองจุดสอง"
JUNK = "ขยะจากลายน้ำสแกนบรรทัดนี้"


def by_text(d, text):
    return next(it for it in d["items"] if (it["tesseract"] or {}).get("text") == text)


def test_short_tesseract_only_box_is_fragment_and_not_counted(prep):
    d = prep.build_items("f.PDF", 1, ty(L1), te_boxes([(L1, [50, 100, 900, 130], 90), (JUNK, [60, 85, 200, 96], 60)]))  # สูง 11 px
    f = by_text(d, JUNK)
    assert f["kind"] == "fragment" and d["counts"]["fragment"] == 1 and d["counts"]["tesseract_only"] == 0
    assert d["counts"]["to_judge"] == d["counts"]["agree_audit"] == 1
    assert f["covered_by"] == [by_text(d, L1)["id"]]  # เป็นเศษของบรรทัดที่อยู่ติดกัน


def test_fragment_height_threshold_is_strict(prep):
    h = prep.FRAGMENT_MAX_HEIGHT
    d = prep.build_items("f.PDF", 1, ty(L1), te_boxes([(L1, [50, 300, 900, 330], 90), (JUNK, [60, 100, 200, 100 + h - 1], 60),
                                                        (MISSED, [60, 500, 800, 500 + h], 90)]))
    assert by_text(d, JUNK)["kind"] == "fragment" and by_text(d, MISSED)["kind"] == "tesseract_only"


def test_overlap_both_ways_is_duplicate_with_covered_by(prep):
    d = prep.build_items("f.PDF", 1, ty(L1), te_boxes([(L1, [50, 100, 900, 140], 94), (JUNK, [60, 105, 880, 135], 44)]))
    dup = by_text(d, JUNK)
    assert dup["kind"] == "duplicate" and dup["covered_by"] == [by_text(d, L1)["id"]] and dup["maybe_duplicate_of"] == []
    assert d["counts"]["duplicate"] == 1 and d["counts"]["to_judge"] == 1 and d["counts"]["maybe_duplicate"] == 0


def test_one_way_overlap_stays_to_judge_with_maybe_duplicate_of(prep):
    """บรรทัดจริงที่อยู่ในบรรทัดกว้างผิดปกติ (Tesseract รวมแถบข้างกับตาราง): ทับ 100% ของตัวเอง แต่เป็นสัดส่วนเล็กของบรรทัดกว้าง"""
    wide = [0, 100, 2000, 140]
    d = prep.build_items("f.PDF", 1, ty(L1), te_boxes([(L1, wide, 90), (MISSED, [100, 100, 800, 140], 87)], size=(2200, 1000)))
    it = by_text(d, MISSED)
    assert it["kind"] == "tesseract_only" and it["covered_by"] == []
    assert it["maybe_duplicate_of"] == [by_text(d, L1)["id"]]
    assert d["counts"]["tesseract_only"] == 1 and d["counts"]["duplicate"] == 0 and d["counts"]["maybe_duplicate"] == 1
    assert d["counts"]["to_judge"] == d["counts"]["agree_audit"] + 1


def test_duplicate_overlap_thresholds_are_inclusive_and_named(prep):
    assert prep.DUPLICATE_OVERLAP_SELF == 0.5 and prep.DUPLICATE_OVERLAP_OTHER == 0.5
    cover = [0, 100, 1000, 140]  # พื้นที่ 40,000
    # ตัวเองกว้าง 500 สูง 40 (20,000) อยู่ใน cover ทั้งหมด -> ทับ 100% ของตัวเอง, 50% ของ cover พอดี = duplicate
    d = prep.build_items("f.PDF", 1, ty(L1), te_boxes([(L1, cover, 90), (MISSED, [0, 100, 500, 140], 80)], size=(1200, 1000)))
    assert by_text(d, MISSED)["kind"] == "duplicate"
    # แคบลงหนึ่งพิกเซล -> ทับ cover < 50% = ทางเดียว (ยังให้คนตัดสิน)
    d = prep.build_items("f.PDF", 1, ty(L1), te_boxes([(L1, cover, 90), (MISSED, [0, 100, 499, 140], 80)], size=(1200, 1000)))
    it = by_text(d, MISSED)
    assert it["kind"] == "tesseract_only" and it["maybe_duplicate_of"]
    # ตัวเองทับ cover น้อยกว่า 50% ของตัวเอง (100/200 ... ลดอีกหนึ่งพิกเซล) -> ไม่เกี่ยวกัน
    d = prep.build_items("f.PDF", 1, ty(L1), te_boxes([(L1, cover, 90), (MISSED, [901, 100, 1101, 140], 80)], size=(1200, 1000)))
    it = by_text(d, MISSED)
    assert it["kind"] == "tesseract_only" and it["maybe_duplicate_of"] == [] and it["covered_by"] == []


def test_overlap_only_counts_within_the_same_image(prep):
    p = lambda b: {"size": [1000, 1000], "lines": [{"text": b[0], "bbox": b[1], "conf": 90, "block": 1, "par": 1}]}
    res = OcrResult(text="", backend="tesseract-local", model="t", revision="v", seconds=1.0,
                    extra={"parts": [p((L1, [50, 100, 900, 140])), p((MISSED, [60, 105, 880, 135]))]})
    d = prep.build_items("f.PDF", 1, ty(L1), res)
    assert by_text(d, MISSED)["kind"] == "tesseract_only" and d["counts"]["duplicate"] == 0


def test_overlap_with_tesseract_only_neighbour_is_not_a_duplicate(prep):
    """ผู้ทับต้องเป็น agree/diff เท่านั้น: สองบรรทัดที่ Typhoon อ่านตกทั้งคู่ ทับกัน = ยังให้คนตัดสินทั้งคู่"""
    d = prep.build_items("f.PDF", 1, ty(L1), te_boxes([(L1, [50, 10, 900, 40], 90), (MISSED, [50, 100, 900, 140], 90), (JUNK, [60, 105, 880, 135], 90)]))
    assert d["counts"]["tesseract_only"] == 2 and d["counts"]["duplicate"] == 0


def test_filtered_items_keep_positions_ids_and_order(prep):
    d = prep.build_items("f.PDF", 1, ty(NL.join([L1, L3])), te_boxes([
        (L1, [50, 100, 900, 140], 94), (JUNK, [60, 105, 880, 135], 44), (L3, [50, 300, 900, 340], 94), ("กข", [60, 290, 90, 298], 70)]))
    assert [it["order"] for it in d["items"]] == list(range(len(d["items"])))
    assert all(it["pos"] is not None for it in d["items"]) and len({it["id"] for it in d["items"]}) == len(d["items"])
    assert d["counts"]["duplicate"] == 1 and d["counts"]["fragment"] == 1
    json.dumps(d, ensure_ascii=False)


def test_attach_images_adds_related_texts_and_page_sizes(prep):
    wide = [0, 100, 2000, 140]
    d = prep.build_items("f.PDF", 1, ty(NL.join([L1, L3])), te_boxes([
        (L1, wide, 90), (MISSED, [100, 100, 800, 140], 87),  # ทับทางเดียว
        (L3, [50, 300, 900, 340], 94), (JUNK, [60, 305, 880, 335], 44), ("กข", [60, 290, 90, 298], 70)], size=(2200, 1000)))
    page = prep.attach_images(d, [png(2200, 1000)])
    maybe = next(it for it in page["items"] if it["kind"] == "tesseract_only")
    assert [r["relation"] for r in maybe["related"]] == ["maybe_duplicate_of"] and maybe["related"][0]["tesseract"] == L1
    assert maybe["related"][0]["typhoon"]  # แสดงข้อความ Typhoon ของรายการที่ทับด้วย
    dup = next(it for it in page["items"] if it["kind"] == "duplicate")
    assert dup["related"][0]["relation"] == "covered_by" and dup["related"][0]["tesseract"] == L3
    frag = next(it for it in page["items"] if it["kind"] == "fragment")
    assert frag["crop"] is not None and frag["related"]  # ภาพรวมรายการข้างเคียงไว้ให้เห็นบริบท
    assert page["page_sizes"][0] == [2200, 1000]


def test_total_and_dataset_id_exclude_filtered_items(prep):
    d = prep.build_items("f.PDF", 1, ty(L1), te_boxes([(L1, [50, 100, 900, 140], 94), (JUNK, [60, 105, 880, 135], 44), ("กข", [60, 90, 90, 98], 70)]))
    html = prep.build_html([prep.attach_images(d, [png()])])
    g = lambda h: json.loads(re.search(r'type="application/json">(.*?)</script>', h, re.S).group(1).replace("<\\/", "</"))
    data = g(html)
    assert data["total"] == d["counts"]["to_judge"] == 1 and len(data["pages"][0]["items"]) == 3  # กรองออกยังอยู่ในหน้าตรวจ
    d2 = prep.build_items("f.PDF", 1, ty(L1), te_boxes([(L1, [50, 100, 900, 140], 94)]))
    assert data["dataset_id"] == g(prep.build_html([prep.attach_images(d2, [png()])]))["dataset_id"]  # ขึ้นกับรายการที่ต้องตัดสินเท่านั้น


def test_review_page_filtered_items_and_wide_warning_in_fake_dom(prep, tmp_path):
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("ไม่มี node")
    d = prep.build_items("f.PDF", 1, ty(NL.join([L1, L3])), te_boxes([
        (L1, [50, 100, 900, 140], 94), (JUNK, [60, 105, 880, 135], 44), ("กข", [60, 90, 90, 98], 70),
        (L3, [0, 300, 2000, 340], 94), (MISSED, [100, 300, 800, 340], 87)], size=(2200, 1000)))
    f = tmp_path / "review.html"
    f.write_text(prep.build_html([prep.attach_images(d, [png(2200, 1000)])]), encoding="utf-8")
    r = subprocess.run([node, str(ROOT / "tests" / "ocr_review_smoke.js"), str(f)], capture_output=True, timeout=60)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    out = json.loads(r.stdout.decode("utf-8"))
    n = d["counts"]["to_judge"]
    assert d["counts"]["fragment"] == 1 and d["counts"]["duplicate"] == 1 and d["counts"]["maybe_duplicate"] == 1
    assert out["countInitial"] == f"ทำแล้ว 0 / {n} รายการ"  # fragment/duplicate ไม่นับ
    assert out["wideWarnDisplay"] == "block" and out["wideImgWidth"] == "2400px"
    assert out["narrowImgWidth"] == "600px" and out["narrowWarnDisplay"] == "none"
    assert "pagewrap" in out["pageNear"] and 'class="hl"' in out["pageNear"]
    assert "ถูกกรองออก" in out["fragCard"] and 'id="bAbsent"' not in out["fragCard"] and out["fragCountUnchanged"]
    assert "ถูกกรองออก" in out["dupCard"] and "ทับกับรายการ" in out["dupCard"] and L1 in out["dupCard"]
    assert "อาจซ้ำกับ" in out["maybeCard"] and 'id="bAbsent"' in out["maybeCard"] and L3 in out["maybeCard"]  # ยังตัดสินได้ + ข้อความของรายการที่ทับ
    assert out["defaultHasFiltered"] is False


def test_template_contains_wide_crop_warning_text():
    t = (ROOT / "scripts" / "ocr_review_template.html").read_text(encoding="utf-8")
    assert "ภาพอาจขาดด้านข้าง — พิมพ์ให้ครบบรรทัดตามเอกสาร" in t


def _frag_kind(prep, text, conf, height=11):
    d = prep.build_items("f.PDF", 1, ty(L1), te_boxes([(L1, [50, 300, 900, 330], 90), (text, [60, 100, 400, 100 + height], conf)]))
    return by_text(d, text)["kind"]


def test_small_confident_real_text_is_not_a_fragment(prep):
    """ตัวหนังสือเล็กจริงในอินโฟกราฟิก (สูง < 20 px แต่ conf สูงและยาวพอ) ต้องให้คนตัดสิน ไม่ถูกกรองออก"""
    assert prep.FRAGMENT_MAX_CONF == 70.0 and prep.FRAGMENT_MIN_TEXT_CHARS == 4
    assert _frag_kind(prep, "Hydrogen & Carbon Business", 94.9) == "tesseract_only"
    assert _frag_kind(prep, "& OPPORTUNITY", 94.7) == "tesseract_only"
    assert _frag_kind(prep, "TION", 70.9) == "tesseract_only"  # 4 ตัว conf >= 70


def test_fragment_needs_low_conf_or_very_short_text(prep):
    assert _frag_kind(prep, "aw ๑ ประกอบรายการ", 47.0) == "fragment"  # conf ต่ำ
    assert _frag_kind(prep, "ประกอบรายการ", 69.9) == "fragment" and _frag_kind(prep, "ประกอบรายการ", 70.0) == "tesseract_only"  # ขอบ conf
    assert _frag_kind(prep, "ข่", 95.0) == "fragment"  # สั้นกว่า 4 ตัวแม้ conf สูง
    assert _frag_kind(prep, "ABC", 95.0) == "fragment" and _frag_kind(prep, "ABCD", 95.0) == "tesseract_only"  # ขอบจำนวนตัว
    assert _frag_kind(prep, "A - B . C ,", 95.0) == "fragment"  # เครื่องหมาย/ช่องว่างไม่นับเป็นตัว
    assert _frag_kind(prep, "ประกอบรายการ", 95.0, height=20) == "tesseract_only"  # สูงถึงเกณฑ์ = ไม่ใช่เศษ


def test_returned_small_line_overlapping_one_way_gets_maybe_duplicate_label(prep):
    wide = [0, 100, 2000, 140]
    d = prep.build_items("f.PDF", 1, ty(L1), te_boxes([(L1, wide, 90), ("Hydrogen & Carbon Business", [100, 110, 400, 124], 94.9)], size=(2200, 1000)))
    it = by_text(d, "Hydrogen & Carbon Business")
    assert it["kind"] == "tesseract_only" and it["maybe_duplicate_of"] == [by_text(d, L1)["id"]]


# ---------- ขอบช่วง Typhoon: snap / รวมช่องว่างสั้น / invariant (T06b ช่วงที่ 1 แก้ A) ----------

CONTRACT = "คู่สัญญาฝ่ายใดฝ่ายหนึ่งมีความประสงค์ที่จะขอเลิกสัญญา"  # "คู่ส" + "ัญญา…"


def ty_items(d, kind):
    return [it for it in d["items"] if it["kind"] == kind]


def test_start_mid_syllable_is_snapped_back_to_word_start_and_becomes_diff(prep):
    """Tesseract ขึ้นต้นบรรทัดที่ "ัญญา…" (ไม่มี "คู่ส") — ช่วง Typhoon ต้องขยายเป็น "คู่สัญญา…" และเป็น diff ไม่ใช่ agree"""
    d = prep.build_items("f.PDF", 1, ty("ข้อตกลงนี้ " + CONTRACT), te([CONTRACT[4:]]))
    (it,) = ty_items(d, "diff")
    assert it["typhoon"]["text"] == CONTRACT and it["tesseract"]["text"] == CONTRACT[4:]
    assert it["snap"] == 4 and it["cut_mid"] is False and d["counts"]["agree"] + d["counts"]["agree_audit"] == 0
    assert d["span_stats"]["agree_to_diff"][0]["typhoon_before"] == CONTRACT[4:] and d["span_stats"]["agree_to_diff"][0]["typhoon_after"] == CONTRACT


def test_snap_never_exceeds_snap_max_and_keeps_thai_cluster_together(prep):
    assert prep.SNAP_MAX == 10
    base = "ก" * 30 + "ติดต่อสอบถามข้อมูลเพิ่มเติมได้ที่หน่วยงาน"
    d = prep.build_items("f.PDF", 1, ty(base), te([base[31:]]))  # Tesseract เริ่มที่ "ิดต่อ…" (วรรณยุกต์/สระบนไม่มีพยัญชนะ)
    (it,) = [i for i in d["items"] if i["tesseract"]]
    assert it["typhoon"]["text"].startswith("ติดต่อ")  # ดึง "ต" เข้ามา ไม่ขึ้นต้นด้วยสระลอย
    assert it["cut_mid"] is True and it["cut_mid_sides"] == ["start"] and it["snap"] == 1
    assert d["span_stats"]["cut_mid"] == [it["id"]]


def test_snap_stops_at_the_neighbouring_item(prep):
    a, b = CONTRACT[:20], CONTRACT[20:]  # แบ่งกลางคำโดยไม่มีช่องว่าง — บรรทัดตามภาพห่อกลางคำ
    d = prep.build_items("f.PDF", 1, ty(CONTRACT), te([a, b]))
    assert [it["kind"] for it in d["items"]] == ["agree", "agree"] and d["invariant"]["dropped"] == 0
    assert [it["snap"] for it in d["items"]] == [0, 0] and not any(it["cut_mid"] for it in d["items"])  # ชนกันตรงๆ ไม่ใช่การตัดกลางคำที่ตกหล่น


def test_short_gap_between_items_is_merged_into_the_previous_item_on_same_line(prep):
    d = prep.build_items("f.PDF", 1, ty(f"{L1} xy {L3}"), te([L1, L3]))
    assert d["counts"]["typhoon_only"] == 0 and d["invariant"]["dropped"] == 0
    first, second = d["items"][0], d["items"][1]
    assert first["kind"] == "diff" and first["typhoon"]["text"] == L1 + " xy" and first["gap_merged"] == 2
    assert second["kind"] == "agree" and second["typhoon"]["text"] == L3
    assert d["span_stats"]["gap_merged_runs"] == 1 and d["span_stats"]["gap_merged_chars"] == 2


def test_short_whole_line_gap_goes_to_previous_item_else_next(prep):
    d = prep.build_items("f.PDF", 1, ty(NL.join([L1, "2", L3])), te([L1, L3]))
    assert d["items"][0]["typhoon"]["text"] == L1 + " 2" and d["items"][0]["kind"] == "diff" and d["items"][1]["kind"] == "agree"
    d = prep.build_items("f.PDF", 1, ty(NL.join(["2", L1])), te([L1]))  # ไม่มีรายการก่อนหน้า -> ถัดไป
    (it,) = [i for i in d["items"] if i["tesseract"]]
    assert it["typhoon"]["text"] == "2 " + L1 and it["kind"] == "diff" and d["invariant"]["dropped"] == 0


def test_short_gap_prefers_neighbour_on_the_same_typhoon_line(prep):
    # บรรทัด Typhoon ที่ 2 คือ "2 L3": "2" ต้องไปกับ L3 (บรรทัดเดียวกัน) ไม่ใช่ L1 (บรรทัดก่อนหน้า)
    d = prep.build_items("f.PDF", 1, ty(NL.join([L1, "2 " + L3])), te([L1, L3]))
    first, second = d["items"][0], d["items"][1]
    assert first["kind"] == "agree" and second["typhoon"]["text"] == "2 " + L3 and second["kind"] == "diff"


def test_short_gap_with_no_neighbouring_item_becomes_typhoon_only(prep):
    d = prep.build_items("f.PDF", 1, ty("xy"), te([]))
    (it,) = d["items"]
    assert it["kind"] == "typhoon_only" and it["typhoon"]["text"] == "xy" and d["invariant"]["dropped"] == 0
    assert d["span_stats"]["short_gap_typhoon_only"] == 1


def test_invariant_every_typhoon_char_in_exactly_one_item(prep):
    import random

    rng = random.Random(7)
    words = [L1, L2, L3, CONTRACT, "หมายเหตุประกอบงบการเงิน", "กำไรสุทธิต่อหุ้น", "2. การบริหารจัดการความเสี่ยง", "ไม่ปรากฏ"]
    for _ in range(40):
        lines = rng.sample(words, rng.randint(2, 6))
        text = NL.join(rng.choice([l, l + " " + rng.choice(words)]) for l in lines)
        flat = text.replace(NL, " ")
        slices = []
        for _ in range(rng.randint(1, 9)):  # บรรทัด Tesseract = ช่วงสุ่ม (ตัดกลางคำ/ซ้อนกัน/ซ้ำ) + ตัวอักษรเพี้ยนบ้าง
            a = rng.randint(0, max(len(flat) - 15, 0))
            seg = flat[a:a + rng.randint(8, 70)].replace(" ", "")
            if seg and rng.random() < 0.4:
                i = rng.randrange(len(seg))
                seg = seg[:i] + "x" + seg[i + 1:]
            slices.append(seg)
        d = prep.build_items("f.PDF", 1, ty(text), te([s for s in slices if len(s) >= 2]))
        assert d["invariant"]["dropped"] == 0 and d["invariant"]["doubled"] == 0
        covered = sum(it["pos_end"] - it["pos"] for it in d["items"] if it["kind"] in ("agree", "diff", "typhoon_only"))
        assert covered == d["invariant"]["hay_chars"]


def test_overlapping_matches_are_resolved_best_first(prep):
    """บรรทัด Tesseract สองบรรทัดคล้ายข้อความ Typhoon ช่วงเดียวกัน (อ่านซ้ำ) — ต้องไม่ครอบซ้ำ"""
    d = prep.build_items("f.PDF", 1, ty(L1), te([L1, L1.replace("ธนาคาร", "ธนากร")]))
    assert sorted(it["kind"] for it in d["items"]) == ["agree", "tesseract_only"] and d["invariant"]["doubled"] == 0
    assert d["span_stats"]["demoted_by_overlap"] == 1


def test_repeated_phrase_is_relocated_to_its_second_occurrence(prep):
    d = prep.build_items("f.PDF", 1, ty(NL.join([L1, L2, L1])), te([L1, L2, L1]))
    assert [it["kind"] for it in d["items"]] == ["agree"] * 3 and d["span_stats"]["relocated"] == 1
    assert [it["pos"] for it in d["items"]] == sorted(it["pos"] for it in d["items"])


def test_pos_end_is_saved_for_every_item(prep):
    invented = "ข้อมูลเพิ่มเติมเกณฑ์การประเมินความเสี่ยงและตัวชี้วัดความเสี่ยง"
    missed = "หมายเหตุประกอบงบการเงินรวมข้อสี่จุดสองจุดสอง"
    d = prep.build_items("f.PDF", 1, ty(NL.join([L1, invented, L3])), te([L1, missed, L3]))
    by = {it["kind"]: it for it in d["items"]}
    assert by["typhoon_only"]["pos_end"] - by["typhoon_only"]["pos"] == len(prep.canon(invented))
    assert by["tesseract_only"]["pos_end"] == by["tesseract_only"]["pos"]
    agree = [it for it in d["items"] if it["kind"] == "agree"]
    assert [it["pos_end"] - it["pos"] for it in agree] == [len(prep.canon(L1)), len(prep.canon(L3))]


# ---------- แยกคอลัมน์ (แถบข้าง) ถึงรายการและภาพ debug ----------

def test_column_of_each_tesseract_line_and_page_level_split_recorded(prep):
    side = {"size": [1000, 500], "column_split": {"x": 300, "method": "side-band-v1", "crossing_lines_before": 6},
            "lines": [{"text": L1, "bbox": [20, 50, 280, 80], "conf": 90, "block": 1, "par": 1, "col": 0},
                      {"text": L3, "bbox": [320, 50, 900, 80], "conf": 90, "block": 1001, "par": 1, "col": 1}]}
    res = OcrResult(text="", backend="tesseract-local", model="t", revision="v", seconds=1.0, extra={"parts": [side, {"size": [100, 100], "lines": []}]})
    d = prep.build_items("f.PDF", 1, ty(NL.join([L1, L3])), res)
    assert {it["tesseract"]["text"]: it["tesseract"]["column"] for it in d["items"]} == {L1: 0, L3: 1}
    assert d["column_split"] == [300, None]


def test_column_debug_images_have_one_file_per_region(prep, tmp_path):
    from PIL import Image

    side = {"size": [1000, 500], "column_split": {"x": 300, "method": "side-band-v1", "crossing_lines_before": 6},
            "lines": [{"text": L1, "bbox": [20, 50, 280, 80], "conf": 90, "block": 1, "par": 1, "col": 0},
                      {"text": L3, "bbox": [320, 50, 900, 80], "conf": 90, "block": 1001, "par": 1, "col": 1}]}
    res = OcrResult(text="", backend="tesseract-local", model="t", revision="v", seconds=1.0, extra={"parts": [side]})
    paths = prep.save_column_debug("ONEREPORTKBANKT.PDF", 87, res, [png(1000, 500)], out_dir=tmp_path)
    assert [p.name for p in paths] == ["kbank87_cols_img0_side.png", "kbank87_cols_img0_body.png"]
    assert Image.open(paths[0]).size == (300, 500) and Image.open(paths[1]).size == (700, 500)
    plain = OcrResult(text="", backend="tesseract-local", model="t", revision="v", seconds=1.0, extra={"size": [10, 10], "lines": []})
    assert prep.save_column_debug("x.PDF", 1, plain, [png(10, 10)], out_dir=tmp_path) == []
