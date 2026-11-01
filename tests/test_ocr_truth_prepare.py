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
