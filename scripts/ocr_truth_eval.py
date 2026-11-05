"""วัดความแม่น OCR เทียบข้อความที่ถูกที่คนตัดสินแล้ว (T06b ช่วงที่ 2)

    python scripts/ocr_truth_eval.py

อ่านเฉพาะ eval/datasets/ocr_truth/judgments.json + *.items.json (ผลจากแคช OCR ผ่าน ocr_truth_prepare.py แล้ว) — ไม่เรียกโมเดล ไม่รัน OCR
เขียนผลที่ eval/results/ocr_truth_eval.json + .md; ถ้าคู่ dedup ที่เจอไม่ตรงกับที่คาดไว้ (``EXPECTED_DEDUP``) พิมพ์ส่วนต่างแล้วออกด้วย 2 โดยไม่เขียนผล

กติกา (จาก task spec; ดูเหตุผลเรื่องการตัดสินใจเสริมใน docstring ของแต่ละฟังก์ชัน)
1. ``canon`` (eval/ocr_canon.py ใช้ร่วมกับ ocr_truth_prepare.py) = normalizer กลาง -> ํา เป็น ำ -> ตัวห้อย/ตัวยก เป็นเลขปกติ -> คำพูดโค้งเป็นตรง -> ตัด bullet นำหน้า -> ตัด markdown -> ตัดช่องว่างทั้งหมด
2. truth ต่อหน้า = ข้อความของรายการ agree/diff/typhoon_only/tesseract_only เรียงตาม (pos, order) ต่อกัน
   - agree ที่ไม่ถูกคนตัดสิน = ข้อความ Typhoon; รายการที่คนตัดสิน = ``judgments[id].text`` (absent = ว่าง)
   - แยก (ก) เนื้อความ / (ข) แผนผัง (``DIAGRAM_RANGES``); ตัดหมวดนอก scope ออก (``EXCLUDED_ITEMS``) และนับแยก
   - หมวด ``table_header_restructured`` (``TABLE_HEADER_ITEMS`` = KBANK 87 _142–_145): ตัดออกจาก truth/hypothesis/อ่านตก/error ระดับคำ วัดแยกด้วย char 3-gram ไม่สนลำดับ
3. dedup typhoon_only x tesseract_only (image เดียวกัน, แถบ y ซ้อน, canon เท่ากันหรือครอบ) -> นับครั้งเดียว เก็บอันที่ยาวกว่า (เท่ากัน = เก็บ typhoon_only)
4. ตัววัด: CER (edit distance บน canon), บรรทัดแต่ง/อ่านตก, คะแนนแผนผัง (char 3-gram ไม่สนลำดับ), อัตราผิดของ agree ที่สุ่มตรวจ, เลขหน้า, รายการ error ระดับคำ

การตัดสินใจที่ spec ไม่ได้ระบุ (ทำตามนี้ ถ้าไม่เห็นด้วยแก้ที่นี่)
- ลำดับของ hypothesis: ทั้ง Typhoon และ Tesseract เรียงตาม (pos, order) เดียวกับ truth -> CER วัดความถูกของตัวอักษร ไม่ลงโทษลำดับอ่านข้ามคอลัมน์
- หมวดนอก scope (KBANK 87 _184) และ table_header_restructured ถูกตัดออกจาก hypothesis ของ Typhoon ด้วย (ไม่งั้นกลายเป็นตัวอักษรเกินที่ truth ไม่มี); ค่า CER ถ้าไม่ตัดออกเก็บใน json (``cer_keep_excluded``)
- Tesseract hypothesis = รายการ agree/diff/tesseract_only ที่ไม่ใช่ (ข); ไม่รวม fragment/duplicate และบรรทัดที่ถูกข้ามเป็นขยะ (``skipped_tesseract``)
- tesseract_only ที่อยู่ในคู่ dedup (ไม่ว่าเก็บหรือทิ้ง) ไม่นับเป็น "Typhoon อ่านตก" เพราะ Typhoon อ่านข้อความนั้นได้ (เป็น typhoon_only)
- typhoon_only ที่ไม่มี region (ไม่ทราบตำแหน่ง) ถือว่าแถบ y ซ้อนกับทุกบรรทัดในหน้า; เงื่อนไข canon เท่ากัน/ครอบเป็นตัวกัน
"""

import argparse
import glob
import json
import math
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.ocr_canon import canon  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRUTH_DIR = ROOT / "eval" / "datasets" / "ocr_truth"
RESULTS_DIR = ROOT / "eval" / "results"

KPI_ACCURACY = 0.90  # 1 − CER
NGRAM = 3
SHORT_NAMES = {"E_ONE_REPORT_CPALLT": "CPALL", "ONEREPORTKBANKT": "KBANK", "ONEREPORTPTTT": "PTT"}

# หมวดที่ตัดออกจาก truth ทั้งสองชุดและนับแยก
EXCLUDED_ITEMS = {
    "ONEREPORTKBANKT_p87_184": "handwriting",  # ลายมือ/ลายเซ็น อยู่นอก scope
}
# หมวด table_header_restructured: หัวตารางครึ่งขวาที่ Typhoon จัดวางใหม่ (รวมบรรทัดข้ามช่อง) — ตัดออกจาก truth/hypothesis ของ CER, ไม่นับเป็นอ่านตก/
# แต่งขึ้น/error ระดับคำ และวัดแยกแบบไม่สนลำดับ (char 3-gram) เหมือนแผนผัง
TABLE_HEADER_CATEGORY = "table_header_restructured"
TABLE_HEADER_ITEMS = {f"ONEREPORTKBANKT_p87_{n}": TABLE_HEADER_CATEGORY for n in (142, 143, 144, 145)}
# ชุด (ข) แผนผัง = รายการ tesseract_only ในช่วงเลขลำดับนี้ (รวมปลายทั้งสอง)
DIAGRAM_RANGES = {
    "ONEREPORTPTTT_p43": [(46, 96)],
    "ONEREPORTPTTT_p293": [(4, 10), (15, 27)],
}
# คู่ dedup ที่คาดว่าจะเจอ (typhoon_only, tesseract_only)
EXPECTED_DEDUP = {
    ("E_ONE_REPORT_CPALLT_p442_003", "E_ONE_REPORT_CPALLT_p442_002"),
    ("ONEREPORTKBANKT_p81_044", "ONEREPORTKBANKT_p81_043"),
    ("ONEREPORTKBANKT_p87_056", "ONEREPORTKBANKT_p87_055"),
    ("ONEREPORTKBANKT_p87_141", "ONEREPORTKBANKT_p87_140"),
    ("ONEREPORTPTTT_p43_102", "ONEREPORTPTTT_p43_101"),
    ("ONEREPORTPTTT_p293_001", "ONEREPORTPTTT_p293_003"),
    ("ONEREPORTPTTT_p293_012", "ONEREPORTPTTT_p293_003"),
}
# เลขหน้าจริงที่ Typhoon ควรอ่านได้ (ลำดับตามที่พิมพ์บนหน้า)
EXPECTED_PAGE_NUMBERS = {
    ("E_ONE_REPORT_CPALLT", 442): [436], ("E_ONE_REPORT_CPALLT", 462): [456], ("ONEREPORTKBANKT", 2): [],
    ("ONEREPORTKBANKT", 81): [158, 159], ("ONEREPORTKBANKT", 87): [170, 171],
    ("ONEREPORTPTTT", 293): [291], ("ONEREPORTPTTT", 43): [41],
}
# error ระดับคำที่ต้องเห็นในรายการ (คำที่ Typhoon เขียน) — ไม่เห็น = หยุดถาม
REQUIRED_ERRORS = [("ONEREPORTKBANKT_p81_077", "ระบุดับ"), ("ONEREPORTPTTT_p43_013", "วางแผนทาง")]


# ---------- ข้อมูล ----------

def item_number(item_id: str) -> int:
    return int(item_id.rsplit("_", 1)[1])


def page_stem(page: dict) -> str:
    return f"{Path(page['file']).stem}_p{page['page']}"


def page_key(page: dict) -> str:
    return f"{SHORT_NAMES.get(Path(page['file']).stem, Path(page['file']).stem)} {page['page']}"


def sort_key(it: dict) -> tuple:
    return it["pos"], it["order"]


def judged_text(it: dict, judgments: dict) -> str:
    """ข้อความที่ถูกของรายการ: คำตัดสิน (absent = ว่าง) ถ้ามี; agree ที่รับอัตโนมัติและไม่ถูกตรวจ = ข้อความ Typhoon; อย่างอื่นที่ไม่มีคำตัดสิน = error"""
    j = judgments.get(it["id"])
    if j is not None:
        return "" if j["verdict"] == "absent" else j["text"]
    if it["kind"] == "agree" and it["auto_accept"]:
        return it["typhoon"]["text"]
    raise KeyError(f"{it['id']} ({it['kind']}): ไม่มีคำตัดสิน")


def diagram_ids(page: dict, ranges: dict) -> set[str]:
    spans = ranges.get(page_stem(page), [])
    return {it["id"] for it in page["items"]
            if it["kind"] == "tesseract_only" and any(lo <= item_number(it["id"]) <= hi for lo, hi in spans)}


# ---------- dedup ----------

def _y_overlap(tess: dict, region: dict | None) -> bool:
    """tesseract_only (bbox) กับแถบของ typhoon_only (region: image + bbox) อยู่ภาพเดียวกันและช่วง y ซ้อนกัน; ไม่มี region = ไม่ทราบตำแหน่ง ถือว่าซ้อน"""
    if region is None:
        return True
    if tess["tesseract"]["image"] != region["image"]:
        return False
    (_, y0, _, y1), (_, a, _, b) = tess["tesseract"]["bbox"], region["bbox"]
    return min(y1, b) - max(y0, a) > 0


def find_dedup_pairs(items: list[dict], judgments: dict) -> list[dict]:
    """typhoon_only x tesseract_only ที่อยู่ image เดียวกัน แถบ y ซ้อนกัน และ canon ของข้อความที่ตัดสินแล้วเท่ากัน/อันหนึ่งอยู่ในอีกอัน
    (ข้อความว่างข้าม) -> ``dropped`` = อันที่สั้นกว่า; เท่ากัน = ทิ้ง tesseract_only (typhoon_only มี pos จริงของข้อความ Typhoon)"""
    typhoon_only = [it for it in items if it["kind"] == "typhoon_only"]
    tess_only = [it for it in items if it["kind"] == "tesseract_only"]
    pairs = []
    for t in typhoon_only:
        ct = canon(judged_text(t, judgments))
        if not ct:
            continue
        for s in tess_only:
            cs = canon(judged_text(s, judgments))
            if not cs or not _y_overlap(s, t["region"]):
                continue
            if ct == cs:
                relation, dropped = "equal", s["id"]
            elif cs in ct:
                relation, dropped = "typhoon_only_contains", s["id"]
            elif ct in cs:
                relation, dropped = "tesseract_only_contains", t["id"]
            else:
                continue
            pairs.append({"typhoon_only": t["id"], "tesseract_only": s["id"], "relation": relation,
                          "kept": t["id"] if dropped == s["id"] else s["id"], "dropped": dropped,
                          "typhoon_text": judged_text(t, judgments), "tesseract_text": judged_text(s, judgments)})
    return pairs


# ---------- ตัววัด ----------

def cer_stats(truth: str, hyp: str) -> dict:
    from rapidfuzz.distance import Levenshtein

    dist = Levenshtein.distance(truth, hyp)
    cer = dist / len(truth) if truth else (0.0 if not hyp else 1.0)
    return {"ref_len": len(truth), "hyp_len": len(hyp), "distance": dist, "cer": cer, "accuracy": 1 - cer, "pass": 1 - cer >= KPI_ACCURACY}


def ngrams(text: str, n: int = NGRAM) -> Counter:
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def ngram_scores(truth_lines: list[str], hyp_lines: list[str], n: int = NGRAM) -> dict:
    """recall/precision/F1 ของ char n-gram แบบ multiset ไม่สนลำดับ (canon ต่อบรรทัด n-gram ไม่ข้ามบรรทัด)"""
    ref, hyp = Counter(), Counter()
    for ln in truth_lines:
        ref.update(ngrams(canon(ln), n))
    for ln in hyp_lines:
        hyp.update(ngrams(canon(ln), n))
    overlap = sum((ref & hyp).values())
    recall = overlap / sum(ref.values()) if ref else 0.0
    precision = overlap / sum(hyp.values()) if hyp else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {"ref_ngrams": sum(ref.values()), "hyp_ngrams": sum(hyp.values()), "overlap": overlap,
            "recall": recall, "precision": precision, "f1": f1}


def upper_bound_binomial(k: int, n: int, conf: float = 0.95) -> float | None:
    """ขอบบนแบบ Clopper–Pearson ด้านเดียวของอัตราผิด เมื่อพบ k ผิดจาก n"""
    if n == 0:
        return None
    if k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        cdf = sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1))
        lo, hi = (mid, hi) if cdf > 1 - conf else (lo, mid)
    return hi


def word_tokens(c: str) -> list[str]:
    from pythainlp.tokenize import word_tokenize

    return [w for w in word_tokenize(c, engine="newmm", keep_whitespace=False) if w.strip()]


def word_errors(ids: list[str], truth: str, typhoon: str) -> list[dict]:
    """ความต่างระดับคำ (newmm บน canon) ระหว่างข้อความที่ถูกกับที่ Typhoon เขียน; ``typhoon_context`` / ``truth_context`` = ส่วนที่ต่างพร้อมคำข้างเคียงข้างละ 1 คำ
    ``ids`` = รายการที่ถูกรวมเป็นช่วงเดียวกัน (ดู ``typhoon_errors``)"""
    a, b = word_tokens(canon(truth)), word_tokens(canon(typhoon))
    out = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        out.append({"id": ids[0], "ids": ids, "op": op, "truth": "".join(a[i1:i2]), "typhoon": "".join(b[j1:j2]),
                    "truth_context": "".join(a[max(i1 - 1, 0):i2 + 1]), "typhoon_context": "".join(b[max(j1 - 1, 0):j2 + 1])})
    return out


def typhoon_errors(items: list[dict], judgments: dict, excluded: dict) -> list[dict]:
    """error ระดับคำของข้อความ Typhoon (agree/diff/typhoon_only เรียงตาม pos): รายการที่ติดกันและ canon ต่างจากข้อความที่ถูก ถูกรวมเป็นช่วงเดียวก่อน diff
    เพราะขอบช่วงของ Typhoon กับของข้อความที่ถูกอาจเลื่อนข้ามรายการข้างเคียง (เช่น "2568" อยู่รายการหน้า/หลังต่างกัน) — diff ทีละรายการจะเห็นเป็น
    ตัวขาด + ตัวเกินที่ไม่ใช่ความผิดของ Typhoon; ช่วงที่รวมแล้วเท่ากัน = ไม่ใช่ error. typhoon_only ที่แต่งขึ้น (absent) ไม่เข้าที่นี่ (รายงานใน fabricated)"""
    errors, run = [], []

    def flush():
        if run:
            truth = " ".join(t for _, t, _ in run)
            typhoon = " ".join(y for _, _, y in run)
            if canon(truth) != canon(typhoon):
                errors.extend(word_errors([i for i, _, _ in run], truth, typhoon))
            run.clear()

    for it in items:
        if not it["typhoon"] or it["id"] in excluded:
            continue
        truth_text = judged_text(it, judgments) if (it["id"] in judgments or it["kind"] != "agree") else it["typhoon"]["text"]
        if canon(truth_text) == canon(it["typhoon"]["text"]) or (it["kind"] == "typhoon_only" and not canon(truth_text)):
            flush()
        else:
            run.append((it["id"], truth_text, it["typhoon"]["text"]))
    flush()
    return errors


# ---------- ต่อหน้า ----------

def evaluate_page(page: dict, judgments: dict, excluded: dict = EXCLUDED_ITEMS, diagram_ranges: dict = DIAGRAM_RANGES,
                  table_header: dict = TABLE_HEADER_ITEMS) -> dict:
    items = sorted(page["items"], key=sort_key)
    ids = {it["id"]: it for it in items}
    scored_kinds = ("agree", "diff", "typhoon_only", "tesseract_only")
    header_ids = {i for i in table_header if i in ids}
    out_of_scope = {**excluded, **{i: table_header[i] for i in header_ids}}  # ไม่เข้า truth / hypothesis / อ่านตก / error ระดับคำ
    candidates = [it for it in items if it["kind"] in scored_kinds and it["id"] not in out_of_scope]
    diagram = diagram_ids(page, diagram_ranges)
    pairs = find_dedup_pairs(candidates, judgments)
    dropped = {p["dropped"] for p in pairs}
    in_pair = {p["typhoon_only"] for p in pairs} | {p["tesseract_only"] for p in pairs}

    truth_items = [it for it in candidates if it["id"] not in diagram and it["id"] not in dropped]
    truth_lines = [judged_text(it, judgments) for it in truth_items]
    truth = "".join(canon(t) for t in truth_lines)
    typhoon_hyp = "".join(canon(it["typhoon"]["text"]) for it in items if it["typhoon"] and it["id"] not in out_of_scope)
    typhoon_hyp_keep = "".join(canon(it["typhoon"]["text"]) for it in items if it["typhoon"])
    tess_items = [it for it in items if it["kind"] in ("agree", "diff", "tesseract_only") and it["id"] not in diagram
                  and it["id"] not in header_ids]
    tess_hyp = "".join(canon(it["tesseract"]["text"]) for it in tess_items)

    fabricated = [{"id": it["id"], "text": it["typhoon"]["text"]} for it in candidates
                  if it["kind"] == "typhoon_only" and not canon(judged_text(it, judgments))]
    missed = [{"id": it["id"], "text": judged_text(it, judgments), "tesseract": it["tesseract"]["text"]} for it in candidates
              if it["kind"] == "tesseract_only" and it["id"] not in diagram and it["id"] not in in_pair
              and canon(judged_text(it, judgments))]

    diagram_result = None
    if diagram:
        d_items = [it for it in items if it["id"] in diagram]
        d_truth = [judged_text(it, judgments) for it in d_items if canon(judged_text(it, judgments))]
        diagram_result = {
            "ids": sorted(diagram), "truth_lines": len(d_truth), "figures": len(page["figures"]),
            "typhoon": ngram_scores(d_truth, page["figures"]),
            "tesseract": ngram_scores(d_truth, [it["tesseract"]["text"] for it in d_items]),
        }

    header_result = None
    if header_ids:
        h_items = [it for it in items if it["id"] in header_ids]
        h_truth = [judged_text(it, judgments) for it in h_items if canon(judged_text(it, judgments))]
        header_result = {
            "category": TABLE_HEADER_CATEGORY, "ids": sorted(header_ids), "truth_lines": len(h_truth),
            "items": [{"id": it["id"], "kind": it["kind"], "typhoon": it["typhoon"] and it["typhoon"]["text"],
                       "tesseract": it["tesseract"] and it["tesseract"]["text"], "judged": judged_text(it, judgments)} for it in h_items],
            "typhoon": ngram_scores(h_truth, [it["typhoon"]["text"] for it in h_items if it["typhoon"]]),
            "tesseract": ngram_scores(h_truth, [it["tesseract"]["text"] for it in h_items if it["tesseract"]]),
        }

    audited = [it for it in items if it["kind"] == "agree" and it["audit"]]
    sample = [it for it in audited if it["audit_reason"] == "sample"]
    cut_mid = [it for it in audited if it["audit_reason"] == "cut_mid"]

    def changed(group):
        return [{"id": it["id"], "typhoon": it["typhoon"]["text"], "judged": judgments[it["id"]]["text"], "verdict": judgments[it["id"]]["verdict"]}
                for it in group if canon(judgments[it["id"]]["text"]) != canon(it["typhoon"]["text"])]

    errors = typhoon_errors(items, judgments, out_of_scope)

    return {
        "key": page_key(page), "file": page["file"], "page": page["page"],
        "truth_items": len(truth_items), "truth_chars": len(truth),
        "excluded": {i: {"category": excluded[i], "text": (ids[i]["typhoon"] or ids[i]["tesseract"])["text"]} for i in excluded if i in ids},
        "diagram_ids": sorted(diagram), "table_header": header_result,
        "dedup_pairs": pairs,
        "cer": {"typhoon": cer_stats(truth, typhoon_hyp), "tesseract": cer_stats(truth, tess_hyp),
                "typhoon_keep_excluded": cer_stats(truth, typhoon_hyp_keep)},
        "fabricated": fabricated, "missed": missed, "diagram": diagram_result,
        "agree": {"auto_accepted": sum(1 for it in items if it["kind"] == "agree" and it["auto_accept"]),
                  "sampled": len(sample), "sample_changed": changed(sample), "cut_mid": len(cut_mid), "cut_mid_changed": changed(cut_mid)},
        "page_numbers": {"typhoon": page["typhoon_page_numbers"],
                         "expected": EXPECTED_PAGE_NUMBERS.get((Path(page["file"]).stem, page["page"])),
                         "ok": page["typhoon_page_numbers"] == EXPECTED_PAGE_NUMBERS.get((Path(page["file"]).stem, page["page"]))},
        "errors": errors,
        "counts": {k: sum(1 for it in items if it["kind"] == k) for k in ("agree", "diff", "typhoon_only", "tesseract_only", "fragment", "duplicate")},
    }


def aggregate(results: list[dict]) -> dict:
    out = {}
    for who in ("typhoon", "tesseract", "typhoon_keep_excluded"):
        out[who] = cer_stats_from_sums(sum(r["cer"][who]["ref_len"] for r in results), sum(r["cer"][who]["hyp_len"] for r in results),
                                       sum(r["cer"][who]["distance"] for r in results))
    sampled = sum(r["agree"]["sampled"] for r in results)
    changed = sum(len(r["agree"]["sample_changed"]) for r in results)
    unchecked = sum(r["agree"]["auto_accepted"] - r["agree"]["sampled"] for r in results)
    rate, upper = (changed / sampled if sampled else None), upper_bound_binomial(changed, sampled)
    out["agree"] = {"sampled": sampled, "changed": changed, "error_rate": rate, "error_rate_upper95": upper,
                    "unchecked_auto_accepted": unchecked,
                    "expected_errors_unchecked": rate * unchecked if rate is not None else None,
                    "upper95_errors_unchecked": upper * unchecked if upper is not None else None}
    return out


def cer_stats_from_sums(ref_len: int, hyp_len: int, distance: int) -> dict:
    cer = distance / ref_len if ref_len else 0.0
    return {"ref_len": ref_len, "hyp_len": hyp_len, "distance": distance, "cer": cer, "accuracy": 1 - cer, "pass": 1 - cer >= KPI_ACCURACY}


def load_pages(truth_dir: Path = TRUTH_DIR) -> list[dict]:
    pages = [json.loads(Path(f).read_text(encoding="utf-8")) for f in glob.glob(str(truth_dir / "*.items.json"))]
    return sorted(pages, key=lambda p: (Path(p["file"]).stem, p["page"]))


def check_dedup(results: list[dict]) -> tuple[set, set]:
    found = {(p["typhoon_only"], p["tesseract_only"]) for r in results for p in r["dedup_pairs"]}
    return EXPECTED_DEDUP - found, found - EXPECTED_DEDUP  # (ขาด, เกิน)


# ---------- รายงาน ----------

def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def render_md(results: list[dict], agg: dict, gen: dict) -> str:
    L = ["# ผลวัด OCR เทียบข้อความที่ถูก (T06b ช่วงที่ 2)", "",
         f"- truth: judgments.json ({gen['judgments']} คำตัดสิน) + items.json {len(results)} หน้า · KPI: 1−CER ≥ {KPI_ACCURACY:.0%}",
         "- ไม่ได้รัน OCR ใหม่ (อ่าน items.json ที่สร้างจากแคช)", "",
         "## 1. CER ของเนื้อความ (canon, edit distance ระดับตัวอักษร)", "",
         "| หน้า | ตัวอักษร truth | Typhoon dist | Typhoon 1−CER | Tesseract dist | Tesseract 1−CER |", "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        ty, te = r["cer"]["typhoon"], r["cer"]["tesseract"]
        L.append(f"| {r['key']} | {ty['ref_len']} | {ty['distance']} | {pct(ty['accuracy'])} {'✅' if ty['pass'] else '❌'} "
                 f"| {te['distance']} | {pct(te['accuracy'])} {'✅' if te['pass'] else '❌'} |")
    ty, te = agg["typhoon"], agg["tesseract"]
    L += [f"| **รวม** | {ty['ref_len']} | {ty['distance']} | **{pct(ty['accuracy'])}** {'✅' if ty['pass'] else '❌'} "
          f"| {te['distance']} | **{pct(te['accuracy'])}** {'✅' if te['pass'] else '❌'} |", "",
          f"รวม = ผลรวม distance / ผลรวมตัวอักษร truth · ถ้าไม่ตัดหมวดนอก scope ออกจาก hypothesis ของ Typhoon: 1−CER รวม = {pct(agg['typhoon_keep_excluded']['accuracy'])}", "",
          "> truth ของ agree ที่ไม่ถูกสุ่มตรวจ = ข้อความ Typhoon เอง จึงอาจทำให้ Typhoon ดีเกินจริงเท่าที่กลุ่มนั้นมีผิด (ดูหัวข้อ 4)", "",
          "## 2. บรรทัดที่ Typhoon แต่งขึ้น / อ่านตก", "",
          "แต่งขึ้น = typhoon_only ที่คนตัดสิน absent (ไม่นับหมวดนอก scope) · อ่านตก = tesseract_only เนื้อความ มีจริง ไม่อยู่ในคู่ dedup", ""]
    fab = [(r["key"], f) for r in results for f in r["fabricated"]]
    miss = [(r["key"], m) for r in results for m in r["missed"]]
    L.append(f"**แต่งขึ้น {len(fab)} บรรทัด**" + ("" if fab else " (ไม่มี)"))
    L += [f"- {k} `{f['id']}`: {f['text']}" for k, f in fab]
    L += ["", f"**อ่านตก {len(miss)} บรรทัด**", "", "| หน้า | id | ข้อความที่ถูก | Tesseract อ่านเป็น |", "|---|---|---|---|"]
    L += [f"| {k} | {m['id'].rsplit('_', 1)[1]} | {m['text']} | {m['tesseract']} |" for k, m in miss]
    L += ["", "## 3. แผนผัง (char 3-gram ไม่สนลำดับ)", "",
          "| หน้า | บรรทัด (ข) ที่มีจริง | ระบบ | recall | precision | F1 |", "|---|---:|---|---:|---:|---:|"]
    for r in results:
        if r["diagram"]:
            for who in ("typhoon", "tesseract"):
                d = r["diagram"][who]
                L.append(f"| {r['key']} | {r['diagram']['truth_lines']} | {who} | {pct(d['recall'])} | {pct(d['precision'])} | {pct(d['f1'])} |")
    L += ["", "> Typhoon ใช้ข้อความใน figures ของหน้า (เป็นคำอธิบายภาพยาวๆ มีร้อยแก้ว precision จึงต่ำโดยธรรมชาติ) · Tesseract ใช้บรรทัดชุด (ข) · "
          "truth ของ (ข) มาจากบรรทัด Tesseract ที่คนตัดสิน จึงเอนเอียงเข้าหา Tesseract — อ่าน recall ของ Typhoon เป็นหลัก", "",
          "## 3.1 หัวตารางที่ Typhoon จัดใหม่ (table_header_restructured, char 3-gram ไม่สนลำดับ)", "",
          "ตัดออกจาก truth/hypothesis ของหัวข้อ 1 และไม่นับเป็นอ่านตก/error ระดับคำ · วัดเหมือนแผนผัง (n-gram ต่อบรรทัด) · truth มาจากบรรทัด Tesseract ที่คนตัดสิน "
          "จึงเอนเอียงเข้าหา Tesseract — อ่าน recall ของ Typhoon เป็นหลัก", "",
          "| หน้า | id | บรรทัด truth | ระบบ | recall | precision | F1 |", "|---|---|---:|---|---:|---:|---:|"]
    for r in results:
        h = r["table_header"]
        if h:
            for who in ("typhoon", "tesseract"):
                L.append(f"| {r['key']} | {'+'.join(i.rsplit('_', 1)[1] for i in h['ids'])} | {h['truth_lines']} | {who} "
                         f"| {pct(h[who]['recall'])} | {pct(h[who]['precision'])} | {pct(h[who]['f1'])} |")
    for r in results:
        if r["table_header"]:
            L.append("")
            L += [f"- {r['key']} {it['id'].rsplit('_', 1)[1]} [{it['kind']}] ถูก: {it['judged'] or '∅'} · Typhoon: {it['typhoon'] or '∅'} · Tesseract: {it['tesseract'] or '∅'}"
                  for it in r["table_header"]["items"]]
    L += ["", "## 4. ผลสุ่มตรวจ agree", "",
          "| หน้า | agree รับอัตโนมัติ | สุ่มตรวจ | คนแก้ | cut_mid (ตรวจทุกอัน แยกจากกลุ่มสุ่ม) | cut_mid ที่คนแก้ |", "|---|---:|---:|---:|---:|---:|"]
    L += [f"| {r['key']} | {r['agree']['auto_accepted']} | {r['agree']['sampled']} | {len(r['agree']['sample_changed'])} "
          f"| {r['agree']['cut_mid']} | {len(r['agree']['cut_mid_changed'])} |" for r in results]
    a = agg["agree"]
    L += ["", f"- รวมสุ่มตรวจ {a['sampled']} รายการ คนแก้ข้อความ {a['changed']} รายการ → อัตราผิดโดยประมาณ {pct(a['error_rate'])} "
              f"(ขอบบน 95% ด้านเดียว {pct(a['error_rate_upper95'])})",
          f"- กลุ่มที่รับอัตโนมัติและไม่ถูกตรวจ {a['unchecked_auto_accepted']} รายการ → คาดว่าผิด ≈ {a['expected_errors_unchecked']:.1f} (ขอบบน 95% ≈ {a['upper95_errors_unchecked']:.1f}) รายการ"]
    for r in results:
        for c in r["agree"]["sample_changed"] + r["agree"]["cut_mid_changed"]:
            L.append(f"  - แก้: {c['id']}: {c['typhoon']} → {c['judged']}")
    L += ["", "## 5. เลขหน้า", "", "| หน้า | Typhoon อ่านได้ | ค่าจริง | ผล |", "|---|---|---|---|"]
    L += [f"| {r['key']} | {r['page_numbers']['typhoon']} | {r['page_numbers']['expected']} | {'✅' if r['page_numbers']['ok'] else '❌'} |" for r in results]
    L += ["", "## 6. รายการ error ของ Typhoon ในเนื้อความ (ระดับคำ)", "",
          "คำ = newmm บน canon · `ที่ถูก → Typhoon` พร้อมคำข้างเคียงข้างละ 1 คำ", "", "| หน้า | id | op | ที่ถูก | Typhoon | บริบท (Typhoon) |", "|---|---|---|---|---|---|"]
    n_err = 0
    for r in results:
        for e in r["errors"]:
            n_err += 1
            L.append(f"| {r['key']} | {'+'.join(i.rsplit('_', 1)[1] for i in e['ids'])} | {e['op']} | {e['truth'] or '∅'} | {e['typhoon'] or '∅'} | {e['typhoon_context']} |")
    L += ["", f"รวม {n_err} จุด (นับจากรายการที่ Typhoon เขียนและคนแก้; บรรทัดที่อ่านตกอยู่หัวข้อ 2, แต่งขึ้นหัวข้อ 2)", "",
          "## 7. คู่ dedup", "", "| หน้า | typhoon_only | tesseract_only | ความสัมพันธ์ | เก็บ | ทิ้ง |", "|---|---|---|---|---|---|"]
    for r in results:
        for p in r["dedup_pairs"]:
            L.append(f"| {r['key']} | {p['typhoon_only'].rsplit('_', 1)[1]} | {p['tesseract_only'].rsplit('_', 1)[1]} | {p['relation']} "
                     f"| {p['kept'].rsplit('_', 1)[1]} | {p['dropped'].rsplit('_', 1)[1]} |")
    L += ["", "## 8. หมวดที่ตัดออก / ชุด (ข)", ""]
    for r in results:
        for i, e in r["excluded"].items():
            L.append(f"- {r['key']} `{i}` — {e['category']}: {e['text']}")
    for r in results:
        if r["diagram_ids"]:
            L.append(f"- {r['key']} ชุด (ข): tesseract_only {len(r['diagram_ids'])} รายการ ({r['diagram']['truth_lines']} บรรทัดมีจริง): "
                     + ", ".join(i.rsplit('_', 1)[1] for i in r["diagram_ids"]))
    return "\n".join(L) + "\n"


# ---------- main ----------

def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="วัด OCR เทียบข้อความที่ถูก").parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    judgments = json.loads((TRUTH_DIR / "judgments.json").read_text(encoding="utf-8"))["judgments"]
    pages = load_pages()
    results = [evaluate_page(p, judgments) for p in pages]

    print("== ชุด (ข) แผนผัง (tesseract_only) ==")
    for r, p in zip(results, pages):
        by_id = {it["id"]: it for it in p["items"]}
        for i in r["diagram_ids"]:
            print(f"{r['key']} {i.rsplit('_', 1)[1]} [{judgments[i]['verdict']}] {by_id[i]['tesseract']['text']}  =>  {judgments[i]['text']}")
        if r["diagram_ids"]:
            print(f"-- {r['key']}: {len(r['diagram_ids'])} รายการ")
    print("\n== คู่ dedup ที่เจอ ==")
    for r in results:
        for q in r["dedup_pairs"]:
            print(f"{r['key']} typhoon_only {q['typhoon_only'].rsplit('_', 1)[1]} / tesseract_only {q['tesseract_only'].rsplit('_', 1)[1]} "
                  f"[{q['relation']}] เก็บ {q['kept'].rsplit('_', 1)[1]} ทิ้ง {q['dropped'].rsplit('_', 1)[1]}")
    missing, extra = check_dedup(results)
    print(f"คู่ที่คาดแต่ไม่เจอ: {sorted(missing) or 'ไม่มี'}\nคู่ที่เจอเกินคาด: {sorted(extra) or 'ไม่มี'}")
    if missing or extra:
        print("[หยุด] คู่ dedup ไม่ตรงกับที่คาด — ไม่เขียนผล", file=sys.stderr)
        return 2
    texts = {(e["id"]): [] for r in results for e in r["errors"]}
    for r in results:
        for e in r["errors"]:
            texts[e["id"]].append(e["typhoon_context"])
    absent = [(i, w) for i, w in REQUIRED_ERRORS if not any(w in c for c in texts.get(i, []))]
    if absent:
        print(f"[หยุด] ไม่เห็น error ที่ต้องมี: {absent}", file=sys.stderr)
        return 3

    agg = aggregate(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gen = {"judgments": len(judgments)}
    out_json = {"kpi_accuracy": KPI_ACCURACY, "aggregate": agg, "pages": results,
                "dedup_expected_missing": sorted(missing), "dedup_unexpected": sorted(extra)}
    (RESULTS_DIR / "ocr_truth_eval.json").write_text(json.dumps(out_json, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS_DIR / "ocr_truth_eval.md").write_text(render_md(results, agg, gen), encoding="utf-8")
    print(f"\nเขียน {RESULTS_DIR / 'ocr_truth_eval.json'} และ .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
