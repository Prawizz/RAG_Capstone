"""ตัวสะสมผลนับของ normalizer ที่ใช้ร่วมกันของ extractor ชนิด office (docx / pptx)"""

from collections import Counter

from app.thai.normalizer import normalize_with_report


class Tally:
    """ผลนับของ normalizer ที่สะสมราย element -> รวมเป็นราย unit"""

    def __init__(self):
        self.fixes: Counter = Counter()
        self.signals: Counter = Counter()

    def add(self, other: "Tally") -> None:
        self.fixes.update(other.fixes)
        self.signals.update(other.signals)

    def norm(self, text: str) -> str:
        res = normalize_with_report(text)
        self.fixes.update({k: v for k, v in res.fixes.items() if v})
        self.signals.update({k: v for k, v in res.signals.items() if v})
        return res.text

    def signal_dict(self) -> dict[str, float]:
        out = {f"norm_fix:{k}": float(v) for k, v in self.fixes.items()}
        out.update({f"norm_sig:{k}": float(v) for k, v in self.signals.items()})
        return out
