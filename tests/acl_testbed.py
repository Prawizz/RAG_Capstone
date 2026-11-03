"""ตัวอ่าน testbed สิทธิ์จาก ``eval/datasets/access_control.jsonl`` (ใช้ร่วมกันหลาย test)

ไฟล์ jsonl เป็นคนเขียน ไม่ได้สร้างจาก ``app/access.py`` (กฎเหล็กข้อ 6) โมดูลนี้แค่อ่าน ไม่ตัดสินอะไรเอง
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.access import AclStore, Box, FileAcl, Principal

TESTBED = Path(__file__).resolve().parents[1] / "eval" / "datasets" / "access_control.jsonl"


@dataclass(frozen=True)
class Case:
    user_id: str
    file_id: str
    expected: str  # "allow" | "deny"
    layer: Optional[str]
    note: str = ""

    @property
    def should_allow(self) -> bool:
        return self.expected == "allow"

    def __str__(self) -> str:  # ชื่อที่อ่านรู้เรื่องเวลา pytest รายงาน
        return f"{self.user_id}->{self.file_id}={self.expected}"


@dataclass
class Testbed:
    store: AclStore
    principals: dict[str, Principal]
    cases: list[Case]
    boxes: dict[str, Box]
    files: dict[str, FileAcl]


def load_testbed(path: Path = TESTBED) -> Testbed:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    boxes, files, principals, cases = {}, {}, {}, []
    store = AclStore()
    for row in rows:
        kind = row["kind"]
        if kind == "box":
            box = Box(
                box_id=row["box_id"],
                visibility=row["visibility"],
                departments=row["departments"],
                min_security_level=row["min_security_level"],
            )
            boxes[box.box_id] = box
            store.register_box(box)
        elif kind == "file":
            file = FileAcl(
                file_id=row["file_id"],
                box_id=row["box_id"],
                departments=row.get("departments"),
                min_security_level=row.get("min_security_level"),
            )
            files[file.file_id] = file
            store.register_file(file)
        elif kind == "principal":
            principals[row["user_id"]] = Principal(
                user_id=row["user_id"],
                departments=row["departments"],
                security_level=row["security_level"],
                box_ids=row["box_ids"],
            )
        elif kind == "case":
            cases.append(
                Case(
                    user_id=row["user_id"],
                    file_id=row["file_id"],
                    expected=row["expected"],
                    layer=row.get("layer"),
                    note=row.get("note", ""),
                )
            )
    return Testbed(store=store, principals=principals, cases=cases, boxes=boxes, files=files)
