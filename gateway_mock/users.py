"""ผู้ใช้สมมติของ mock Gateway — ตรงกับ PERSONAS ใน ``scripts/bootstrap_demo.py`` (มี test เทียบให้ตรงกัน)

Gateway เป็นคนถือ identity: แผนกและระดับความลับมาจากที่นี่ ไม่ใช่จากผู้เรียก (ผู้ใช้ปลายทางส่ง X-User-* เองไม่ได้)
สมาชิกกล่องไม่อยู่ที่นี่ — RAG เป็นแหล่งความจริงของสมาชิกและ ACL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class User:
    user_id: str
    departments: str  # ค่าที่ส่งใน X-User-Departments (คั่นด้วยจุลภาค, ว่าง = ไม่สังกัดแผนก)
    security_level: int
    note: str = ""


DEMO_USERS: tuple[User, ...] = (
    User("somchai.demo", "marketing", 1, "สมาชิก annual-reports เท่านั้น"),
    User("wanida.demo", "finance", 3, "เห็น annual-reports + finance-private"),
    User("preecha.demo", "finance", 1, "สมาชิก finance-private แต่ระดับ 1 < 2"),
    User("anan.demo", "sales", 4, "สมาชิก finance-private แต่แผนก sales ไม่อยู่ในกล่อง"),
    User("malee.demo", "finance", 4, "admin ของ finance-private"),
)


class UserDirectory:
    def __init__(self, users: tuple[User, ...] = DEMO_USERS) -> None:
        self._by_id = {u.user_id: u for u in users}

    def get(self, user_id: Optional[str]) -> Optional[User]:
        return self._by_id.get(user_id or "")

    def all(self) -> list[User]:
        return list(self._by_id.values())
