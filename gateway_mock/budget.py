"""งบประมาณราย user แบบง่าย (นับจำนวนคำถามในหน่วยความจำ) — จำลองว่า \"งบ/คิว\" เป็นหน้าที่ของ Gateway ไม่ใช่ RAG"""

from __future__ import annotations

import threading
from collections import Counter


class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, limit_per_user: int) -> None:
        self._limit = limit_per_user  # <= 0 = ไม่จำกัด
        self._used: Counter[str] = Counter()
        self._lock = threading.Lock()

    def charge(self, user_id: str) -> None:
        with self._lock:
            if self._limit > 0 and self._used[user_id] >= self._limit:
                raise BudgetExceeded(user_id)
            self._used[user_id] += 1

    def refund(self, user_id: str) -> None:
        """คืนโควตา 1 ครั้ง — ใช้เมื่อคำถามไม่ได้รับบริการเพราะระบบหลังบ้านล่ม (ไม่ใช่ความผิดของผู้ใช้)"""
        with self._lock:
            if self._used[user_id] > 0:
                self._used[user_id] -= 1

    def used(self, user_id: str) -> int:
        return self._used[user_id]
