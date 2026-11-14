"""ตั้ง demo ผ่าน HTTP ของ API ที่รันอยู่: สร้างกล่อง + สมาชิก + ingest PDF รายงานประจำปี 3 เล่ม แล้วรอจน index เสร็จ

    # 1) เริ่ม Qdrant + API (ดู README หัวข้อ "รัน API + demo")
    # 2) ตั้ง demo (key จาก env SERVICE_KEY หรือ .env — ตัวเดียวกับที่ API ใช้)
    python scripts/bootstrap_demo.py --base-url http://127.0.0.1:8000
    python scripts/bootstrap_demo.py --reset         # ลบ DATA_DIR (ยกเว้น ocr_cache) + collection ของ Qdrant (ถามยืนยัน; API ต้องหยุดก่อน)

สิ่งที่สร้าง (ผ่าน admin API ที่มีอยู่: POST /v1/boxes, PUT /v1/boxes/{box_id}/members, POST /v1/ingest, GET /v1/jobs):
- ``annual-reports`` (public, ไม่จำกัดแผนก/ระดับ): KBANK, CPALL, PTT
- ``finance-private`` (private, แผนก finance, ระดับ >= 2): สำเนา PTT 1 เล่ม — ไว้โชว์ ACL
- user / แผนก / ระดับ เป็นชื่อสมมติทั้งหมด (``PERSONAS``) — ผู้สร้างกล่อง (``--admin-user``) ต้องอยู่ใน SYSTEM_ADMIN_USERS ของ API

**ห้าม OCR ใหม่** — ก่อนส่งไฟล์ ตรวจว่าทุกหน้าที่ต้อง OCR มีผลในแคช OCR แล้ว: extract ไฟล์ในเครื่องนี้ (โค้ด extract ชุดเดียวกับ
worker) แล้วเดินเส้นทาง OCR จริง (``run_ocr`` รวมการแบ่งชิ้น/หมุนภาพ) ด้วย backend ที่ห้ามรันโมเดล — เจอหน้าไหนไม่อยู่ในแคช =
หยุด (เว้นแต่ ``--allow-ocr``) · หลังจบนับไฟล์ในแคช OCR ก่อน/หลัง: ไฟล์ใหม่ = จำนวนครั้งที่ OCR จริง (ต้องเป็น 0)
ส่วนนี้อ่านไฟล์ในเครื่องอย่างเดียว (ไม่แตะ API/DATA_DIR) และต้องรันบนเครื่องเดียวกับ API (ใช้แคช OCR ชุดเดียวกัน)

รันซ้ำได้: กล่องที่มีอยู่แล้ว (422) ตรวจผ่าน /v1/sources ว่าเป็นของเราและ visibility ตรง · สมาชิก PUT ซ้ำได้ ·
ไฟล์ซ้ำได้ 200 duplicate (ไม่ index ใหม่) แล้วรายงานสถานะเดิม
จำนวน chunk นับจาก Qdrant ตรง (API ไม่มี endpoint นี้) ด้วย QDRANT_URL / COLLECTION_* จากค่าตั้งชุดเดียวกับ API
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app.config import build_settings, load_env_file  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "eval" / "datasets"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_ADMIN = "demo-sysadmin"
POLL_SECONDS = 3.0  # ต่ำกว่า keep-alive timeout ของ uvicorn (5 วินาที)
GET_RETRIES = 3
TERMINAL = {"done", "done_with_gaps", "failed"}

BOXES = [
    {"box_id": "annual-reports", "name": "รายงานประจำปี (เดโม)", "visibility": "public"},
    {"box_id": "finance-private", "name": "เอกสารฝ่ายการเงิน (เดโม)", "visibility": "private",
     "departments": ["finance"], "min_security_level": 2},
]
FILES = [
    ("annual-reports", "ONEREPORTKBANKT.PDF", None),
    ("annual-reports", "E_ONE_REPORT_CPALLT.PDF", None),
    ("annual-reports", "ONEREPORTPTTT.PDF", None),
    ("finance-private", "ONEREPORTPTTT.PDF", "ONEREPORTPTTT_สำเนาฝ่ายการเงิน.PDF"),
]


@dataclass(frozen=True)
class Persona:
    user_id: str
    departments: str  # ค่า header X-User-Departments
    level: int
    roles: dict = field(default_factory=dict)  # box_id -> role
    note: str = ""


# ชื่อสมมติทั้งหมด (ลงท้าย .demo) — แผนก/ระดับคือค่าที่ Gateway จะส่งมาใน header X-User-*
PERSONAS = [
    Persona("somchai.demo", "marketing", 1, {"annual-reports": "viewer"},
            "เห็น annual-reports · ไม่ใช่สมาชิก finance-private -> ได้ผลว่าง (เหมือนกล่องที่ไม่มีอยู่)"),
    Persona("wanida.demo", "finance", 3, {"annual-reports": "viewer", "finance-private": "viewer"},
            "เห็นทั้งสองกล่อง (finance ระดับ 3)"),
    Persona("preecha.demo", "finance", 1, {"finance-private": "viewer"},
            "สมาชิก finance-private แต่ระดับ 1 < 2 -> ได้ผลว่าง"),
    Persona("anan.demo", "sales", 4, {"finance-private": "viewer"},
            "สมาชิก finance-private แต่แผนก sales ไม่อยู่ในกล่อง -> ได้ผลว่าง"),
    Persona("malee.demo", "finance", 4, {"finance-private": "admin"},
            "admin ของ finance-private (เพิ่มไฟล์/สมาชิกได้)"),
]


class BootstrapError(SystemExit):
    pass


# ---------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------


def service_key() -> str:
    key = os.environ.get("SERVICE_KEY") or load_env_file(ROOT / ".env").get("SERVICE_KEY")
    if not key:
        raise BootstrapError("ไม่มี SERVICE_KEY (ตั้งใน env หรือ .env ให้ตรงกับที่ API ใช้)")
    return key


class Api:
    def __init__(self, base_url: str, key: str, admin: str) -> None:
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(600.0, connect=5.0))
        self.key = key
        self.admin = admin

    def headers(self, user: str | None = None, departments: str = "", level: int = 0) -> dict:
        return {"X-Service-Key": self.key, "X-User-Id": user or self.admin, "X-User-Departments": departments,
                "X-User-Security-Level": str(level)}

    def get(self, path: str, headers: dict) -> httpx.Response:
        """GET แบบลองซ้ำเมื่อ connection หลุด (keep-alive ที่ server ปิดพอดีตอนส่ง) — GET ไม่เปลี่ยนสถานะ ลองซ้ำได้ปลอดภัย"""
        for attempt in range(GET_RETRIES):
            try:
                return self.http.get(path, headers=headers)
            except httpx.TransportError:
                if attempt == GET_RETRIES - 1:
                    raise
                time.sleep(1.0)
        raise AssertionError("unreachable")

    def ready(self) -> dict:
        r = self.get("/v1/health/ready", headers={"X-Service-Key": self.key})
        if r.status_code == 401:
            raise BootstrapError("401: SERVICE_KEY ไม่ตรงกับของ API")
        return r.json()

    def ensure_box(self, box: dict) -> str:
        r = self.http.post("/v1/boxes", headers=self.headers(), json=box)
        if r.status_code == 201:
            return "สร้างใหม่"
        if r.status_code == 404:
            raise BootstrapError(f"สร้างกล่องไม่ได้ (404): {self.admin} ไม่อยู่ใน SYSTEM_ADMIN_USERS ของ API")
        if r.status_code == 422:  # มีอยู่แล้ว — ต้องเป็นกล่องที่เราเป็น owner และ visibility ตรง
            sources = self.get("/v1/sources", headers=self.headers()).json()["sources"]
            mine = {s["box_id"]: s for s in sources}
            got = mine.get(box["box_id"])
            if got is None or got["role"] != "owner":
                raise BootstrapError(f"กล่อง {box['box_id']}: 422 แต่ {self.admin} ไม่ใช่ owner — {r.json()['message']}")
            if got["visibility"] != box["visibility"]:
                raise BootstrapError(f"กล่อง {box['box_id']} มีอยู่แล้วแต่ visibility = {got['visibility']} "
                                     f"(ต้องการ {box['visibility']} — visibility เปลี่ยนไม่ได้ ใช้ --reset)")
            return "มีอยู่แล้ว"
        raise BootstrapError(f"POST /v1/boxes {box['box_id']}: {r.status_code} {r.text}")

    def put_member(self, box_id: str, user_id: str, role: str) -> None:
        r = self.http.put(f"/v1/boxes/{box_id}/members", headers=self.headers(), json={"user_id": user_id, "role": role})
        if r.status_code != 200:
            raise BootstrapError(f"PUT member {user_id} -> {box_id}: {r.status_code} {r.text}")

    def ingest(self, box_id: str, path: Path, display_name: str | None) -> dict:
        data = {"box_id": box_id}
        if display_name:
            data["display_name"] = display_name
        with path.open("rb") as f:
            r = self.http.post("/v1/ingest", headers=self.headers(), data=data, files={"file": (path.name, f)})
        if r.status_code not in (200, 202):
            raise BootstrapError(f"POST /v1/ingest {path.name} -> {box_id}: {r.status_code} {r.text}")
        return r.json()

    def job(self, job_id: str) -> dict:
        r = self.get(f"/v1/jobs/{job_id}", headers=self.headers())
        if r.status_code != 200:
            raise BootstrapError(f"GET /v1/jobs/{job_id}: {r.status_code} {r.text}")
        return r.json()


# ---------------------------------------------------------------------------------------------
# ตรวจแคช OCR (อ่านในเครื่องอย่างเดียว)
# ---------------------------------------------------------------------------------------------


class CacheMiss(RuntimeError):
    pass


class CacheOnlyBackend:
    """backend ตัวจริง (ชื่อ/revision/พารามิเตอร์เดียวกัน -> cache key เดียวกัน) แต่ห้ามรันโมเดล: ไม่อยู่ในแคช = ``CacheMiss``"""

    def __init__(self, real) -> None:
        self._real = real
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def recognize(self, image_png: bytes):
        self.calls += 1
        raise CacheMiss("ไม่มีผลในแคช OCR")


def ocr_cache_files() -> int:
    from app.extraction.ocr_cache import cache_dir

    d = cache_dir()
    return sum(1 for _ in d.glob("*.json")) if d.is_dir() else 0


def check_ocr_cache(paths: list[Path], backend_name: str) -> dict[str, dict]:
    """ต่อไฟล์: จำนวน unit ที่ต้อง OCR และหน้าที่ไม่มีในแคช (เดินเส้นทางเดียวกับ worker: extract_file -> run_ocr)"""
    from app.extraction.ocr import get_backend, run_ocr
    from app.extraction.pipeline import extract_file
    from app.models import UnitStatus

    backend = CacheOnlyBackend(get_backend(backend_name))
    out: dict[str, dict] = {}
    for path in paths:
        t0 = time.perf_counter()
        doc = extract_file(path, file_id="bootstrap-check")
        todo = [u for u in doc.units if u.status == UnitStatus.NEEDS_OCR]
        missing: list[int] = []
        if todo:
            seen = [backend.calls]

            def on_unit(u, seen=seen):
                if backend.calls != seen[0]:
                    missing.append(int(u.position["page"]))
                    seen[0] = backend.calls

            run_ocr(doc, path, backend, on_unit=on_unit)
        out[path.name] = {"units": len(doc.units), "needs_ocr": len(todo), "missing": missing,
                          "seconds": time.perf_counter() - t0}
        print(f"  {path.name}: {len(doc.units)} หน้า · ต้อง OCR {len(todo)} · ไม่อยู่ในแคช {len(missing)}"
              f" ({out[path.name]['seconds']:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------------------------


def reset(base_url: str) -> int:
    settings = build_settings()
    try:
        httpx.get(f"{base_url.rstrip('/')}/v1/health/live", timeout=2.0)
        raise BootstrapError(f"API ยังทำงานอยู่ที่ {base_url} — หยุด API ก่อน reset (SQLite ถูกเปิดอยู่ และ ACL ใน "
                             "หน่วยความจำของ API จะไม่ตรงกับที่ลบ)")
    except httpx.TransportError:
        pass
    from app.extraction.ocr_cache import cache_dir
    from qdrant_client import QdrantClient

    data_dir = settings.data_dir.resolve()
    keep = cache_dir().resolve()
    targets = sorted(p for p in data_dir.iterdir() if p.resolve() != keep) if data_dir.is_dir() else []
    if any(keep.is_relative_to(p.resolve()) for p in targets):
        raise BootstrapError(f"แคช OCR ({keep}) อยู่ใต้สิ่งที่จะลบ — ไม่ลบ")
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=10, check_compatibility=False)
    collections = [c for c in (settings.collection_public, settings.collection_private) if client.collection_exists(c)]

    print(f"จะลบใน DATA_DIR = {data_dir} (เก็บแคช OCR ไว้: {keep})")
    for p in targets:
        print(f"  - {p.name}{'/' if p.is_dir() else ''}")
    print(f"จะลบ collection ของ Qdrant ({settings.qdrant_url}): {', '.join(collections) or '(ไม่มี)'}")
    if not targets and not collections:
        print("ไม่มีอะไรให้ลบ")
        return 0
    answer = input("พิมพ์ yes เพื่อยืนยันการลบ: ").strip()
    if answer != "yes":
        print("ยกเลิก (ไม่ได้ลบอะไร)")
        return 1
    for p in targets:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    for c in collections:
        client.delete_collection(c)
    client.close()
    print("ลบแล้ว — เริ่ม API ใหม่ แล้วรันสคริปต์นี้อีกครั้งโดยไม่ใส่ --reset")
    return 0


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------


def count_chunks(file_id: str, visibility: str) -> int | None:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client import models as qm

        s = build_settings()
        client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=10, check_compatibility=False)
        name = s.collection_private if visibility == "private" else s.collection_public
        flt = qm.Filter(must=[qm.FieldCondition(key="file_id", match=qm.MatchValue(value=file_id))])
        n = client.count(name, count_filter=flt, exact=True).count
        client.close()
        return n
    except Exception as e:  # noqa: BLE001 — เป็นแค่ตัวเลขประกอบรายงาน
        print(f"  (นับ chunk ไม่ได้: {type(e).__name__}: {e})")
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ตั้ง demo ผ่าน HTTP ของ API ที่รันอยู่")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--admin-user", default=DEFAULT_ADMIN, help="ต้องอยู่ใน SYSTEM_ADMIN_USERS ของ API")
    ap.add_argument("--allow-ocr", action="store_true", help="ยอมให้ OCR หน้าที่ไม่มีในแคช (ช้า: นาทีต่อหน้า)")
    ap.add_argument("--reset", action="store_true", help="ลบ DATA_DIR (ยกเว้น ocr_cache) + collection ของ Qdrant (ถามยืนยัน)")
    ap.add_argument("--timeout", type=float, default=4 * 3600, help="วินาทีที่รอ job ทั้งหมด")
    ap.add_argument("--summary-json", type=Path, help="เขียนสรุปผลเป็น JSON")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.reset:
        return reset(args.base_url)

    api = Api(args.base_url, service_key(), args.admin_user)
    try:
        ready = api.ready()
    except httpx.TransportError as e:
        raise BootstrapError(f"ต่อ API ที่ {args.base_url} ไม่ได้: {e}")
    comps, runtime = ready["components"], ready["runtime"]
    print(f"API {args.base_url}: {ready['status']} · qdrant={comps['qdrant']} embedding={comps['embedding']} "
          f"ocr={comps['ocr']} · embedding บน {runtime['embedding_device']} · OCR backend {runtime['ocr_backend']}")
    for note in runtime.get("notes", []):
        print(f"  note: {note}")
    if comps["qdrant"] == "down" or comps["embedding"] == "down":
        raise BootstrapError("API ยังไม่พร้อม (qdrant หรือ embedding down)")

    paths = {name: DATASETS / name for _, name, _ in FILES}
    for p in paths.values():
        if not p.is_file():
            raise BootstrapError(f"ไม่มีไฟล์ {p}")

    print("\n[1/4] ตรวจแคช OCR (extract ในเครื่อง + เดินเส้นทาง OCR โดยห้ามรันโมเดล)")
    cache_before = ocr_cache_files()
    check = check_ocr_cache(list(paths.values()), runtime["ocr_backend"])
    missing = {name: c["missing"] for name, c in check.items() if c["missing"]}
    if missing and not args.allow_ocr:
        for name, pages in missing.items():
            print(f"  {name}: หน้าที่ไม่มีในแคช {pages[:30]}{' …' if len(pages) > 30 else ''}")
        raise BootstrapError("หยุด: มีหน้าที่ต้อง OCR ใหม่ (ใส่ --allow-ocr ถ้าตั้งใจ OCR)")

    print("\n[2/4] กล่อง + สมาชิก")
    for box in BOXES:
        print(f"  {box['box_id']} ({box['visibility']}): {api.ensure_box(box)}")
    for p in PERSONAS:
        for box_id, role in p.roles.items():
            api.put_member(box_id, p.user_id, role)
    print(f"  สมาชิก {sum(len(p.roles) for p in PERSONAS)} รายการ (PUT ซ้ำได้)")

    print("\n[3/4] ส่งไฟล์")
    jobs = []
    for box_id, name, display in FILES:
        t0 = time.time()
        r = api.ingest(box_id, paths[name], display)
        jobs.append({"box_id": box_id, "file": display or name, "job_id": r["job_id"], "file_id": r["file_id"],
                     "duplicate": r["duplicate"], "submitted": t0, "upload_s": time.time() - t0})
        print(f"  {display or name} -> {box_id}: {'ซ้ำ (ไม่ index ใหม่)' if r['duplicate'] else 'เข้าคิว'} "
              f"status={r['status']} job={r['job_id']}", flush=True)

    print("\n[4/4] รอ job")
    deadline = time.time() + args.timeout
    last: dict[str, str] = {}
    while True:
        states = {j["job_id"]: api.job(j["job_id"]) for j in jobs}
        for j in jobs:
            s = states[j["job_id"]]
            line = f"{s['status']} {s['progress']['pages_done']}/{s['progress']['pages_total']}"
            if last.get(j["job_id"]) != line:
                print(f"  {time.strftime('%H:%M:%S')} {j['file']} ({j['box_id']}): {line}", flush=True)
                last[j["job_id"]] = line
            if s["status"] in TERMINAL and "finished" not in j:
                j["finished"] = time.time()
        if all(s["status"] in TERMINAL for s in states.values()):
            break
        if time.time() > deadline:
            raise BootstrapError("หมดเวลารอ job")
        time.sleep(POLL_SECONDS)
    cache_after = ocr_cache_files()

    visibility = {b["box_id"]: b["visibility"] for b in BOXES}
    rows = []
    for j in jobs:
        s = states[j["job_id"]]
        rows.append({
            "file": j["file"], "box_id": j["box_id"], "file_id": j["file_id"], "job_id": j["job_id"],
            "duplicate": j["duplicate"], "status": s["status"], "units": s["units"],
            "gaps": [{"label": g["label"], "status": g["status"], "reason": g["reason"]} for g in s["gaps"]],
            "error": s["error"], "chunks": count_chunks(j["file_id"], visibility[j["box_id"]]),
            "seconds_wall": round(j["finished"] - j["submitted"], 1),
            "job_created_at": s["created_at"], "job_updated_at": s["updated_at"],
        })

    print("\nสรุปต่อไฟล์")
    print("| ไฟล์ | กล่อง | ซ้ำ | status | ok | failed | needs_ocr | rejected | chunk | เวลา (s) |")
    print("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        u = r["units"]
        print(f"| {r['file']} | {r['box_id']} | {'ใช่' if r['duplicate'] else '-'} | {r['status']} | {u['ok']} | "
              f"{u['failed']} | {u['needs_ocr']} | {u['rejected']} | {r['chunks']} | {r['seconds_wall']} |")
    for r in rows:
        for g in r["gaps"]:
            print(f"  gap {r['file']} {g['label']}: {g['status']} — {g['reason']}")
        if r["error"]:
            print(f"  error {r['file']}: {r['error']}")
    new_ocr = cache_after - cache_before
    print(f"\nแคช OCR: {cache_before} -> {cache_after} ไฟล์ · OCR ใหม่ = {new_ocr} ครั้ง"
          f" (ตรวจก่อนส่ง: หน้าไม่อยู่ในแคช {sum(len(v) for v in missing.values())})")

    print("\npersona สำหรับเดโม (ค่าที่ Gateway ส่งใน header X-User-*; ชื่อสมมติ)")
    for p in PERSONAS:
        print(f"  {p.user_id:14s} departments={p.departments:10s} level={p.level}  {p.roles} — {p.note}")

    if args.summary_json:
        args.summary_json.write_text(json.dumps({
            "rows": rows, "ocr_cache_before": cache_before, "ocr_cache_after": cache_after, "new_ocr": new_ocr,
            "precheck": check, "ready": ready,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    failed = [r for r in rows if r["status"] == "failed"]
    return 1 if failed or (new_ocr and not args.allow_ocr) else 0


if __name__ == "__main__":
    raise SystemExit(main())
