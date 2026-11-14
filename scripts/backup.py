"""สำรองข้อมูลทั้งระบบเป็น zip เดียว: SQLite + ไฟล์บนดิสก์ + Qdrant (2 collection) พร้อม manifest

    python scripts/backup.py                         # -> backups/rag-backup-<เวลา UTC>.zip
    python scripts/backup.py --out D:/bk/rag.zip     # ระบุที่เก็บเอง
    python scripts/restore.py <zip> --data-dir ...   # คู่กัน (ดู restore.py)

ของในไฟล์ zip
    manifest.json                    เวลา, เวอร์ชัน schema, จำนวนแถวต่อตาราง, จำนวนไฟล์/ไบต์ต่อโฟลเดอร์, จำนวน point ต่อ collection,
                                     sha256 ของทุกไฟล์ใน zip (restore ตรวจก่อนเขียนอะไร)
    sqlite/rag.db                    จาก **SQLite online backup API** (ไม่ copy rag.db ตรงๆ: ฐานเป็น WAL — ไฟล์ .db อย่างเดียว
                                     ไม่มีข้อมูลที่ยังอยู่ใน -wal และ copy ระหว่างมีคนเขียนได้ไฟล์เสีย) ไฟล์สำรองเป็น journal_mode=DELETE
    data/files/ data/documents/      ต้นฉบับ + ผล extract (ตาม DATA_DIR)
    data/ocr_cache/                  แคช OCR (OCR_CACHE_DIR)
    qdrant/<collection>.snapshot     จาก **snapshot API ของ Qdrant** (สร้างบน server -> ดาวน์โหลด -> ลบ snapshot บน server)

ความสอดคล้อง: SQLite กับ Qdrant ต้องสำรองพร้อมกัน (point อ้าง file_id ใน SQLite) — สคริปต์ไม่ยอมสำรองถ้ามีงาน ingest
กำลังทำ (extracting/ocr/indexing) หรือไฟล์ค้างสถานะ deleting เพราะ point/ไฟล์บนดิสก์กำลังเปลี่ยน (``--allow-busy`` = ข้าม)
ลำดับ: SQLite ก่อน แล้ว Qdrant แล้วไฟล์ — ไฟล์ที่อัปโหลดเข้ามาหลังจุด SQLite จะเป็นไฟล์เกิน (ไม่มีแถวอ้างถึง) ไม่ทำให้ข้อมูลผิด
สคริปต์นี้อ่านอย่างเดียว (SQLite เปิดแบบ read-only) ยกเว้นสร้าง/ลบ snapshot ชั่วคราวบน Qdrant server
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.config import build_settings  # noqa: E402
from app.extraction.ocr_cache import cache_dir  # noqa: E402

FORMAT = "rag-backup/1"
TABLES = ("boxes", "members", "file_acls", "files", "jobs")
DATA_FOLDERS = ("files", "documents")  # ใต้ DATA_DIR (ocr_cache แยก — อยู่ที่ OCR_CACHE_DIR)
CHUNK = 1 << 20


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


# ---------- Qdrant ----------


class QdrantSnapshots:
    """snapshot API ของ Qdrant ผ่าน HTTP (qdrant-client สร้าง/ลบได้ แต่ดาวน์โหลด/อัปโหลดไฟล์ต้องใช้ endpoint ตรง)"""

    def __init__(self, url: str, api_key: Optional[str] = None, timeout: float = 600.0) -> None:
        self.url = url.rstrip("/")
        self.headers = {"api-key": api_key} if api_key else {}
        self.timeout = timeout

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.url, headers=self.headers, timeout=self.timeout)

    def exists(self, collection: str) -> bool:
        with self._client() as c:
            r = c.get(f"/collections/{collection}/exists")
            r.raise_for_status()
            return bool(r.json()["result"]["exists"])

    def count(self, collection: str) -> int:
        with self._client() as c:
            r = c.post(f"/collections/{collection}/points/count", json={"exact": True})
            r.raise_for_status()
            return int(r.json()["result"]["count"])

    def download(self, collection: str, dest: Path) -> dict:
        """สร้าง snapshot บน server -> ดาวน์โหลดลง ``dest`` -> ลบ snapshot บน server (ไม่ทิ้งไฟล์ค้าง)"""
        with self._client() as c:
            r = c.post(f"/collections/{collection}/snapshots", params={"wait": "true"})
            r.raise_for_status()
            snap = r.json()["result"]
            name = snap["name"]
            try:
                with c.stream("GET", f"/collections/{collection}/snapshots/{name}") as resp:
                    resp.raise_for_status()
                    with open(dest, "wb") as f:
                        for block in resp.iter_bytes(CHUNK):
                            f.write(block)
            finally:
                c.delete(f"/collections/{collection}/snapshots/{name}", params={"wait": "true"})
        return {"server_snapshot": name, "checksum": snap.get("checksum")}

    def upload(self, collection: str, path: Path, checksum: Optional[str] = None) -> None:
        """กู้ collection จากไฟล์ snapshot (priority=snapshot: แทนที่ของเดิมทั้งหมดถ้ามี)"""
        params = {"wait": "true", "priority": "snapshot"}
        if checksum:
            params["checksum"] = checksum
        with self._client() as c, open(path, "rb") as f:
            r = c.post(f"/collections/{collection}/snapshots/upload", params=params,
                       files={"snapshot": (path.name, f, "application/octet-stream")})
            r.raise_for_status()

    def delete_collection(self, collection: str) -> None:
        with self._client() as c:
            c.delete(f"/collections/{collection}").raise_for_status()


# ---------- SQLite ----------


def sqlite_backup(db: Path, dest: Path) -> dict:
    """SQLite online backup API จากฐานที่เปิดแบบ read-only (อ่านรวมข้อมูลใน WAL) -> ไฟล์เดียว (journal_mode=DELETE)"""
    if not db.is_file():
        raise SystemExit(f"ไม่พบ {db}")
    src = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst)
        dst.execute("PRAGMA journal_mode=DELETE")
        version = dst.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        counts = {t: dst.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
        integrity = dst.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        dst.close()
        src.close()
    if integrity != "ok":
        raise SystemExit(f"สำเนา SQLite ไม่ผ่าน integrity_check: {integrity}")
    return {"schema_version": int(version[0]) if version else None, "tables": counts}


def busy_state(db: Path) -> dict:
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        running = conn.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('extracting', 'ocr', 'indexing')").fetchone()[0]
        deleting = conn.execute("SELECT COUNT(*) FROM files WHERE status = 'deleting'").fetchone()[0]
    finally:
        conn.close()
    return {"jobs_in_progress": running, "files_deleting": deleting}


# ---------- zip ----------


@dataclass
class Member:
    arcname: str
    path: Path


def walk(folder: Path, prefix: str) -> list[Member]:
    if not folder.is_dir():
        return []
    return [Member(f"{prefix}/{p.relative_to(folder).as_posix()}", p)
            for p in sorted(folder.rglob("*")) if p.is_file() and not p.name.endswith(".tmp")]


def write_zip(out: Path, members: Iterable[Member], manifest: dict) -> None:
    tmp = out.with_suffix(out.suffix + ".partial")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for m in members:
            z.write(m.path, m.arcname)
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    tmp.replace(out)  # ไม่มี zip ครึ่งๆ กลางๆ ที่ชื่อจริง


def backup(*, data_dir: Path, ocr_cache_dir: Path, qdrant: QdrantSnapshots, collections: dict[str, str],
           out: Path, allow_busy: bool = False, log=print) -> dict:
    db = data_dir / "rag.db"
    state = busy_state(db)
    if (state["jobs_in_progress"] or state["files_deleting"]) and not allow_busy:
        raise SystemExit(f"มีงานกำลังทำ {state['jobs_in_progress']} / ไฟล์ค้าง deleting {state['files_deleting']} — "
                         "รอให้จบ (หรือหยุด API) แล้วสำรองใหม่ (--allow-busy = สำรองทั้งที่ข้อมูลกำลังเปลี่ยน)")
    out.parent.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    with tempfile.TemporaryDirectory(prefix="rag-backup-", dir=out.parent) as tmpdir:
        work = Path(tmpdir)
        log("SQLite (online backup API) ...")
        sqlite_info = sqlite_backup(db, work / "rag.db")
        members = [Member("sqlite/rag.db", work / "rag.db")]

        qinfo: dict[str, dict] = {}
        for role, name in collections.items():
            if not qdrant.exists(name):
                raise SystemExit(f"ไม่มี collection {name} บน Qdrant")
            points = qdrant.count(name)
            log(f"Qdrant snapshot {name} ({points} point) ...")
            snap = work / f"{name}.snapshot"
            meta = qdrant.download(name, snap)
            members.append(Member(f"qdrant/{name}.snapshot", snap))
            qinfo[name] = {"role": role, "points": points, "file": f"qdrant/{name}.snapshot",
                           "bytes": snap.stat().st_size, **meta}

        folders: dict[str, dict] = {}
        for folder in DATA_FOLDERS:
            found = walk(data_dir / folder, f"data/{folder}")
            members += found
            folders[folder] = {"files": len(found), "bytes": sum(m.path.stat().st_size for m in found)}
        found = walk(ocr_cache_dir, "data/ocr_cache")
        members += found
        folders["ocr_cache"] = {"files": len(found), "bytes": sum(m.path.stat().st_size for m in found)}

        log(f"sha256 ของ {len(members)} ไฟล์ ...")
        manifest = {
            "format": FORMAT,
            "created_at": created_at,
            "source": {"data_dir": str(data_dir), "ocr_cache_dir": str(ocr_cache_dir), "qdrant_url": qdrant.url},
            "busy_at_backup": state,
            "sqlite": sqlite_info,
            "qdrant": qinfo,
            "folders": folders,
            "checksums": {m.arcname: sha256_of(m.path) for m in members},
        }
        log(f"เขียน {out} ...")
        write_zip(out, members, manifest)
    return manifest


def main(argv: Optional[list[str]] = None) -> int:
    settings = build_settings()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=settings.data_dir)
    ap.add_argument("--ocr-cache-dir", type=Path, default=None, help="ค่าเริ่มต้น OCR_CACHE_DIR / data/ocr_cache")
    ap.add_argument("--qdrant-url", default=settings.qdrant_url)
    ap.add_argument("--collections", nargs=2, metavar=("PUBLIC", "PRIVATE"),
                    default=[settings.collection_public, settings.collection_private])
    ap.add_argument("--out", type=Path, default=None, help="ค่าเริ่มต้น backups/rag-backup-<UTC>.zip")
    ap.add_argument("--allow-busy", action="store_true")
    args = ap.parse_args(argv)
    out = args.out or ROOT / "backups" / f"rag-backup-{utc_stamp()}.zip"
    manifest = backup(
        data_dir=args.data_dir, ocr_cache_dir=args.ocr_cache_dir or cache_dir(),
        qdrant=QdrantSnapshots(args.qdrant_url, settings.qdrant_api_key),
        collections={"public": args.collections[0], "private": args.collections[1]}, out=out,
        allow_busy=args.allow_busy,
    )
    size = out.stat().st_size
    print(f"\nสำรองแล้ว: {out} ({size / 1e6:.1f} MB)")
    print(f"  SQLite schema v{manifest['sqlite']['schema_version']}: " +
          ", ".join(f"{k} {v}" for k, v in manifest["sqlite"]["tables"].items()))
    for name, q in manifest["qdrant"].items():
        print(f"  Qdrant {name}: {q['points']} point ({q['bytes'] / 1e6:.1f} MB)")
    for name, f in manifest["folders"].items():
        print(f"  {name}/: {f['files']} ไฟล์ ({f['bytes'] / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
