"""กู้ข้อมูลจาก zip ของ ``scripts/backup.py`` ลง DATA_DIR + collection ของ Qdrant แล้วตรวจจำนวนกับ manifest

    python scripts/restore.py backups/rag-backup-....zip --data-dir D:/rag-restore \\
        --collections restore_public restore_private        # ลงที่ใหม่ (ค่าเริ่มต้น = ชื่อ collection เดิมใน manifest)
    python scripts/restore.py <zip> --data-dir data --force    # ทับของเดิม (หยุด API ก่อน)

ลำดับ: ตรวจ manifest + sha256 ของทุกไฟล์ใน zip **ก่อนเขียนอะไร** -> ตรวจปลายทาง (มี rag.db / files / documents อยู่แล้ว
หรือ collection ปลายทางมีอยู่แล้ว = ไม่ทำ เว้นแต่ ``--force``) -> แตกไฟล์ (rag.db เขียนชื่อชั่วคราวแล้ว rename) ->
อัปโหลด snapshot เข้า collection ปลายทาง (Qdrant snapshot API, priority=snapshot) -> นับ point / แถว / ไฟล์ เทียบ manifest
ไม่ตรง = exit 1
ใช้ collection ชื่อใหม่ -> ตั้ง ``COLLECTION_PUBLIC`` / ``COLLECTION_PRIVATE`` ให้ API ตามที่สคริปต์พิมพ์บอก
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import build_settings  # noqa: E402
from scripts.backup import CHUNK, FORMAT, TABLES, QdrantSnapshots  # noqa: E402

PREFIXES = {"data/files/": "files", "data/documents/": "documents", "data/ocr_cache/": "ocr_cache"}


class RestoreError(SystemExit):
    pass


def read_manifest(z: zipfile.ZipFile) -> dict:
    try:
        manifest = json.loads(z.read("manifest.json"))
    except KeyError:
        raise RestoreError("ไม่มี manifest.json — ไม่ใช่ไฟล์จาก scripts/backup.py") from None
    if manifest.get("format") != FORMAT:
        raise RestoreError(f"รูปแบบ {manifest.get('format')!r} ไม่รองรับ (ต้องเป็น {FORMAT})")
    return manifest


def verify_checksums(z: zipfile.ZipFile, manifest: dict) -> None:
    expected = manifest["checksums"]
    names = {n for n in z.namelist() if n != "manifest.json"}
    if names != set(expected):
        raise RestoreError(f"ไฟล์ใน zip ไม่ตรง manifest (ขาด {sorted(set(expected) - names)[:5]}, "
                           f"เกิน {sorted(names - set(expected))[:5]})")
    for name, digest in expected.items():
        h = hashlib.sha256()
        with z.open(name) as f:
            while block := f.read(CHUNK):
                h.update(block)
        if h.hexdigest() != digest:
            raise RestoreError(f"sha256 ของ {name} ไม่ตรง manifest — ไฟล์สำรองเสีย")


def safe_target(base: Path, rel: str) -> Path:
    """กัน zip-slip: path ต้องอยู่ใต้ ``base``"""
    parts = PurePosixPath(rel).parts
    if not parts or any(p in ("..", "") for p in parts) or PurePosixPath(rel).is_absolute():
        raise RestoreError(f"path ใน zip ไม่ปลอดภัย: {rel}")
    target = base.joinpath(*parts)
    if base.resolve() not in target.resolve().parents:
        raise RestoreError(f"path ใน zip ไม่ปลอดภัย: {rel}")
    return target


def extract_member(z: zipfile.ZipFile, name: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".restoring")
    with z.open(name) as src, open(tmp, "wb") as out:
        shutil.copyfileobj(src, out, CHUNK)
    tmp.replace(dest)


def sqlite_counts(db: Path) -> tuple[Optional[int], dict[str, int]]:
    conn = sqlite3.connect(db)
    try:
        version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        return (int(version[0]) if version else None,
                {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES})
    finally:
        conn.close()


def restore(zip_path: Path, *, data_dir: Path, ocr_cache_dir: Path, qdrant: QdrantSnapshots,
            collections: Optional[dict[str, str]] = None, force: bool = False, log=print) -> dict:
    with zipfile.ZipFile(zip_path) as z:
        manifest = read_manifest(z)
        log(f"ตรวจ sha256 ของ {len(manifest['checksums'])} ไฟล์ ...")
        verify_checksums(z, manifest)

        source = {q["role"]: name for name, q in manifest["qdrant"].items()}
        targets = {role: (collections or {}).get(role, name) for role, name in source.items()}

        # ---- ปลายทางต้องว่าง (เว้นแต่ --force) ----
        occupied = [str(p) for p in (data_dir / "rag.db", data_dir / "files", data_dir / "documents")
                    if p.exists() and (p.is_file() or any(p.iterdir()))]
        existing = [t for t in targets.values() if qdrant.exists(t)]
        if (occupied or existing) and not force:
            raise RestoreError(f"ปลายทางมีข้อมูลอยู่แล้ว: {occupied + [f'collection {c}' for c in existing]} — "
                               "เลือกที่ใหม่ หรือ --force เพื่อทับ (หยุด API ก่อน)")

        # ---- ไฟล์ ----
        data_dir.mkdir(parents=True, exist_ok=True)
        if force:
            for folder in ("files", "documents"):
                shutil.rmtree(data_dir / folder, ignore_errors=True)
            for suffix in ("-wal", "-shm"):
                (data_dir / f"rag.db{suffix}").unlink(missing_ok=True)
        log("แตกไฟล์ ...")
        for name in manifest["checksums"]:
            if name == "sqlite/rag.db":
                extract_member(z, name, data_dir / "rag.db")
                continue
            for prefix, folder in PREFIXES.items():
                if name.startswith(prefix):
                    base = ocr_cache_dir if folder == "ocr_cache" else data_dir / folder
                    extract_member(z, name, safe_target(base, name[len(prefix):]))
                    break

        # ---- Qdrant ----
        with tempfile.TemporaryDirectory(prefix="rag-restore-") as tmpdir:
            for name, q in manifest["qdrant"].items():
                target = targets[q["role"]]
                snap = Path(tmpdir) / f"{name}.snapshot"
                with z.open(q["file"]) as src, open(snap, "wb") as out:
                    shutil.copyfileobj(src, out, CHUNK)
                log(f"Qdrant: {name} -> {target} ...")
                qdrant.upload(target, snap)
                snap.unlink()

    # ---- ตรวจ ----
    problems: list[str] = []
    version, tables = sqlite_counts(data_dir / "rag.db")
    if version != manifest["sqlite"]["schema_version"] or tables != manifest["sqlite"]["tables"]:
        problems.append(f"SQLite ไม่ตรง: v{version} {tables} vs manifest {manifest['sqlite']}")
    points = {}
    for name, q in manifest["qdrant"].items():
        target = targets[q["role"]]
        points[target] = qdrant.count(target)
        if points[target] != q["points"]:
            problems.append(f"{target}: {points[target]} point ไม่ตรง manifest ({q['points']})")
    folders = {}
    for folder, info in manifest["folders"].items():
        base = ocr_cache_dir if folder == "ocr_cache" else data_dir / folder
        n = sum(1 for p in base.rglob("*") if p.is_file()) if base.is_dir() else 0
        folders[folder] = n
        # files/documents ปลายทางว่างก่อนแตก (หรือถูกล้างด้วย --force) ต้องเท่ากันพอดี · ocr_cache อาจมีของเดิมอยู่: ต้องไม่น้อยกว่า
        if (n < info["files"]) if folder == "ocr_cache" else (n != info["files"]):
            problems.append(f"{folder}/: {n} ไฟล์ ไม่ตรง manifest ({info['files']})")
    report = {"manifest": {k: manifest[k] for k in ("created_at", "sqlite", "folders")},
              "restored": {"data_dir": str(data_dir), "ocr_cache_dir": str(ocr_cache_dir), "sqlite_tables": tables,
                           "schema_version": version, "points": points, "folders": folders,
                           "collections": targets},
              "problems": problems}
    return report


def main(argv: Optional[list[str]] = None) -> int:
    settings = build_settings()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zip", type=Path)
    ap.add_argument("--data-dir", type=Path, required=True, help="DATA_DIR ปลายทาง (บังคับระบุ กันทับของจริงโดยไม่ตั้งใจ)")
    ap.add_argument("--ocr-cache-dir", type=Path, default=None, help="ค่าเริ่มต้น <data-dir>/ocr_cache")
    ap.add_argument("--qdrant-url", default=settings.qdrant_url)
    ap.add_argument("--collections", nargs=2, metavar=("PUBLIC", "PRIVATE"), default=None,
                    help="ชื่อ collection ปลายทาง (ค่าเริ่มต้น = ชื่อเดิมใน manifest)")
    ap.add_argument("--force", action="store_true", help="ทับข้อมูล/collection ที่มีอยู่")
    args = ap.parse_args(argv)
    ocr_dir = args.ocr_cache_dir or args.data_dir / "ocr_cache"
    collections = None if args.collections is None else {"public": args.collections[0],
                                                          "private": args.collections[1]}
    report = restore(args.zip, data_dir=args.data_dir, ocr_cache_dir=ocr_dir,
                     qdrant=QdrantSnapshots(args.qdrant_url, settings.qdrant_api_key), collections=collections,
                     force=args.force)
    r = report["restored"]
    print(f"\nกู้แล้ว -> {r['data_dir']} (SQLite schema v{r['schema_version']}: " +
          ", ".join(f"{k} {v}" for k, v in r["sqlite_tables"].items()) + ")")
    for name, n in r["points"].items():
        print(f"  Qdrant {name}: {n} point")
    for name, n in r["folders"].items():
        print(f"  {name}/: {n} ไฟล์")
    print("ตั้งค่า API ให้ชี้ที่นี่: DATA_DIR=%s OCR_CACHE_DIR=%s COLLECTION_PUBLIC=%s COLLECTION_PRIVATE=%s"
          % (r["data_dir"], r["ocr_cache_dir"], r["collections"]["public"], r["collections"]["private"]))
    if report["problems"]:
        print("\nไม่ตรง manifest:")
        for p in report["problems"]:
            print(f"  - {p}")
        return 1
    print("ตรวจแล้ว: จำนวนแถว / point / ไฟล์ ตรงกับ manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
