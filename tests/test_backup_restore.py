"""scripts/backup.py + scripts/restore.py

test ปกติ: snapshot ของ Qdrant เป็นของปลอม (เก็บไบต์ไว้ในหน่วยความจำ) — ตรวจ manifest, sha256, SQLite online backup,
การไม่ยอมทับปลายทาง, zip เสีย, งานกำลังทำ
``-m qdrant``: snapshot API ของ Qdrant จริง (collection ชื่อเฉพาะของ test ลบทิ้งท้ายเสมอ) + ค้นจาก collection ที่กู้ได้ผลเดิม
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from app.access import AclStore, Box, FileAcl, Visibility
from app.storage import FileRecord, JobRecord, SqliteStorage
from scripts.backup import backup
from scripts.restore import RestoreError, restore


class FakeSnapshots:
    def __init__(self, collections: dict[str, int]) -> None:
        self.data = {name: (f"snapshot-of-{name}".encode() * 100, n) for name, n in collections.items()}
        self.url = "http://fake-qdrant"
        self.uploaded: list[str] = []

    def exists(self, name):
        return name in self.data

    def count(self, name):
        return self.data[name][1]

    def download(self, name, dest: Path):
        dest.write_bytes(self.data[name][0])
        return {"server_snapshot": f"{name}.snapshot", "checksum": None}

    def upload(self, name, path: Path, checksum=None):
        src = path.read_bytes()
        n = next(n for b, n in self.data.values() if b == src)
        self.data[name] = (src, n)
        self.uploaded.append(name)


def make_data(root: Path) -> tuple[Path, Path]:
    data, ocr = root / "data", root / "ocr_cache"
    s = SqliteStorage(data)
    acl = AclStore(memberships=s.memberships, persistence=s)
    acl.create_box(Box(box_id="b", name="กล่องทดสอบ", visibility=Visibility.PUBLIC), "owner1")
    for i in range(3):
        fid = f"file_{i}"
        acl.register_file(FileAcl(file_id=fid, box_id="b"))
        s.insert_file(FileRecord(file_id=fid, box_id="b", filename=f"รายงาน {i}.pdf", content_hash=f"h{i}",
                                 size_bytes=10, status="done", submitted_by="owner1"))
        s.insert_job(JobRecord(job_id=f"job_{i}", file_id=fid, box_id="b", submitted_by="owner1", status="done"))
        s.original_path(fid).write_bytes(b"%PDF-1.7 " + bytes([i]) * 10)
        s.document_path(fid).write_text(json.dumps({"file_id": fid, "units": []}), encoding="utf-8")
    s.close()  # WAL ยังไม่ checkpoint บางส่วน -> online backup ต้องอ่านรวม -wal ได้
    ocr.mkdir()
    (ocr / "a.json").write_text('{"fields": {"content_hash": "h0"}}', encoding="utf-8")
    return data, ocr


def run_backup(tmp_path: Path, snaps=None, **kw) -> tuple[Path, dict, Path, Path]:
    data, ocr = make_data(tmp_path / "src")
    snaps = snaps or FakeSnapshots({"src_public": 7, "src_private": 3})
    out = tmp_path / "out" / "bk.zip"
    manifest = backup(data_dir=data, ocr_cache_dir=ocr, qdrant=snaps,
                      collections={"public": "src_public", "private": "src_private"}, out=out, log=lambda *_: None, **kw)
    return out, manifest, data, ocr


def test_backup_zip_has_manifest_counts_and_checksums(tmp_path):
    out, manifest, data, _ = run_backup(tmp_path)
    assert manifest["format"] == "rag-backup/1" and manifest["created_at"].endswith("Z")
    assert manifest["sqlite"] == {"schema_version": 2,
                                  "tables": {"boxes": 1, "members": 1, "file_acls": 3, "files": 3, "jobs": 3}}
    assert manifest["qdrant"]["src_public"]["points"] == 7 and manifest["qdrant"]["src_private"]["role"] == "private"
    assert manifest["folders"] == {"files": {"files": 3, "bytes": 57}, "documents": {"files": 3, "bytes":
                                   sum(p.stat().st_size for p in (data / "documents").iterdir())},
                                   "ocr_cache": {"files": 1, "bytes": 34}}
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert names == set(manifest["checksums"]) | {"manifest.json"}
        assert {"sqlite/rag.db", "qdrant/src_public.snapshot", "data/files/file_0", "data/documents/file_0.json",
                "data/ocr_cache/a.json"} <= names
        db = tmp_path / "check.db"
        db.write_bytes(z.read("sqlite/rag.db"))
    conn = sqlite3.connect(db)  # สำเนาเป็นไฟล์เดียว (journal DELETE) ที่มีข้อมูลครบ
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert conn.execute("SELECT filename FROM files WHERE file_id = 'file_2'").fetchone()[0] == "รายงาน 2.pdf"
    conn.close()
    assert not list(out.parent.glob("*.partial")) and not list(out.parent.glob("rag-backup-*"))  # ไม่มีของค้าง


def test_restore_to_a_new_place_matches_the_manifest(tmp_path):
    snaps = FakeSnapshots({"src_public": 7, "src_private": 3})
    out, manifest, data, _ = run_backup(tmp_path, snaps)
    dest = tmp_path / "restored"
    report = restore(out, data_dir=dest / "data", ocr_cache_dir=dest / "ocr", qdrant=snaps,
                     collections={"public": "tmp_public", "private": "tmp_private"}, log=lambda *_: None)
    assert report["problems"] == []
    assert report["restored"]["sqlite_tables"] == manifest["sqlite"]["tables"]
    assert report["restored"]["points"] == {"tmp_public": 7, "tmp_private": 3}
    assert snaps.uploaded == ["tmp_public", "tmp_private"]  # ไม่แตะ collection ต้นทาง
    assert (dest / "data" / "files" / "file_1").read_bytes() == (data / "files" / "file_1").read_bytes()
    assert (dest / "ocr" / "a.json").is_file()
    s = SqliteStorage(dest / "data")  # เปิดใช้ได้จริง
    try:
        assert s.get_file("file_1").filename == "รายงาน 1.pdf"
    finally:
        s.close()


def test_restore_refuses_occupied_targets_without_force(tmp_path):
    snaps = FakeSnapshots({"src_public": 7, "src_private": 3})
    out, _, data, ocr = run_backup(tmp_path, snaps)
    with pytest.raises(RestoreError, match="ปลายทางมีข้อมูลอยู่แล้ว"):  # ทับของจริงต้นทาง
        restore(out, data_dir=data, ocr_cache_dir=ocr, qdrant=snaps, log=lambda *_: None)
    with pytest.raises(RestoreError, match="collection src_public"):
        restore(out, data_dir=tmp_path / "empty", ocr_cache_dir=tmp_path / "o", qdrant=snaps, log=lambda *_: None)
    assert not (tmp_path / "empty").exists() and snaps.uploaded == []


def test_tampered_backup_is_rejected_before_writing_anything(tmp_path):
    snaps = FakeSnapshots({"src_public": 7, "src_private": 3})
    out, _, _, _ = run_backup(tmp_path, snaps)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(bad, "w") as dst:
        for item in src.infolist():
            payload = src.read(item.filename)
            dst.writestr(item, b"tampered" if item.filename == "data/files/file_0" else payload)
    with pytest.raises(RestoreError, match="sha256"):
        restore(bad, data_dir=tmp_path / "r", ocr_cache_dir=tmp_path / "ro", qdrant=snaps,
                collections={"public": "x1", "private": "x2"}, log=lambda *_: None)
    assert not (tmp_path / "r").exists() and snaps.uploaded == []


def test_backup_refuses_while_jobs_are_running(tmp_path):
    data, ocr = make_data(tmp_path / "src")
    s = SqliteStorage(data)
    s.update_job("job_1", status="ocr")
    s.close()
    kw = dict(data_dir=data, ocr_cache_dir=ocr, qdrant=FakeSnapshots({"p": 1, "q": 1}),
              collections={"public": "p", "private": "q"}, log=lambda *_: None)
    with pytest.raises(SystemExit, match="มีงานกำลังทำ 1"):
        backup(out=tmp_path / "a.zip", **kw)
    assert backup(out=tmp_path / "b.zip", allow_busy=True, **kw)["busy_at_backup"]["jobs_in_progress"] == 1


# ---------- Qdrant จริง ----------


@pytest.mark.qdrant
def test_backup_and_restore_with_real_qdrant_snapshots(tmp_path):
    from qdrant_client import QdrantClient

    from app.config import Settings
    from app.embedding import HashEmbedder
    from app.models import Document
    from app.vectorstore import VectorStore, embed_query
    from scripts.backup import QdrantSnapshots
    from tests.test_retrieve_api import unit

    base = Settings()
    try:
        client = QdrantClient(url=base.qdrant_url, timeout=10, check_compatibility=False)
        client.get_collections()
    except Exception as e:
        pytest.skip(f"ไม่มี Qdrant ที่ {base.qdrant_url}: {e}")
    src_names, dst_names = ("pytest_backup_public", "pytest_backup_private"), ("pytest_restore_public",
                                                                               "pytest_restore_private")
    for n in src_names + dst_names:
        if client.collection_exists(n):
            client.delete_collection(n)
    settings = Settings(collection_public=src_names[0], collection_private=src_names[1], embedding_test_mode=True)
    data = tmp_path / "src"
    storage = SqliteStorage(data)
    acl = AclStore(memberships=storage.memberships, persistence=storage)
    acl.create_box(Box(box_id="pub", visibility=Visibility.PUBLIC), "u1")
    acl.register_file(FileAcl(file_id="f1", box_id="pub"))
    storage.insert_file(FileRecord(file_id="f1", box_id="pub", filename="a.docx", content_hash="h", size_bytes=1,
                                   status="done", submitted_by="u1"))
    store = VectorStore(QdrantClient(url=base.qdrant_url, timeout=30), acl, settings)
    store.ensure_collections()
    texts = ["รายได้รวมของบริษัทเพิ่มขึ้นจากปีก่อน " * 5, "ค่าใช้จ่ายในการบริหารลดลง " * 5, "นโยบายความยั่งยืน " * 5]
    units = [unit(i, t) for i, t in enumerate(texts)]
    doc = Document(file_id="f1", filename="a.pdf", content_hash="h", size_bytes=1, detected_type="pdf", units=units,
                   pipeline_version="t", normalizer_version="t")
    emb = HashEmbedder(settings.vector_size)
    store.upsert_document(doc, emb)
    principal = acl.principal_for("u1", [], 0)
    query = embed_query(texts[0][:20], emb)
    before = [(h.file_id, h.unit_index, h.chunk_index) for h in store.search(query, principal, "pub", limit=3).hits]
    storage.close()
    snaps = QdrantSnapshots(base.qdrant_url, base.qdrant_api_key)
    try:
        out = tmp_path / "bk.zip"
        manifest = backup(data_dir=data, ocr_cache_dir=tmp_path / "no_ocr", qdrant=snaps,
                          collections={"public": src_names[0], "private": src_names[1]}, out=out, log=lambda *_: None)
        assert manifest["qdrant"][src_names[0]]["points"] == len(before) > 0
        assert not client.list_snapshots(src_names[0])  # ไม่ทิ้ง snapshot ค้างบน server
        dest = tmp_path / "dst"
        report = restore(out, data_dir=dest, ocr_cache_dir=dest / "ocr_cache", qdrant=snaps,
                         collections={"public": dst_names[0], "private": dst_names[1]}, log=lambda *_: None)
        assert report["problems"] == [], report["problems"]
        restored_settings = Settings(collection_public=dst_names[0], collection_private=dst_names[1],
                                     embedding_test_mode=True)
        s2 = SqliteStorage(dest)
        acl2 = AclStore(memberships=s2.memberships, persistence=s2)
        store2 = VectorStore(QdrantClient(url=base.qdrant_url, timeout=30), acl2, restored_settings)
        store2.ensure_collections()  # schema/payload index ตรงกับที่โค้ดต้องการ
        after = [(h.file_id, h.unit_index, h.chunk_index)
                 for h in store2.search(query, acl2.principal_for("u1", [], 0), "pub", limit=3).hits]
        assert after == before
        s2.close()
    finally:
        for n in src_names + dst_names:
            if client.collection_exists(n):
                client.delete_collection(n)
