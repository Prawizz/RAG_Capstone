"""ของที่ API ใช้ร่วมกันใน process: ที่เก็บถาวร, AclStore, การรับไฟล์, ingest worker, query embedder, vector store

สร้างครั้งเดียวต่อ app (``app.state.services``): ตอนเริ่ม (lifespan) หรือครั้งแรกที่มี request ต้องใช้ (lazy — test ที่ไม่เปิด
lifespan ได้ services โดยไม่มี worker thread) · worker thread เริ่มใน lifespan เท่านั้น
test ใส่ ``app.state.services`` เอง (OCR ปลอม + hash embedding + Qdrant ในกระบวนการ) ได้ผ่าน ``build_services(...)``

vector store ตัวเดียวต่อ process (``StoreProvider``) ใช้ร่วมกันระหว่าง worker (เขียน) กับ ``/v1/retrieve`` (อ่าน) —
QdrantClient ของจริงใช้ข้าม thread ได้ และ Qdrant ในกระบวนการของ test ต้องเป็นก้อนเดียวกันถึงจะค้นเจอสิ่งที่ worker เขียน
สร้างครั้งแรกที่ใช้ (+ ``ensure_collections``) สร้างไม่สำเร็จ = ไม่เก็บไว้ ลองใหม่รอบหน้า
query embedder (``app/query_embedder.py``) = BGE-M3 บน CPU แยกจาก embedder ของ worker โหลดใน lifespan
ลบไฟล์ (``app/file_deleter.py``): ไฟล์ที่ค้าง deleting ถูก withdraw จาก AclStore ตอนสร้าง services (ก่อน request แรก)
แล้วลบต่อใน lifespan + worker ตอนว่าง
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional

from app.access import AclStore
from app.config import Settings
from app.embedding import Embedder, get_embedder
from app.extraction.ocr import OcrBackend, get_backend
from app.file_deleter import FileDeleter
from app.ingest_service import IngestService
from app.ingest_worker import IngestWorker, unload_ocr_models
from app.query_embedder import QueryEmbedder
from app.reranker import PairScorer, Reranker
from app.storage import SqliteStorage
from app.vectorstore import VectorStore

_build_lock = threading.Lock()


class StoreProvider:
    def __init__(self, factory: Callable[[], VectorStore]) -> None:
        self._factory = factory
        self._store: Optional[VectorStore] = None
        self._lock = threading.Lock()

    def get(self) -> VectorStore:
        """vector store ของ process (สร้าง + ``ensure_collections`` ครั้งแรก) — Qdrant ล่มตอนสร้าง = โยนต่อ ไม่เก็บไว้"""
        with self._lock:
            if self._store is None:
                store = self._factory()
                store.ensure_collections()
                self._store = store
            return self._store


@dataclass
class Services:
    settings: Settings
    storage: SqliteStorage
    acl: AclStore
    ingest: IngestService
    worker: IngestWorker
    stores: StoreProvider
    query_embedder: QueryEmbedder
    deleter: FileDeleter
    reranker: Reranker

    def close(self) -> None:
        self.worker.stop()
        self.query_embedder.close()
        self.reranker.close()
        self.storage.close()


def build_services(
    settings: Settings,
    *,
    vector_store_factory: Optional[Callable[[AclStore], VectorStore]] = None,
    embedder_factory: Optional[Callable[[], Embedder]] = None,
    ocr_backend_factory: Optional[Callable[[], OcrBackend]] = None,
    ocr_unloader: Callable[[], None] = unload_ocr_models,
    query_embedder_factory: Optional[Callable[[], Embedder]] = None,
    reranker_factory: Optional[Callable[[], PairScorer]] = None,
) -> Services:
    storage = SqliteStorage(settings.data_dir)
    acl = AclStore(memberships=storage.memberships, persistence=storage)
    make_store = vector_store_factory or (lambda a: VectorStore.connect(a, settings))
    stores = StoreProvider(lambda: make_store(acl))
    deleter = FileDeleter(storage, acl, stores.get)
    deleter.withdraw_pending()
    worker = IngestWorker(
        storage, acl, settings,
        vector_store_factory=stores.get,
        embedder_factory=embedder_factory or (lambda: get_embedder(settings)),
        ocr_backend_factory=ocr_backend_factory or (lambda: get_backend(settings=settings)),
        ocr_unloader=ocr_unloader,
        idle_hook=deleter.finish_pending,
    )
    ingest = IngestService(storage, acl, notify=worker.notify)
    return Services(settings=settings, storage=storage, acl=acl, ingest=ingest, worker=worker, stores=stores,
                    query_embedder=QueryEmbedder(settings, query_embedder_factory), deleter=deleter,
                    reranker=Reranker(settings, reranker_factory))


def services_of(app, settings: Settings) -> Services:
    svc = getattr(app.state, "services", None)
    if svc is None:
        with _build_lock:
            svc = getattr(app.state, "services", None)
            if svc is None:
                svc = build_services(settings)
                app.state.services = svc
    return svc
