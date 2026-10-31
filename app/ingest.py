"""[M4] Ingestion Pipeline: File -> Units -> Normalization -> Chunking -> Indexing

หน้าที่:
- จัดการกระบวนการนำเข้าเอกสาร (Ingestion) แบบครบวงจร
- สกัดข้อความจากไฟล์ (PDF, DOCX, PPTX, XLSX)
- กรองและตรวจสอบคุณภาพรายหน้า (Unit Quality Gate)
- ทำ Hierarchical Chunking (Child + Parent) ผ่าน app/chunking.py
- ส่งต่อ Child Chunks เข้าสู่ Hybrid Index (BM25 + Dense BGE-M3) ผ่าน app/retrieval.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from app.chunking import ChildChunk, ParentChunk, chunk_document
from app.extraction.pipeline import extract_file
from app.models import Document, DocumentStatus, Unit, UnitStatus
from app.retrieval import HybridRetriever


@dataclass
class IngestionReport:
    """รายงานสรุปผลการ Ingestion เอกสารหนึ่งไฟล์"""
    file_id: str
    filename: str
    document_status: DocumentStatus
    total_units: int
    ok_units: int
    skipped_units: int
    total_children: int
    total_parents: int
    extraction_ms: float
    chunking_ms: float
    indexing_ms: float
    total_ms: float


class IngestionService:
    """Service ควบคุมขั้นตอน Ingestion ครบวงจรสำหรับ Production RAG"""

    def __init__(self, retriever: Optional[HybridRetriever] = None):
        self.retriever = retriever or HybridRetriever()
        self.documents: dict[str, Document] = {}
        self.all_children: list[ChildChunk] = []
        self.all_parents: list[ParentChunk] = []

    def ingest_file(
        self,
        file_path: str | Path,
        pages: Optional[tuple[int, int] | list[int] | Sequence[int]] = None,
    ) -> IngestionReport:
        """ประมวลผลไฟล์ตั้งแต่ต้นจนจบ: Extract -> Chunk -> Index"""
        path = Path(file_path)
        start_time = time.time()

        # ขั้นตอนที่ 1: สกัดข้อความและตรวจสอบคุณภาพราย Unit
        t0 = time.time()
        collected_units: list[Unit] = []

        if isinstance(pages, (list, tuple)) and len(pages) > 2 and isinstance(pages[0], int):
            # รายการหน้าแบบระบุเจาะจง (เช่น [5, 6, 7, 8, 144])
            last_doc = None
            for p in pages:
                sub_doc = extract_file(path, pages=(p, p))
                last_doc = sub_doc
                collected_units.extend(sub_doc.units)
            doc = last_doc or extract_file(path)
            doc.units = collected_units
        else:
            # ช่วงหน้า (start, end) หรือทั้งไฟล์
            page_range = pages if isinstance(pages, tuple) and len(pages) == 2 else None
            doc = extract_file(path, pages=page_range)
            collected_units = doc.units

        self.documents[doc.file_id] = doc
        extraction_ms = (time.time() - t0) * 1000

        # นับสถิติ Unit
        ok_units = [u for u in collected_units if u.status == UnitStatus.OK]
        skipped_count = len(collected_units) - len(ok_units)

        # ขั้นตอนที่ 2: Thai-Aware Hierarchical Chunking
        t0 = time.time()
        children, parents = chunk_document(ok_units, document_id=path.name)
        self.all_children.extend(children)
        self.all_parents.extend(parents)
        chunking_ms = (time.time() - t0) * 1000

        # ขั้นตอนที่ 3: Indexing เข้าสู่ Hybrid Search (BM25 + BGE-M3 Dense)
        t0 = time.time()
        self.retriever.index(self.all_children, self.all_parents)
        indexing_ms = (time.time() - t0) * 1000

        total_ms = (time.time() - start_time) * 1000

        return IngestionReport(
            file_id=doc.file_id,
            filename=doc.filename,
            document_status=doc.status,
            total_units=len(collected_units),
            ok_units=len(ok_units),
            skipped_units=skipped_count,
            total_children=len(children),
            total_parents=len(parents),
            extraction_ms=extraction_ms,
            chunking_ms=chunking_ms,
            indexing_ms=indexing_ms,
            total_ms=total_ms,
        )

    def query(
        self,
        question: str,
        top_k: int = 5,
        fusion_method: str = "rrf",
    ) -> dict:
        """ค้นหาและแพ็กเกจคำตอบสำหรับ AI Gateway แบบครบวงจร"""
        results, decision = self.retriever.retrieve(
            question, top_k=top_k, fusion_method=fusion_method
        )

        if decision.should_abstain:
            return {
                "status": "abstained",
                "tier_failed": decision.tier_failed,
                "confidence_score": decision.score,
                "threshold": decision.threshold,
                "fallback_message": decision.message,
                "payload": None,
            }

        payload = self.retriever.prepare_llm_payload(results)
        return {
            "status": "success",
            "tier_failed": None,
            "confidence_score": results[0].reranker_score if results else 0.0,
            "threshold": self.retriever.abstention_gate.conformal_threshold,
            "fallback_message": None,
            "payload": payload,
        }
