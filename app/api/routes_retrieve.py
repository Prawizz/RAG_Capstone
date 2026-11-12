"""[M7] POST /v1/retrieve — Production Enterprise RAG Retrieval Endpoint

เชื่อมต่อสถาปัตยกรรมระดับโลกแบบ End-to-End:
1. Two-Gate ACL Enforcement (Hard Rule 9):
   - ตรวจสิทธิ์ 2 ด่าน (Pre-filter ใน Qdrant + Post-retrieval Re-check กับ AclStore)
2. Dual-Calendar & Acronym Query Expansion (app/query_expansion.py):
   - ขยายปี พ.ศ. ↔ ค.ศ. และคำย่อทางการเงินเพื่อเพิ่ม Recall@5
3. Salience-Weighted Late-Interaction Reranking (app/rerank.py):
   - ถ่วงน้ำหนักคำแก่นเรื่อง (Core Intent) 3 เท่าเหนือ Corporate Boilerplate
4. Conformal Prediction Abstention Gate:
   - ปฏิเสธการตอบเมื่อหลักฐานไม่เพียงพอ ป้องกัน Hallucination
5. Multi-Evidence Consistency & Conflict Detector (app/consistency.py):
   - ตรวจจับความขัดแย้งเชิงตัวเลข พร้อมแนบคำแนะนำให้ LLM ตอบอย่างรอบคอบ
6. Sentence-level Citation Pack (app/citations.py):
   - ติดแท็กอ้างอิง [S1], [S2] โดยตัดบริบทสังเคราะห์ [แถวแรกของตาราง...] ออกตาม Decision 89
"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status

from app.access import AclStore
from app.api.schemas import (
    CitationItemSchema,
    HitItemSchema,
    RetrieveRequest,
    RetrieveResponse,
)
from app.citations import build_citations_from_hits, format_citation_pack
from app.config import Settings
from app.consistency import detect_evidence_conflicts
from app.embedding import Embedder, get_embedder
from app.query_expansion import expand_query
from app.rerank import ConformalAbstentionGate, FALLBACK_ABSTAIN_MESSAGE, rerank_hits
from app.vectorstore import VectorStore, embed_query

router = APIRouter(prefix="/v1", tags=["Retrieval"])

_global_settings = Settings()
_global_acl_store = AclStore()
_global_embedder: Optional[Embedder] = None
_global_vector_store: Optional[VectorStore] = None


def get_settings() -> Settings:
    return _global_settings


def get_acl_store() -> AclStore:
    return _global_acl_store


def get_embed_service(settings: Settings = Depends(get_settings)) -> Embedder:
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = get_embedder(settings)
    return _global_embedder


def get_vector_store_service(
    acl: AclStore = Depends(get_acl_store),
    settings: Settings = Depends(get_settings),
) -> VectorStore:
    global _global_vector_store
    if _global_vector_store is None:
        try:
            _global_vector_store = VectorStore.connect(acl, settings)
        except Exception:
            _global_vector_store = VectorStore.in_memory(acl, settings)
    return _global_vector_store


@router.post("/retrieve", response_model=RetrieveResponse, summary="สืบค้นข้อมูลพร้อม Citations และ Abstention Gate")
def retrieve_endpoint(
    req: RetrieveRequest,
    store: VectorStore = Depends(get_vector_store_service),
    embedder: Embedder = Depends(get_embed_service),
) -> RetrieveResponse:
    """Endpoint สกัดข้อมูลองค์กรระดับประโยคพร้อมแท็กอ้างอิงและคัดกรองความมั่นใจ"""
    principal = req.principal.to_domain()

    # 1. ตรวจสอบว่าผู้ใช้มีสิทธิ์เข้าถึง box หรือไม่ (Fail-closed ตาม Hard Rule 9)
    box = store.acl.get_box(req.box_id)
    if box is None or req.box_id not in principal.box_ids:
        return RetrieveResponse(
            query=req.query,
            hits=[],
            formatted_context="",
            citations={},
            is_abstain=True,
            abstain_reason="NO_EVIDENCE_FOUND",
            fallback_message=FALLBACK_ABSTAIN_MESSAGE,
        )

    # 2. Dual-Calendar & Acronym Query Expansion
    expanded = expand_query(req.query)
    expanded_terms_list = [t for t in expanded.salient_terms if t not in req.query]

    # ทำ Embedding ของ Query (นำข้อความที่ขยายมาช่วยเสริม Vector representation)
    query_obj = embed_query(expanded.expanded_text, embedder)

    # 3. สืบค้นผ่าน Qdrant Vector Store (มี Pre-filter และ Re-check สิทธิ์ 2 ด่าน)
    search_result = store.search(
        query=query_obj,
        principal=principal,
        box_id=req.box_id,
        limit=req.limit,
        mode=req.search_mode,
    )

    candidate_hits = list(search_result.hits)

    # 4. Salience-Weighted Intent Reranking (ถ้าเปิดใช้งาน)
    if req.enable_rerank and candidate_hits:
        reranked = rerank_hits(req.query, candidate_hits)
        active_hits = [r.hit for r in reranked]
    else:
        from app.rerank import RerankedHit

        reranked = [
            RerankedHit(
                hit=h,
                original_score=h.score,
                rerank_score=h.score,
                lexical_overlap=0.0,
            )
            for h in candidate_hits
        ]
        active_hits = candidate_hits

    # 5. Conformal Abstention Gate (ด่านปฏิเสธการตอบเมื่อหลักฐานไม่พอ)
    if req.enable_abstention:
        gate = ConformalAbstentionGate()
        decision = gate.evaluate(req.query, reranked)
        if decision.should_abstain:
            return RetrieveResponse(
                query=req.query,
                hits=[],
                formatted_context="",
                citations={},
                is_abstain=True,
                abstain_reason=decision.reason,
                fallback_message=decision.fallback_message or FALLBACK_ABSTAIN_MESSAGE,
                expanded_terms=expanded_terms_list,
            )

    # 6. Multi-Evidence Consistency & Conflict Detector
    consistency = detect_evidence_conflicts(active_hits)

    # 7. สร้าง Sentence-level Citations และจัดเตรียม Context สำหรับ LLM
    citations_map = build_citations_from_hits(active_hits)
    pack = format_citation_pack(
        active_hits,
        citations_map,
        consistency_guidance=consistency.guidance_text,
    )

    hit_items = [
        HitItemSchema(
            file_id=h.file_id,
            filename=h.filename,
            unit_index=h.unit_index,
            label=h.label,
            text=h.text,
            score=h.score,
            position=dict(h.position),
        )
        for h in active_hits
    ]

    citation_items = {
        k: CitationItemSchema(
            sentence_id=v.sentence_id,
            doc_id=v.doc_id,
            label=v.label,
            char_start=v.char_start,
            char_end=v.char_end,
            snippet=v.snippet,
            position=v.position,
            score=v.score,
        )
        for k, v in citations_map.items()
    }

    return RetrieveResponse(
        query=req.query,
        hits=hit_items,
        formatted_context=pack.formatted_context,
        citations=citation_items,
        is_abstain=False,
        abstain_reason=None,
        fallback_message=None,
        has_conflict=consistency.has_conflict,
        conflict_guidance=consistency.guidance_text if consistency.has_conflict else None,
        expanded_terms=expanded_terms_list,
    )
