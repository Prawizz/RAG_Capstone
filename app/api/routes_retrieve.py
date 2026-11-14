"""POST /v1/retrieve — ค้นหลักฐานในกล่องเดียว (ไม่ generate คำตอบ)

ลำดับ:
1. ไม่ส่ง ``box_id`` -> ``source_required`` (ไม่ embed ไม่ค้น)
2. normalize คำถามด้วย normalizer กลาง (กฎเหล็กข้อ 1) -> embed ด้วย query embedder (BGE-M3 บน CPU แยกจาก worker)
   — embed ทุกกรณีที่มี box_id รวมกล่องที่ไม่รู้จัก/ไม่เป็นสมาชิก (เส้นทางเดียวกับกล่องที่เป็นสมาชิกให้มากที่สุด)
3. ``VectorStore.search`` ตาม ``SEARCH_MODE`` (ค่าเริ่มต้น dense+sparse_bge รวมด้วย RRF) + ACL 2 ด่าน
   กล่องที่ไม่รู้จัก / ไม่เป็นสมาชิก = ``EMPTY_RESULT`` -> response เหมือนกล่องที่เป็นสมาชิกแต่ไม่มีผลทุกฟิลด์ยกเว้น trace_id
4. **ลำดับ (rank) มาจาก RRF แต่ score = dense cosine** ระหว่างคำถามกับ chunk (เวกเตอร์ dense ดึงมากับผลค้น — ไม่ยิง
   Qdrant รอบสอง) เพราะค่า RRF ขึ้นกับอันดับอย่างเดียว เทียบข้ามคำถามไม่ได้ จึงตั้งเกณฑ์ไม่ได้ · score จึงไม่จำเป็นต้อง
   ลดลงตาม rank · top_score = max(score) · top_score < RETRIEVE_MIN_SCORE -> no_relevant_context (ยังคืน results)
5. Hit -> RetrieveResult: chunk_id ``{file_id}:u{unit:04d}:c{chunk:03d}`` · text = เนื้อ chunk (ไม่รวม context line) ·
   citation ตาม unit_type · extraction.method + flags ของ unit (payload ``flags``)

ส่วนเสริม (ปิดเป็นค่าเริ่มต้น — ``app/config.py``; ไม่เปลี่ยนความหมายของ score: ยังเป็น dense cosine เสมอ):
- ``RERANKER=bge-v2-m3``: ค้น ``RERANK_CANDIDATES`` (20) อันดับแรกจาก RRF -> cross-encoder (``app/reranker.py``) ให้คะแนนคู่
  (คำถามหลัง normalize, context line + เนื้อ chunk) -> เรียงใหม่ -> คืน top_k (rank = ลำดับของ reranker)
- ``SALIENCE_RERANK=1``: เหมือนกันแต่เรียงด้วย ``app/rerank.py::rerank_hits`` (โมดูลของเพื่อน ไม่แก้ตรรกะ — ส่ง Hit ตามที่
  scripts/run_full_pipeline.py ส่ง: score ของ Hit = ค่า RRF)
- ``QUERY_EXPANSION=1``: embed ``app/query_expansion.py::expand_query(...).expanded_text`` (ผ่าน normalizer กลางอีกรอบ)
  แทนคำถามเดิม — score คิดกับเวกเตอร์ของคำถามที่ขยายแล้ว
log เพิ่ม: rerank_ms, rerank_top (คะแนน reranker ของอันดับ 1 — ไว้วัดผล ไม่ส่งออก API), extras (ส่วนเสริมที่เปิด)

error: Qdrant ล่ม / query embedder / reranker ใช้ไม่ได้ -> 503 unavailable (ไม่ตอบ "ไม่พบ" แทน)
log (``app.api.retrieve``): trace_id, box_id, จำนวนผล, top_score, dropped_by_recheck, เวลา embed/search/map —
**ห้าม log ข้อความคำถามหรือข้อความ chunk** (log แค่ความยาว) · ``dropped_by_recheck`` ห้ามส่งออกทาง API
"""

from __future__ import annotations

import logging
import math
import time
from typing import Annotated, Optional, Sequence

from fastapi import APIRouter, Depends, Request

from app.access import Box
from app.api.deps import CurrentPrincipal, ServiceAuth, get_services
from app.api.errors import ApiError, error_docs, trace_id_of
from app.api.schemas import (
    ApiErrorCode,
    Citation,
    Extraction,
    RetrieveRequest,
    RetrieveResponse,
    RetrieveResult,
    RetrieveStatus,
)
from app.api.services import Services
from app.models import ExtractionMethod, QualityFlag, UnitType
from app.query_embedder import QueryEmbedderUnavailable
from app.query_expansion import expand_query
from app.rerank import rerank_hits
from app.reranker import RERANK_CANDIDATES, RerankerUnavailable, rerank_order
from app.thai.normalizer import normalize
from app.vectorstore import EMPTY_RESULT, Hit, SearchResult

log = logging.getLogger("app.api.retrieve")

router = APIRouter(prefix="/v1", tags=["retrieve"], dependencies=[ServiceAuth])

SCORE_DECIMALS = 4
_KNOWN_FLAGS = {f.value for f in QualityFlag}


def chunk_id(file_id: str, unit_index: int, chunk_index: int) -> str:
    return f"{file_id}:u{unit_index:04d}:c{chunk_index:03d}"


def dense_cosine(a: Sequence[float], b: Optional[Sequence[float]]) -> float:
    if b is None or len(a) != len(b):
        raise ValueError("ไม่มีเวกเตอร์ dense ของ chunk (search ต้องเรียกด้วย with_dense=True)")
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def citation_of(hit: Hit) -> Citation:
    unit_type = UnitType(hit.unit_type)
    pos = hit.position or {}
    fields: dict = {}
    if unit_type == UnitType.PAGE:
        fields["page"] = pos.get("page")
    elif unit_type == UnitType.SLIDE:
        fields["slide"] = pos.get("slide")
    elif unit_type == UnitType.SECTION:
        fields["heading_path"] = list(pos.get("heading_path") or [])
    else:  # sheet_region: position ใช้ key "range" (app/models.py) -> cell_range ใน contract
        fields["sheet"] = pos.get("sheet")
        fields["cell_range"] = pos.get("range")
    return Citation(unit_type=unit_type, label=hit.label, char_start=hit.char_start, char_end=hit.char_end, **fields)


def result_of(rank: int, hit: Hit, score: float) -> RetrieveResult:
    flags = [QualityFlag(f) for f in hit.flags if f in _KNOWN_FLAGS]
    return RetrieveResult(
        rank=rank, chunk_id=chunk_id(hit.file_id, hit.unit_index, hit.chunk_index), score=score, text=hit.text,
        file_id=hit.file_id, filename=hit.filename, citation=citation_of(hit),
        extraction=Extraction(method=ExtractionMethod(hit.method), flags=flags),
    )


def extras_of(settings) -> list[str]:
    out = []
    if settings.query_expansion:
        out.append("query_expansion")
    if settings.reranker != "off":
        out.append(f"reranker:{settings.reranker}")
    if settings.salience_rerank:
        out.append("salience_rerank")
    return out


def passage_of(hit: Hit) -> str:
    """ข้อความเดียวกับที่ embed ตอน index (``Chunk.embed_text``)"""
    return f"{hit.context}\n{hit.text}" if hit.context else hit.text


def build_response(trace_id: str, box_id: str, box: Optional[Box], results: list[RetrieveResult],
                   min_score: float) -> RetrieveResponse:
    if not results:  # ไม่มีผล / ไม่มีสิทธิ์ / กล่องไม่รู้จัก: ทุกฟิลด์เหมือนกัน (visibility, top_score = null)
        return RetrieveResponse(status=RetrieveStatus.NO_RELEVANT_CONTEXT, box_id=box_id, visibility=None,
                                top_score=None, results=[], trace_id=trace_id)
    top = max(r.score for r in results)
    status = RetrieveStatus.OK if top >= min_score else RetrieveStatus.NO_RELEVANT_CONTEXT
    return RetrieveResponse(status=status, box_id=box_id, visibility=box.visibility, top_score=top, results=results,
                            trace_id=trace_id)


@router.post("/retrieve", response_model=RetrieveResponse, summary="ค้นหลักฐานในกล่อง (ไม่ generate คำตอบ)",
             responses=error_docs(ApiErrorCode.UNAVAILABLE))
def retrieve(request: Request, principal: CurrentPrincipal, body: RetrieveRequest,
             services: Annotated[Services, Depends(get_services)]) -> RetrieveResponse:
    tid = trace_id_of(request)
    settings = services.settings
    if body.box_id is None:
        log.info("retrieve trace_id=%s box_id=- status=source_required results=0 query_chars=%d",
                 tid, len(body.query))
        return RetrieveResponse(status=RetrieveStatus.SOURCE_REQUIRED, box_id=None, visibility=None, top_score=None,
                                results=[], trace_id=tid)

    reranking = settings.reranker != "off" or settings.salience_rerank
    limit = max(RERANK_CANDIDATES, body.top_k) if reranking else body.top_k
    normalized = normalize(body.query)
    embed_text = normalize(expand_query(body.query).expanded_text) if settings.query_expansion else normalized

    t0 = time.perf_counter()
    try:
        query = services.query_embedder.embed(embed_text)
    except QueryEmbedderUnavailable as e:
        log.warning("retrieve trace_id=%s box_id=%s unavailable=query_embedder (%s)", tid, body.box_id, e)
        raise ApiError(ApiErrorCode.UNAVAILABLE, "query embedder ใช้ไม่ได้ชั่วคราว") from None
    t1 = time.perf_counter()

    box = services.acl.get_box(body.box_id)
    visible = box is not None and body.box_id in principal.box_ids
    result: SearchResult = EMPTY_RESULT
    if visible:  # ไม่เห็นกล่อง = EMPTY_RESULT (VectorStore.search ก็คืนแบบเดียวกัน — ไม่ต้องแตะ Qdrant)
        try:
            store = services.stores.get()
            result = store.search(query, principal, body.box_id, limit=limit, mode=settings.search_mode,
                                  with_dense=True)
        except Exception as e:  # Qdrant ล่ม/ต่อไม่ได้ — ไม่ตอบ "ไม่พบ" แทน
            log.warning("retrieve trace_id=%s box_id=%s unavailable=vector_store (%s)", tid, body.box_id,
                        type(e).__name__)
            raise ApiError(ApiErrorCode.UNAVAILABLE, "vector store ใช้ไม่ได้ชั่วคราว") from None
    t2 = time.perf_counter()

    hits = list(result.hits)
    rerank_top: Optional[float] = None
    if hits and settings.reranker != "off":
        try:
            scores = services.reranker.score(normalized, [passage_of(h) for h in hits])
        except RerankerUnavailable as e:
            log.warning("retrieve trace_id=%s box_id=%s unavailable=reranker (%s)", tid, body.box_id, e)
            raise ApiError(ApiErrorCode.UNAVAILABLE, "reranker ใช้ไม่ได้ชั่วคราว") from None
        order = rerank_order(scores)[:body.top_k]
        hits = [hits[i] for i in order]
        rerank_top = scores[order[0]]
    elif hits and settings.salience_rerank:
        reranked = rerank_hits(body.query, hits)[:body.top_k]
        hits = [r.hit for r in reranked]
        rerank_top = reranked[0].rerank_score
    else:
        hits = hits[:body.top_k]
    t3 = time.perf_counter()

    results = [result_of(rank, hit, round(dense_cosine(query.dense, hit.dense), SCORE_DECIMALS))
               for rank, hit in enumerate(hits, start=1)]
    response = build_response(tid, body.box_id, box, results, settings.retrieve_min_score)
    t4 = time.perf_counter()
    log.info(
        "retrieve trace_id=%s box_id=%s status=%s results=%d top_score=%s dropped_by_recheck=%d top_k=%d "
        "query_chars=%d text_chars=%d embed_ms=%.1f search_ms=%.1f map_ms=%.1f rerank_ms=%.1f rerank_top=%s extras=%s",
        tid, body.box_id, response.status.value, len(results),
        "-" if response.top_score is None else f"{response.top_score:.4f}", result.dropped_by_recheck, body.top_k,
        len(body.query), sum(len(r.text) for r in results), (t1 - t0) * 1000, (t2 - t1) * 1000, (t4 - t3) * 1000,
        (t3 - t2) * 1000, "-" if rerank_top is None else f"{rerank_top:.4f}", ",".join(extras_of(settings)) or "-",
    )
    return response
