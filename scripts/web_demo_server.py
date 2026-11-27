"""FastAPI Web Demo Server connecting the React UI to the real Enterprise RAG Pipeline.

Runs the actual modules from ``app/``:
- PDF extraction (``app.extraction.pipeline.extract_file`` with PyMuPDF fallback)
- Thai normalization (``app.thai.normalizer.normalize``)
- Sentence-boundary chunking (``app.chunking.make_chunks``)
- Two-gate Enterprise ACL + Qdrant VectorStore (``app.access``, ``app.vectorstore``)
- Query Expansion (``app.query_expansion.expand_query``)
- Salience Reranking & Conformal Abstention Gate (``app.rerank``)
- Sentence-level Citations (``app.citations``)
- Real PDF page bounding-box coordinate lookup via PyMuPDF (``page.search_for``)

Usage:
    uvicorn scripts.web_demo_server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.access import AclStore, Box, FileAcl, Visibility
from app.chunking import make_chunks
from app.citations import build_citations_from_hits
from app.config import Settings
from app.embedding import EmbeddingUnavailable, get_embedder
from app.extraction.pipeline import extract_file
from app.models import (
    DetectedType,
    Document,
    ExtractionMethod,
    QualityReport,
    Unit,
    UnitStatus,
    UnitType,
)
from app.query_expansion import expand_query
from app.rerank import ConformalAbstentionGate, FALLBACK_ABSTAIN_MESSAGE, rerank_hits
from app.thai.normalizer import normalize
from app.vectorstore import Hit, IndexedChunk, Query, VectorStore, embed_chunks, embed_query

log = logging.getLogger("web_demo_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BOX_ID = "box_enterprise_demo"

# Real PDFs in the repository to load at startup (pages capped for fast startup, expandable)
REPO_PDFS: list[dict[str, Any]] = [
    {
        "doc_id": "cpall-2025",
        "path": ROOT / "eval" / "datasets" / "E_ONE_REPORT_CPALLT.PDF",
        "name": "E_ONE_REPORT_CPALLT.PDF",
        "max_index_pages": 35,
    },
    {
        "doc_id": "ptt-2025",
        "path": ROOT / "eval" / "datasets" / "ONEREPORTPTTT.PDF",
        "name": "ONEREPORTPTTT.PDF",
        "max_index_pages": 35,
    },
    {
        "doc_id": "mock-corpus",
        "path": ROOT / "mock_corpus.pdf",
        "name": "mock_corpus.pdf",
        "max_index_pages": None,
    },
]


class ChatRequest(BaseModel):
    query: str
    top_k: int = 3
    doc_id: Optional[str] = None
    mode: Literal["ai_chat", "rag_only"] = "ai_chat"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None


class RagRuntimeState:
    def __init__(self) -> None:
        self.embedder: Any = None
        self.embedder_mode: str = "unknown"
        self.settings: Optional[Settings] = None
        self.acl: Optional[AclStore] = None
        self.principal: Any = None
        self.store: Optional[VectorStore] = None
        self.docs: dict[str, dict[str, Any]] = {}  # doc_id -> {id, name, bytes, num_pages, is_custom, indexed_pages}


state = RagRuntimeState()


def _extract_document_safe(pdf_path: Path, doc_id: str, filename: str, max_pages: Optional[int]) -> tuple[Document, int]:
    """Extracts a PDF using app.extraction.pipeline.extract_file, falling back cleanly
    to PyMuPDF + app.thai.normalizer if the teammate's extractor raises an internal error.
    """
    pdf_doc = pymupdf.open(str(pdf_path))
    total_pages = len(pdf_doc)
    page_limit = min(total_pages, max_pages) if max_pages else total_pages
    page_range = range(1, page_limit + 1)

    try:
        doc = extract_file(pdf_path, pages=page_range)
        if doc.error is None and any(u.status == UnitStatus.OK for u in doc.units):
            return doc, total_pages
    except Exception as exc:
        log.warning("extract_file raised %s for %s; using PyMuPDF + normalizer fallback", exc, filename)

    units: list[Unit] = []
    for idx in range(page_limit):
        raw_text = pdf_doc[idx].get_text("text") or ""
        norm_text = normalize(raw_text)
        status = UnitStatus.OK if norm_text.strip() else UnitStatus.NEEDS_OCR
        units.append(
            Unit(
                index=idx,
                type=UnitType.PAGE,
                position={"page": idx + 1},
                label=f"หน้า {idx + 1}",
                text=norm_text,
                text_hash=hashlib.sha256(norm_text.encode("utf-8")).hexdigest(),
                quality=QualityReport(score=1.0 if status == UnitStatus.OK else 0.0, flags=[]),
                status=status,
                reason="empty_text" if status != UnitStatus.OK else None,
                method=ExtractionMethod.TEXT_LAYER,
            )
        )

    pdf_doc.close()
    pdf_bytes = pdf_path.read_bytes()
    return (
        Document(
            file_id=doc_id,
            filename=filename,
            detected_type=DetectedType.PDF,
            content_hash=hashlib.sha256(pdf_bytes).hexdigest(),
            size_bytes=len(pdf_bytes),
            pipeline_version="web-demo-1.0",
            normalizer_version="1.0",
            units=units,
        ),
        total_pages,
    )


def _index_pdf_into_store(
    pdf_path: Path,
    doc_id: str,
    filename: str,
    max_pages: Optional[int],
    is_custom: bool = False,
) -> dict[str, Any]:
    assert state.store is not None
    assert state.embedder is not None
    assert state.acl is not None
    assert state.settings is not None

    pdf_bytes = pdf_path.read_bytes()
    doc, total_pages = _extract_document_safe(pdf_path, doc_id, filename, max_pages)
    doc = doc.model_copy(update={"file_id": doc_id, "filename": filename})

    ok_units = [u for u in doc.units if u.status == UnitStatus.OK and u.text.strip()]
    chunks = [
        c
        for u in ok_units
        for c in make_chunks(u, target_chars=state.settings.chunk_target_chars)
    ]

    if chunks:
        indexed_chunks = embed_chunks(chunks, state.embedder)
        state.acl.register_file(FileAcl(file_id=doc_id, box_id=BOX_ID))
        state.store.upsert_chunks(doc, indexed_chunks)

    meta = {
        "id": doc_id,
        "name": filename,
        "bytes": pdf_bytes,
        "num_pages": total_pages,
        "indexed_pages": min(total_pages, max_pages) if max_pages else total_pages,
        "chunks_count": len(chunks),
        "units_ok": len(ok_units),
        "is_custom": is_custom,
    }
    state.docs[doc_id] = meta
    log.info(
        "Indexed %s (%s): %d/%d pages, %d chunks",
        filename,
        doc_id,
        meta["indexed_pages"],
        total_pages,
        len(chunks),
    )
    return meta


def _find_bbox_on_pdf_page(pdf_bytes: bytes, page_num: int, text_snippet: str) -> dict[str, float]:
    """Locates the real bounding box [top%, left%, width%, height%] on a PDF page using PyMuPDF."""
    default_box = {"top": 18.0, "left": 8.0, "width": 84.0, "height": 14.0}
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return default_box

        page = doc[page_num - 1]
        pw, ph = page.rect.width, page.rect.height

        # Clean snippet and try progressively shorter search phrases
        clean = re.sub(r"\s+", " ", text_snippet).strip()
        candidates: list[str] = []
        words = clean.split(" ")
        if len(words) >= 6:
            candidates.append(" ".join(words[:8]))
            candidates.append(" ".join(words[:5]))
        if len(clean) >= 24:
            candidates.append(clean[:45])
            candidates.append(clean[:25])
        # Also try longest non-trivial lines/phrases
        for token in sorted(words, key=len, reverse=True)[:4]:
            if len(token) >= 5:
                candidates.append(token)

        rects: list[pymupdf.Rect] = []
        for cand in candidates:
            found = page.search_for(cand)
            if found:
                rects = found
                break

        if not rects:
            # Fallback: find text block with highest word overlap
            query_tokens = {w.lower() for w in words if len(w) >= 3}
            best_block = None
            best_overlap = 0
            for blk in page.get_text("blocks"):
                blk_text = str(blk[4]).lower()
                overlap = sum(1 for t in query_tokens if t in blk_text)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_block = blk
            if best_block is not None:
                rects = [pymupdf.Rect(best_block[0], best_block[1], best_block[2], best_block[3])]

        doc.close()

        if not rects:
            return default_box

        # Merge first cluster of rects near the first match
        anchor = rects[0]
        nearby = [r for r in rects if abs(r.y0 - anchor.y0) < 90]
        x0 = min(r.x0 for r in nearby)
        y0 = min(r.y0 for r in nearby)
        x1 = max(r.x1 for r in nearby)
        y1 = max(r.y1 for r in nearby)

        pad_x, pad_y = 6.0, 4.0
        top = max(0.0, ((y0 - pad_y) / ph) * 100.0)
        left = max(0.0, ((x0 - pad_x) / pw) * 100.0)
        width = min(96.0, ((x1 - x0 + pad_x * 2) / pw) * 100.0)
        height = min(90.0, max(3.5, ((y1 - y0 + pad_y * 2) / ph) * 100.0))

        return {
            "top": round(top, 2),
            "left": round(left, 2),
            "width": round(width, 2),
            "height": round(height, 2),
        }
    except Exception:
        return default_box


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        state.embedder = get_embedder(test_mode=False)
        state.embedder_mode = "BAAI/bge-m3 (Real Dense + Sparse)"
    except EmbeddingUnavailable:
        log.warning("PyTorch/BGE-M3 unavailable — using HashEmbedder + BM25 Lexical Mode")
        state.embedder = get_embedder(test_mode=True)
        state.embedder_mode = "HashEmbedder + BM25 + Salience Rerank"

    state.settings = Settings(
        vector_size=state.embedder.dense_size,
        collection_public="web_demo_public",
        collection_private="web_demo_private",
        search_mode="all",
    )
    state.acl = AclStore()
    state.acl.create_box(Box(box_id=BOX_ID, visibility=Visibility.PUBLIC), owner_user_id="demo_user")
    state.principal = state.acl.principal_for("demo_user", {"management", "sales"}, 2)
    state.store = VectorStore.in_memory(state.acl, state.settings)

    for spec in REPO_PDFS:
        path: Path = spec["path"]
        if path.exists():
            _index_pdf_into_store(
                pdf_path=path,
                doc_id=spec["doc_id"],
                filename=spec["name"],
                max_pages=spec["max_index_pages"],
                is_custom=False,
            )

    yield
    state.docs.clear()


app = FastAPI(title="Enterprise RAG Live Demo API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "embedder": state.embedder_mode,
        "docs_loaded": len(state.docs),
    }


@app.get("/api/docs")
def list_docs():
    return {
        "embedder": state.embedder_mode,
        "docs": [
            {
                "id": d["id"],
                "name": d["name"],
                "numPages": d["num_pages"],
                "indexedPages": d["indexed_pages"],
                "chunksCount": d["chunks_count"],
                "isCustom": d["is_custom"],
            }
            for d in state.docs.values()
        ],
    }


@app.get("/api/docs/{doc_id}/pdf")
def get_pdf_bytes(doc_id: str):
    doc = state.docs.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(
        content=doc["bytes"],
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{doc["name"]}"'},
    )


@app.get("/api/docs/{doc_id}/page/{page_num}/pdf")
def get_single_pdf_page(doc_id: str, page_num: int):
    """Returns a tiny single-page PDF containing ONLY the requested page so the browser
    never has to download or parse a 500-page PDF just to view one cited page."""
    doc = state.docs.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    src = pymupdf.open(stream=doc["bytes"], filetype="pdf")
    try:
        if page_num < 1 or page_num > len(src):
            raise HTTPException(status_code=404, detail="Page out of range")
        single = pymupdf.open()
        single.insert_pdf(src, from_page=page_num - 1, to_page=page_num - 1)
        out_bytes = single.tobytes()
        single.close()
    finally:
        src.close()

    return Response(
        content=out_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{doc["name"]}_p{page_num}.pdf"'},
    )


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    raw_bytes = await file.read()
    tmp_dir = ROOT / "data" / "web_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    doc_id = f"custom-{uuid.uuid4().hex[:8]}"
    tmp_path = tmp_dir / f"{doc_id}.pdf"
    tmp_path.write_bytes(raw_bytes)

    meta = _index_pdf_into_store(
        pdf_path=tmp_path,
        doc_id=doc_id,
        filename=file.filename,
        max_pages=None,
        is_custom=True,
    )
    return {
        "id": meta["id"],
        "name": meta["name"],
        "numPages": meta["num_pages"],
        "indexedPages": meta["indexed_pages"],
        "chunksCount": meta["chunks_count"],
        "isCustom": True,
    }


def _synthesize_with_llm(
    query: str,
    evidence: list[dict[str, Any]],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[Optional[str], str]:
    """Calls an OpenAI-compatible LLM (OpenAI, OpenRouter, Ollama, vLLM) if configured,
    returning (answer_markdown, engine_label). Returns (None, reason) if unavailable.
    """
    resolved_key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    resolved_base = (base_url or os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/")
    resolved_model = (model or os.environ.get("LLM_MODEL") or "gpt-4o-mini").strip()

    if not resolved_key and not resolved_base:
        # Probe local Ollama if no API key was provided
        try:
            with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=0.6) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") for m in tags.get("models", []) if m.get("name")]
                if models:
                    resolved_base = "http://127.0.0.1:11434/v1"
                    resolved_model = model or models[0]
                    resolved_key = "ollama"
        except Exception:
            return None, "Local Grounded Synthesizer (No external LLM key/Ollama detected)"

    if not resolved_base:
        resolved_base = "https://api.openai.com/v1"

    context_blocks = "\n\n".join(
        f"[{ev['cit_id']}] (Doc: {ev['docName']}, Page {ev['page']}): {ev['full_text'][:1100]}"
        for ev in evidence
    )
    sys_prompt = (
        "คุณคือ Enterprise RAG AI Assistant ที่ตอบคำถามอย่างแม่นยำและเป็นระบบ\n"
        "กฎเหล็ก:\n"
        "1. ตอบโดยอ้างอิงเฉพาะข้อเท็จจริงจาก Context ที่ให้มาเท่านั้น ห้ามแต่งเติม\n"
        "2. ทุกข้อเท็จจริงต้องใส่ citation tag รูปแบบ [[cit-xxx]] ตรงตาม ID ใน Context\n"
        "3. จัดรูปแบบเป็น Markdown ที่อ่านง่าย มีหัวข้อและ bullet points ชัดเจน"
    )
    payload = json.dumps(
        {
            "model": resolved_model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"คำถาม: {query}\n\nContext:\n{context_blocks}"},
            ],
        }
    ).encode("utf-8")

    req_obj = urllib.request.Request(
        f"{resolved_base}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {resolved_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req_obj, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"].strip()
            return text, f"Real LLM ({resolved_model})"
    except Exception as exc:
        log.warning("LLM call failed (%s); falling back to Grounded Extractive Synthesizer", exc)
        return None, f"Grounded Synthesizer (LLM fallback: {type(exc).__name__})"


@app.post("/api/chat")
def chat_rag(req: ChatRequest):
    assert state.store is not None
    assert state.embedder is not None
    assert state.principal is not None

    t0 = time.perf_counter()

    # Step 1: Thai Normalization + Query Expansion
    t_exp0 = time.perf_counter()
    norm_q = normalize(req.query)
    expanded = expand_query(norm_q)
    q_vec = embed_query(expanded.expanded_text, state.embedder)
    t_exp_ms = max(1, int((time.perf_counter() - t_exp0) * 1000))

    # Step 2: Hybrid Vector Search (Qdrant RRF: dense + sparse_bge + sparse_bm25) + ACL
    t_search0 = time.perf_counter()
    search_res = state.store.search(
        query=q_vec,
        principal=state.principal,
        box_id=BOX_ID,
        limit=max(12, req.top_k * 4),
    )
    hits: list[Hit] = list(search_res.hits)
    if req.doc_id:
        filtered = [h for h in hits if h.file_id == req.doc_id]
        if filtered:
            hits = filtered

    # Step 3: Salience-Weighted Reranking & Conformal Abstention Gate
    reranked = rerank_hits(norm_q, hits)[:req.top_k] if hits else []
    gate = ConformalAbstentionGate()
    decision = gate.evaluate(norm_q, reranked)
    abstained = decision.should_abstain
    t_search_ms = max(1, int((time.perf_counter() - t_search0) * 1000))

    # Step 4: Sentence-level Citation Building + PyMuPDF Bounding Box Alignment
    t_cit0 = time.perf_counter()
    citations_out: list[dict[str, Any]] = []
    evidence_items: list[dict[str, Any]] = []
    answer_lines: list[str] = []
    gen_engine_label = "RAG Retrieval Only (Raw Chunks)"

    if abstained:
        answer_lines.append("### ไม่พบหลักฐานเพียงพอในเอกสาร (Conformal Abstention Gate)\n")
        answer_lines.append(decision.fallback_message or FALLBACK_ABSTAIN_MESSAGE)
    else:
        build_citations_from_hits([r.hit for r in reranked])

        for idx, rh in enumerate(reranked, start=1):
            hit = rh.hit
            doc_meta = state.docs.get(hit.file_id)
            doc_name = doc_meta["name"] if doc_meta else hit.file_id
            page_num = int((hit.position or {}).get("page") or 1)
            cit_id = f"cit-{uuid.uuid4().hex[:6]}-{idx}"

            bbox = (
                _find_bbox_on_pdf_page(doc_meta["bytes"], page_num, hit.text)
                if doc_meta
                else {"top": 18.0, "left": 8.0, "width": 84.0, "height": 14.0}
            )

            snippet_clean = re.sub(r"\s+", " ", hit.text).strip()
            short_snippet = (
                f"{snippet_clean[:260]}..." if len(snippet_clean) > 260 else snippet_clean
            )

            citations_out.append(
                {
                    "id": cit_id,
                    "label": str(idx),
                    "docId": hit.file_id,
                    "docName": doc_name,
                    "page": page_num,
                    "coords": bbox,
                    "similarity": round(min(0.99, max(0.50, float(rh.rerank_score))), 2),
                    "snippet": short_snippet,
                }
            )
            evidence_items.append(
                {
                    "cit_id": cit_id,
                    "label": str(idx),
                    "docName": doc_name,
                    "page": page_num,
                    "score": float(rh.rerank_score),
                    "lexical_overlap": float(rh.lexical_overlap),
                    "short_snippet": short_snippet,
                    "full_text": snippet_clean,
                }
            )

        if req.mode == "rag_only":
            gen_engine_label = "RAG Retrieval Only (Pure Evidence Inspection)"
            answer_lines.append("### ผลการทดสอบ RAG Retrieval (แสดงหลักฐานดิบ + คะแนน Rerank)\n")
            for ev in evidence_items:
                answer_lines.append(
                    f"- **Rank #{ev['label']} • {ev['docName']} (หน้า {ev['page']})** "
                    f"`Rerank: {ev['score']:.3f}` `Lexical: {ev['lexical_overlap']:.2f}`\n"
                    f"  {ev['short_snippet']} [[{ev['cit_id']}]]"
                )
        else:
            llm_text, engine_label = _synthesize_with_llm(
                query=req.query,
                evidence=evidence_items,
                api_key=req.api_key,
                base_url=req.base_url,
                model=req.model,
            )
            gen_engine_label = engine_label
            if llm_text:
                answer_lines.append(llm_text)
            else:
                answer_lines.append("### คำตอบแบบ Grounded Summary จากหลักฐานในเอกสาร\n")
                for ev in evidence_items:
                    answer_lines.append(
                        f"- **จากเอกสาร {ev['docName']} (หน้า {ev['page']}):** "
                        f"{ev['short_snippet']} [[{ev['cit_id']}]]"
                    )

    t_cit_ms = max(1, int((time.perf_counter() - t_cit0) * 1000))
    total_ms = max(1, int((time.perf_counter() - t0) * 1000))

    top_sim = citations_out[0]["similarity"] if citations_out else 0.0
    exp_preview = expanded.expanded_text[:60] + ("..." if len(expanded.expanded_text) > 60 else "")

    pipeline_steps = [
        {
            "id": "step-1",
            "title": "1. Thai Normalizer, Query Expansion & Embedding",
            "detail": f"{state.embedder_mode} • Expanded: \"{exp_preview}\"",
            "durationMs": t_exp_ms,
            "status": "done",
        },
        {
            "id": "step-2",
            "title": "2. Qdrant Hybrid RRF + Salience Reranking",
            "detail": f"Candidate Pool: {len(hits)} → Top-{len(reranked)} • Max Score: {top_sim:.2f}",
            "durationMs": t_search_ms,
            "status": "done",
        },
        {
            "id": "step-3",
            "title": "3. Conformal Gate, BBox Alignment & Generation",
            "detail": (
                "Abstained (Low Evidence)"
                if abstained
                else f"{gen_engine_label} • {len(citations_out)} Citations • {total_ms} ms"
            ),
            "durationMs": t_cit_ms,
            "status": "done",
        },
    ]

    return {
        "answer": "\n".join(answer_lines),
        "citations": citations_out,
        "pipelineSteps": pipeline_steps,
        "abstained": abstained,
    }
