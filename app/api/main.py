"""[M7] FastAPI app entrypoint for Enterprise RAG API Service"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_health import router as health_router
from app.api.routes_retrieve import router as retrieve_router


def create_app() -> FastAPI:
    """สร้างและกำหนดค่า FastAPI Application"""
    app = FastAPI(
        title="Thai Enterprise RAG API Service",
        description="Production-grade Domain-Agnostic RAG Service with Two-Gate ACL, ALCE Citations, and Abstention Gate",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(retrieve_router)

    return app


app = create_app()
