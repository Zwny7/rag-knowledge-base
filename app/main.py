import logging
import time
from contextlib import asynccontextmanager
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pypdf.errors import PyPdfError
from docx.opc.exceptions import PackageNotFoundError
from pydantic import BaseModel, ConfigDict, Field

from . import settings
from .model import ModelError
from .rag_service import KnowledgeBase, VectorUnavailable

logger = logging.getLogger("uvicorn.error")


class Input(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class Question(Input):
    question: str = Field(min_length=2, max_length=500)
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid"
    top_k: int = Field(default=4, ge=1, le=8)
    use_model: bool = False
    document: str | None = Field(default=None, min_length=1, max_length=255)


class Source(BaseModel):
    id: int
    document: str
    heading: str
    page: int | None
    text: str
    score: float


class Answer(BaseModel):
    answer: str
    mode: Literal["hybrid", "vector", "bm25"]
    answer_type: Literal["model", "excerpt"]
    citations: list[int]
    sources: list[Source]
    index_revision: str


def create_app(data=settings.DATA, docs=settings.DOCUMENTS, *, with_vectors=True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        data.mkdir(parents=True, exist_ok=True)
        app.state.kb = KnowledgeBase(docs, data)
        await run_in_threadpool(app.state.kb.rebuild, with_vectors)
        yield

    app = FastAPI(title="RAG知识库问答", version="1.0", lifespan=lifespan)
    web = settings.ROOT / "web"
    app.mount("/static", StaticFiles(directory=web), name="static")

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        rid = uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        logger.info("request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
                    rid, request.method, request.url.path, response.status_code,
                    (time.perf_counter() - started) * 1000)
        return response

    @app.exception_handler(ModelError)
    async def model_error(_request, error):
        return JSONResponse(status_code=502, content={"detail": str(error)})

    @app.exception_handler(VectorUnavailable)
    async def vector_error(_request, error):
        return JSONResponse(status_code=503, content={"detail": str(error)})

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(web / "index.html")

    @app.get("/api/status")
    def status():
        return app.state.kb.status() | {"model_ready": settings.MODEL_READY}

    @app.post("/api/knowledge/ask", response_model=Answer)
    def ask(payload: Question):
        try:
            return app.state.kb.ask(payload.question, payload.mode, payload.top_k,
                                    payload.use_model, payload.document)
        except LookupError as exc:
            raise HTTPException(404, "所选文档已不在当前索引中，请刷新文档列表。") from exc

    @app.post("/api/knowledge/reindex")
    def reindex():
        try:
            return app.state.kb.rebuild(with_vectors)
        except (ValueError, OSError, PyPdfError, PackageNotFoundError) as exc:
            logger.warning("reindex_failed error_type=%s", type(exc).__name__)
            raise HTTPException(400, "文档读取失败或目录没有有效内容，已保留上次索引。") from exc

    return app


app = create_app()
