"""Fabrica da aplicacao. Uso (raiz do repo): ``uvicorn backend.app.main:create_app --factory``."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .adapters.catalog import DatasetCatalog
from .adapters.run_store import FileRunRepository
from .api.deps import Container
from .api.errors import install_error_handlers
from .api.routers import assistant, atlas, datasets, health, methods, runs
from .config import Settings
from .services.assistant import AssistantService
from .services.atlas import AtlasService
from .services.datasets import DatasetService
from .services.jobs import JobRunner, ThreadPoolJobRunner
from .services.pipeline import PipelineFn, run_pipeline
from .services.kinds import build_kinds
from .services.runs import KindSpec, RunService

API_PREFIX = "/api/v1"


def _make_streams_unicode_safe() -> None:
    """O nucleo imprime avisos com simbolos Unicode (ex.: aviso de premissas); em consoles
    cp1252 (Windows) isso levantaria UnicodeEncodeError dentro da execucao."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def _core_method_weights() -> dict[str, float]:
    from causal_discovery import get_registered_method_weights

    return get_registered_method_weights()


def create_app(
    settings: Settings | None = None,
    *,
    pipeline: PipelineFn = run_pipeline,
    kind_overrides: dict[str, KindSpec] | None = None,
    jobs: JobRunner | None = None,
    method_weights: dict[str, float] | None = None,
) -> FastAPI:
    """Dependencias injetaveis (``pipeline``, ``jobs``) permitem testar sem o nucleo pesado."""
    _make_streams_unicode_safe()
    settings = settings or Settings.from_env()
    # O nucleo vive na raiz do repositorio; garante o import quando executado de outro cwd.
    if str(settings.repo_root) not in sys.path:
        sys.path.insert(0, str(settings.repo_root))

    weights = method_weights if method_weights is not None else _core_method_weights()
    job_runner = jobs or ThreadPoolJobRunner(settings.max_workers)
    catalog = DatasetCatalog(settings.repo_root, settings.uploads_dir)
    dataset_service = DatasetService(catalog, settings.max_upload_bytes)
    atlas_service = AtlasService(settings.repo_root, settings.data_dir / "atlas_history.jsonl")
    kinds = build_kinds(dataset_service, atlas_service, pipeline, available_methods=list(weights))
    kinds.update(kind_overrides or {})  # testes substituem so os tipos pesados
    run_service = RunService(FileRunRepository(settings.runs_dir), job_runner, kinds)
    assistant_service = AssistantService(dataset_service, run_service, atlas_service.cards, settings.ollama_url)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        job_runner.shutdown()

    app = FastAPI(title="Causal Discovery TS API", version="1.0.0", lifespan=lifespan)
    app.state.container = Container(
        datasets=dataset_service,
        runs=run_service,
        atlas=atlas_service,
        assistant=assistant_service,
        method_weights=weights,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Location"],
    )
    install_error_handlers(app)

    api = APIRouter(prefix=API_PREFIX)
    for module in (health, methods, datasets, atlas, runs, assistant):
        api.include_router(module.router)
    app.include_router(api)

    dist = settings.frontend_dist
    if dist is not None:  # producao: um unico deploy serve API + SPA
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Rota inexistente.")
            candidate = (dist / path).resolve()
            if path and candidate.is_file() and dist.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app

