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
from .api.routers import datasets, health, methods, runs
from .config import Settings
from .services.datasets import DatasetService
from .services.jobs import JobRunner, ThreadPoolJobRunner
from .services.pipeline import PipelineFn, run_pipeline
from .services.runs import RunService

API_PREFIX = "/api/v1"


def _core_method_weights() -> dict[str, float]:
    from causal_discovery import get_registered_method_weights

    return get_registered_method_weights()


def create_app(
    settings: Settings | None = None,
    *,
    pipeline: PipelineFn = run_pipeline,
    jobs: JobRunner | None = None,
    method_weights: dict[str, float] | None = None,
) -> FastAPI:
    """Dependencias injetaveis (``pipeline``, ``jobs``) permitem testar sem o nucleo pesado."""
    settings = settings or Settings.from_env()
    # O nucleo vive na raiz do repositorio; garante o import quando executado de outro cwd.
    if str(settings.repo_root) not in sys.path:
        sys.path.insert(0, str(settings.repo_root))

    weights = method_weights if method_weights is not None else _core_method_weights()
    job_runner = jobs or ThreadPoolJobRunner(settings.max_workers)
    catalog = DatasetCatalog(settings.repo_root, settings.uploads_dir)
    dataset_service = DatasetService(catalog, settings.max_upload_bytes)
    run_service = RunService(
        FileRunRepository(settings.runs_dir),
        dataset_service,
        job_runner,
        pipeline,
        available_methods=list(weights),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        job_runner.shutdown()

    app = FastAPI(title="Causal Discovery TS API", version="1.0.0", lifespan=lifespan)
    app.state.container = Container(datasets=dataset_service, runs=run_service, method_weights=weights)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Location"],
    )
    install_error_handlers(app)

    api = APIRouter(prefix=API_PREFIX)
    for module in (health, methods, datasets, runs):
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

