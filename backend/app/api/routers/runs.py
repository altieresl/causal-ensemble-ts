from typing import Any

from fastapi import APIRouter, Depends, Response, status

from ...domain import RunKind
from ..deps import Container, get_container
from ..schemas import (
    AtlasChatRequest,
    AtlasExperimentRequest,
    BenchmarkRequest,
    CreateRunRequest,
    ReplicatedValidationRequest,
    RunView,
)

router = APIRouter(prefix="/runs", tags=["runs"])


def _accepted(container: Container, response: Response, kind: RunKind, params: dict[str, Any]) -> RunView:
    run = container.runs.create(kind.value, params)
    response.headers["Location"] = f"/api/v1/runs/{run.id}"  # Async Request-Reply
    return RunView.from_run(run)


@router.post("", response_model=RunView, status_code=status.HTTP_202_ACCEPTED)
def create_run(body: CreateRunRequest, response: Response, container: Container = Depends(get_container)):
    """Pipeline robusto (ENSEMBLE_AUTO) sobre um dataset."""
    return _accepted(container, response, RunKind.PIPELINE, body.to_params().to_dict())


@router.post("/benchmark", response_model=RunView, status_code=status.HTTP_202_ACCEPTED)
def create_benchmark(body: BenchmarkRequest, response: Response, container: Container = Depends(get_container)):
    """Benchmark sintetico com ground truth conhecido + robustez a mudanca no regime de ruido."""
    return _accepted(container, response, RunKind.BENCHMARK, body.model_dump())


@router.post("/replicated-validation", response_model=RunView, status_code=status.HTTP_202_ACCEPTED)
def create_replicated_validation(
    body: ReplicatedValidationRequest, response: Response, container: Container = Depends(get_container)
):
    """Validacao estatistica pareada em replicas independentes (~5 min por replica)."""
    return _accepted(container, response, RunKind.REPLICATED_VALIDATION, body.model_dump())


@router.post("/atlas-experiment", response_model=RunView, status_code=status.HTTP_202_ACCEPTED)
def create_atlas_experiment(
    body: AtlasExperimentRequest, response: Response, container: Container = Depends(get_container)
):
    """Perfil -> metodos compativeis -> selecao cega -> avaliacao pos-hoc."""
    return _accepted(container, response, RunKind.ATLAS_EXPERIMENT, body.model_dump())


@router.post("/atlas-chat", response_model=RunView, status_code=status.HTTP_202_ACCEPTED)
def create_atlas_chat(body: AtlasChatRequest, response: Response, container: Container = Depends(get_container)):
    """Selecao de metodos via chat local (Ollama), comparada ao filtro estatistico."""
    return _accepted(container, response, RunKind.ATLAS_CHAT, body.model_dump())


@router.get("", response_model=list[RunView])
def list_runs(container: Container = Depends(get_container)) -> list[RunView]:
    return [RunView.from_run(run) for run in container.runs.list()]


@router.get("/{run_id}", response_model=RunView)
def get_run(run_id: str, container: Container = Depends(get_container)) -> RunView:
    return RunView.from_run(container.runs.get(run_id))


@router.get("/{run_id}/result")
def get_result(run_id: str, container: Container = Depends(get_container)) -> dict[str, Any]:
    return container.runs.result(run_id)


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_run(run_id: str, container: Container = Depends(get_container)) -> Response:
    container.runs.delete(run_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
