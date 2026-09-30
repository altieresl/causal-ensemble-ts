from typing import Any

from fastapi import APIRouter, Depends, Response, status

from ..deps import Container, get_container
from ..schemas import CreateRunRequest, RunView

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("", response_model=RunView, status_code=status.HTTP_202_ACCEPTED)
def create_run(
    body: CreateRunRequest,
    response: Response,
    container: Container = Depends(get_container),
) -> RunView:
    run = container.runs.create(body.to_params())
    response.headers["Location"] = f"/api/v1/runs/{run.id}"  # Async Request-Reply
    return RunView.from_run(run)


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
