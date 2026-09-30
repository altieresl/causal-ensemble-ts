from typing import Any

from fastapi import APIRouter, Depends

from ..deps import Container, get_container
from ..schemas import AskRequest

router = APIRouter(prefix="/atlas", tags=["atlas"])


@router.get("/algorithms")
def list_algorithms(container: Container = Depends(get_container)) -> list[dict[str, Any]]:
    """Fichas verificadas do atlas de algoritmos (premissas, familia, referencias)."""
    return container.atlas.list_algorithms()


@router.get("/algorithms/{algorithm_id}")
def get_algorithm(algorithm_id: str, container: Container = Depends(get_container)) -> dict[str, Any]:
    return container.atlas.get_algorithm(algorithm_id)


@router.post("/ask")
def ask(body: AskRequest, container: Container = Depends(get_container)) -> dict[str, Any]:
    """RAG local: trechos recuperados por TF-IDF; resposta do Ollama apenas com generate=true."""
    return container.atlas.ask(body.query, body.k, body.generate, body.model)
