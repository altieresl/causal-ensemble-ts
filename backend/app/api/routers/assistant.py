from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..deps import Container, get_container
from ..schemas import AssistantRequest

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.get("/status")
def assistant_status(container: Container = Depends(get_container)) -> dict[str, Any]:
    """Disponibilidade do Ollama e modelos baixados (para a interface avisar antes de perguntar)."""
    return container.assistant.status()


@router.post("/chat")
def assistant_chat(body: AssistantRequest, container: Container = Depends(get_container)) -> StreamingResponse:
    """Resposta em streaming (NDJSON): ``sources`` → ``token``* → ``done`` | ``error``."""
    messages = [m.model_dump() for m in body.messages]
    container.assistant.validate_messages(messages)  # erros de entrada viram 400 antes do stream
    context = body.context.model_dump() if body.context else None
    return StreamingResponse(
        container.assistant.stream(messages, context, body.model),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
