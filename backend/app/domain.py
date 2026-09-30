"""Tipos de dominio da API, independentes de FastAPI e de pandas."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class DomainError(Exception):
    """Base das falhas previstas; a camada HTTP as traduz para problem+json."""

    status = 400
    title = "Requisicao invalida"


class NotFoundError(DomainError):
    status = 404
    title = "Recurso nao encontrado"


class ConflictError(DomainError):
    status = 409
    title = "Conflito de estado"


class ServiceUnavailableError(DomainError):
    status = 503
    title = "Servico indisponivel"


class RunCancelledError(Exception):
    """Levantada por ``ProgressReporter`` quando a execucao foi cancelada (cancelamento cooperativo)."""


class RunKind(str, Enum):
    PIPELINE = "pipeline"
    BENCHMARK = "benchmark"
    REPLICATED_VALIDATION = "replicated_validation"
    ATLAS_EXPERIMENT = "atlas_experiment"
    ATLAS_CHAT = "atlas_chat"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}


_ID_PATTERN = re.compile(r"^[a-z0-9_]{1,64}$")


def is_safe_id(value: str) -> bool:
    """IDs opacos: impedem path traversal ao mapear IDs para arquivos."""
    return bool(_ID_PATTERN.match(value))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class RunParams:
    dataset_id: str
    columns: list[str] | None = None
    max_lag: int = 2
    make_stationary: bool = True
    normalize: bool = True
    quick_mode: bool = False
    n_bootstrap: int | None = None
    parallel_jobs: int | None = None
    methods: list[str] | None = None
    expert_knowledge: list[dict[str, Any]] = field(default_factory=list)
    selected_relations: list[tuple[str, str]] | None = None
    ensemble_threshold: float = 0.5
    random_state: int = 42
    trajectory_index: int | None = None
    decomposition_period: int | None = None
    panel_evidence: bool = True
    panel_max_lag: int = 1
    objective: dict[str, Any] | None = None  # so registro: o efeito esta em selected_relations

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunParams":
        data = dict(payload)
        if data.get("selected_relations") is not None:
            data["selected_relations"] = [tuple(pair) for pair in data["selected_relations"]]
        return cls(**data)


@dataclass
class Run:
    id: str
    params: dict[str, Any]
    kind: str = RunKind.PIPELINE.value
    status: RunStatus = RunStatus.QUEUED
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    progress: dict[str, Any] | None = None  # {"done": int, "total": int, "message": str}

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "status": self.status.value}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Run":
        data = dict(payload)
        data["status"] = RunStatus(data["status"])
        data.setdefault("kind", RunKind.PIPELINE.value)
        return cls(**data)
