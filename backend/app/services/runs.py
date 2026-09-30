"""Casos de uso de execucoes (Async Request-Reply: cria -> consulta status -> le resultado).

Uma execucao tem um ``kind`` (pipeline, benchmark, validacao replicada, experimento do atlas,
chat); cada kind registra ``validate`` (rapido, no request) e ``execute`` (em segundo plano).
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..adapters.run_store import FileRunRepository
from ..domain import ConflictError, DomainError, Run, RunCancelledError, RunStatus, utc_now
from .jobs import JobRunner

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]


@dataclass(frozen=True)
class KindSpec:
    validate: Callable[[dict[str, Any]], dict[str, Any]]  # devolve os parametros normalizados
    execute: Callable[[dict[str, Any], ProgressFn], dict[str, Any]]


class RunService:
    def __init__(self, repository: FileRunRepository, jobs: JobRunner, kinds: dict[str, KindSpec]) -> None:
        self._repo = repository
        self._jobs = jobs
        self._kinds = kinds
        self._lock = threading.Lock()

    def create(self, kind: str, params: dict[str, Any]) -> Run:
        spec = self._kinds.get(kind)
        if spec is None:
            raise DomainError(f"Tipo de execucao desconhecido: {kind}")
        run = Run(id=f"run_{uuid.uuid4().hex[:10]}", kind=kind, params=spec.validate(params))
        self._repo.save(run)
        self._jobs.submit(lambda: self._execute(run.id))
        return self._repo.get(run.id)

    def get(self, run_id: str) -> Run:
        return self._repo.get(run_id)

    def list(self) -> list[Run]:
        return self._repo.list()

    def result(self, run_id: str) -> dict:
        run = self._repo.get(run_id)
        if run.status is not RunStatus.SUCCEEDED or run.result is None:
            raise ConflictError(f"Execucao '{run_id}' esta '{run.status.value}'; sem resultado disponivel.")
        return run.result

    def delete(self, run_id: str) -> None:
        """Na fila/rodando: cancela (a execucao para no proximo ponto de progresso). Terminada: remove."""
        with self._lock:
            run = self._repo.get(run_id)
            if run.status.terminal:
                self._repo.delete(run_id)
                return
            run.status = RunStatus.CANCELLED
            run.finished_at = utc_now()
            self._repo.save(run)

    # -- internos ---------------------------------------------------------
    def _reporter(self, run_id: str) -> ProgressFn:
        def report(done: int, total: int, message: str) -> None:
            with self._lock:
                run = self._repo.get(run_id) if self._exists(run_id) else None
                if run is None or run.status is RunStatus.CANCELLED:
                    raise RunCancelledError(run_id)
                run.progress = {"done": int(done), "total": int(total), "message": message}
                self._repo.save(run)

        return report

    def _execute(self, run_id: str) -> None:
        with self._lock:
            run = self._repo.get(run_id)
            if run.status is not RunStatus.QUEUED:
                return  # cancelada antes de comecar
            run.status = RunStatus.RUNNING
            run.started_at = utc_now()
            self._repo.save(run)
        try:
            result = self._kinds[run.kind].execute(run.params, self._reporter(run_id))
            outcome = (RunStatus.SUCCEEDED, result, None)
        except RunCancelledError:
            return
        except Exception as error:  # noqa: BLE001 - qualquer falha do nucleo vira status 'failed'
            log.exception("Execucao %s falhou", run_id)
            outcome = (RunStatus.FAILED, None, f"{type(error).__name__}: {error}")
        with self._lock:
            run = self._repo.get(run_id) if self._exists(run_id) else None
            if run is None or run.status is RunStatus.CANCELLED:
                return  # cancelada/removida durante a execucao: descarta o resultado
            run.status, run.result, run.error = outcome
            run.finished_at = utc_now()
            self._repo.save(run)

    def _exists(self, run_id: str) -> bool:
        try:
            self._repo.get(run_id)
            return True
        except DomainError:
            return False
