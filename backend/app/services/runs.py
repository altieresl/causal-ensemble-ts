"""Casos de uso de execucoes (Async Request-Reply: cria -> consulta status -> le resultado)."""

from __future__ import annotations

import logging
import threading
import uuid

from ..adapters.run_store import FileRunRepository
from ..domain import ConflictError, DomainError, Run, RunParams, RunStatus, utc_now
from .datasets import DatasetService
from .jobs import JobRunner
from .pipeline import PipelineFn

log = logging.getLogger(__name__)

VALID_RELATIONS = {"strong", "weak", "inverse", "none"}
VALID_CONSTRAINTS = {"soft", "hard"}


class RunService:
    def __init__(
        self,
        repository: FileRunRepository,
        datasets: DatasetService,
        jobs: JobRunner,
        pipeline: PipelineFn,
        available_methods: list[str] | None = None,
    ) -> None:
        self._repo = repository
        self._datasets = datasets
        self._jobs = jobs
        self._pipeline = pipeline
        self._available_methods = available_methods
        self._lock = threading.Lock()

    def create(self, params: RunParams) -> Run:
        self._validate(params)
        run = Run(id=f"run_{uuid.uuid4().hex[:10]}", params=params)
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
        """Na fila/rodando: cancela (descarta o resultado). Terminada: remove o registro."""
        with self._lock:
            run = self._repo.get(run_id)
            if run.status.terminal:
                self._repo.delete(run_id)
                return
            run.status = RunStatus.CANCELLED
            run.finished_at = utc_now()
            self._repo.save(run)

    # -- internos ---------------------------------------------------------
    def _validate(self, params: RunParams) -> None:
        bundle = self._datasets.load(params.dataset_id)
        available = set(bundle.available_columns)
        columns = list(params.columns or bundle.selected_columns)
        unknown = sorted(set(columns) - available)
        if unknown:
            raise DomainError(f"Colunas inexistentes no dataset: {unknown}")
        if len(columns) < 2:
            raise DomainError("Selecione ao menos 2 colunas.")
        if not 1 <= params.max_lag <= 20:
            raise DomainError("max_lag deve estar entre 1 e 20.")
        if not 0.0 <= params.ensemble_threshold <= 1.0:
            raise DomainError("ensemble_threshold deve estar entre 0 e 1.")
        if params.n_bootstrap is not None and not 1 <= params.n_bootstrap <= 100:
            raise DomainError("n_bootstrap deve estar entre 1 e 100.")
        if params.methods is not None:
            if len(params.methods) < 2:
                raise DomainError("Selecione ao menos 2 metodos para formar um ensemble.")
            if self._available_methods is not None:
                unknown_methods = sorted(set(params.methods) - set(self._available_methods))
                if unknown_methods:
                    raise DomainError(f"Metodos desconhecidos: {unknown_methods}")
        for rule in params.expert_knowledge:
            if rule.get("source") not in columns or rule.get("target") not in columns:
                raise DomainError(f"Regra de especialista referencia coluna fora da selecao: {rule}")
            if str(rule.get("relation", "weak")).lower() not in VALID_RELATIONS:
                raise DomainError(f"relation invalida na regra: {rule}")
            if str(rule.get("constraint", "soft")).lower() not in VALID_CONSTRAINTS:
                raise DomainError(f"constraint invalida na regra: {rule}")
        for source, target in params.selected_relations or []:
            if source not in columns or target not in columns or source == target:
                raise DomainError(f"Relacao invalida: {source} -> {target}")

    def _execute(self, run_id: str) -> None:
        with self._lock:
            run = self._repo.get(run_id)
            if run.status is not RunStatus.QUEUED:
                return  # cancelada antes de comecar
            run.status = RunStatus.RUNNING
            run.started_at = utc_now()
            self._repo.save(run)
        try:
            bundle = self._datasets.load(run.params.dataset_id)
            result = self._pipeline(bundle, run.params)
            outcome = (RunStatus.SUCCEEDED, result, None)
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
