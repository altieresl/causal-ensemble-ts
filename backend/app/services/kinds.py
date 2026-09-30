"""Registro dos tipos de execucao: liga validacao + execucao de cada ``kind`` aos servicos."""

from __future__ import annotations

from typing import Any

from ..domain import DomainError, RunKind, RunParams
from . import benchmark, replicated
from .atlas import AtlasService, validate_experiment_params
from .datasets import DatasetService
from .pipeline import PipelineFn
from .runs import KindSpec

VALID_RELATIONS = {"strong", "weak", "inverse", "none"}
VALID_CONSTRAINTS = {"soft", "hard"}


def _validate_pipeline(datasets: DatasetService, available_methods: list[str] | None):
    def validate(raw: dict[str, Any]) -> dict[str, Any]:
        params = RunParams.from_dict(raw)
        bundle = datasets.load(params.dataset_id, params.columns, params.trajectory_index)
        columns = list(params.columns or bundle.selected_columns)
        if len(columns) < 2:
            raise DomainError("Selecione ao menos 2 colunas.")
        if not 1 <= params.max_lag <= 20:
            raise DomainError("max_lag deve estar entre 1 e 20.")
        if not 0.0 <= params.ensemble_threshold <= 1.0:
            raise DomainError("ensemble_threshold deve estar entre 0 e 1.")
        if params.n_bootstrap is not None and not 1 <= params.n_bootstrap <= 100:
            raise DomainError("n_bootstrap deve estar entre 1 e 100.")
        if params.decomposition_period is not None and params.decomposition_period < 2:
            raise DomainError("decomposition_period deve ser >= 2.")
        if params.methods is not None:
            if len(params.methods) < 2:
                raise DomainError("Selecione ao menos 2 metodos para formar um ensemble.")
            if available_methods is not None:
                unknown = sorted(set(params.methods) - set(available_methods))
                if unknown:
                    raise DomainError(f"Metodos desconhecidos: {unknown}")
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
        if params.selected_relations is not None and not params.selected_relations:
            raise DomainError("Selecione ao menos 1 relacao para analisar.")
        entry = datasets.get(params.dataset_id)
        return {**params.to_dict(), "panel_max_lag": entry.panel_max_lag}

    return validate


def build_kinds(
    datasets: DatasetService,
    atlas: AtlasService,
    pipeline: PipelineFn,
    available_methods: list[str] | None = None,
) -> dict[str, KindSpec]:
    def load(params: dict[str, Any]):
        return datasets.load(params["dataset_id"], params.get("columns"), params.get("trajectory_index"))

    def execute_pipeline(params: dict[str, Any], _progress) -> dict[str, Any]:
        return pipeline(load(params), RunParams.from_dict(params))

    def validate_replicated(raw: dict[str, Any]) -> dict[str, Any]:
        entry = datasets.get(raw["dataset_id"])
        if entry.replicate_excluded_trajectories is None:
            raise DomainError("Este dataset nao possui replicas independentes.")
        return {"dataset_id": entry.id, **replicated.normalize_params({k: v for k, v in raw.items() if k != "dataset_id"})}

    def execute_replicated(params: dict[str, Any], progress) -> dict[str, Any]:
        entry = datasets.get(params["dataset_id"])
        rest = {k: v for k, v in params.items() if k != "dataset_id"}
        return replicated.run_replicated_validation(
            datasets.load(entry.id, params.get("columns")), entry.replicate_excluded_trajectories or (), rest, progress
        )

    def validate_with_dataset(inner):
        def validate(raw: dict[str, Any]) -> dict[str, Any]:
            datasets.load(raw["dataset_id"], raw.get("columns"))  # falha cedo: dataset/colunas invalidos
            return inner(raw)

        return validate

    return {
        RunKind.PIPELINE.value: KindSpec(_validate_pipeline(datasets, available_methods), execute_pipeline),
        RunKind.BENCHMARK.value: KindSpec(
            benchmark.normalize_params, lambda params, progress: benchmark.run_benchmark(params, progress)
        ),
        RunKind.REPLICATED_VALIDATION.value: KindSpec(validate_replicated, execute_replicated),
        RunKind.ATLAS_EXPERIMENT.value: KindSpec(
            validate_with_dataset(lambda raw: {**validate_experiment_params(raw), "dataset_id": raw["dataset_id"], "columns": raw.get("columns")}),
            lambda params, progress: atlas.run_experiment(load(params), params, progress),
        ),
        RunKind.ATLAS_CHAT.value: KindSpec(
            validate_with_dataset(lambda raw: dict(raw)),
            lambda params, progress: atlas.run_chat_recommendation(load(params), params, progress),
        ),
    }
