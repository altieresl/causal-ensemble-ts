"""Contratos HTTP (Pydantic). Sao a fonte do OpenAPI e espelham `frontend/src/api/types.ts`."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..domain import Run, RunParams


class DatasetSummary(BaseModel):
    id: str
    name: str
    description: str
    origin: Literal["builtin", "upload"]
    default_max_lag: int


class DatasetDetails(BaseModel):
    entry: DatasetSummary
    available_columns: list[str]
    selected_columns: list[str]
    n_rows: int
    preview: list[dict[str, Any]]
    has_ground_truth: bool
    default_max_lag: int
    decomposition_period: int | None
    trajectory_count: int
    supports_replicates: bool


class ProfileRequest(BaseModel):
    columns: list[str] | None = None
    # So informe se um especialista declarou; nunca derive de ground truth.
    declared_causal_sufficiency: bool | None = None


class ExpertRule(BaseModel):
    source: str
    target: str
    lag: int | None = Field(default=None, ge=0)
    relation: Literal["strong", "weak", "inverse", "none"] = "weak"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    constraint: Literal["soft", "hard"] = "soft"
    prior_probability: float | None = Field(default=None, ge=0.0, le=1.0)


class CreateRunRequest(BaseModel):
    dataset_id: str
    columns: list[str] | None = None
    max_lag: int = Field(default=2, ge=1, le=20)
    make_stationary: bool = True
    normalize: bool = True
    quick_mode: bool = False
    n_bootstrap: int | None = Field(default=None, ge=1, le=100)
    parallel_jobs: int | None = Field(default=None, ge=1, le=8)
    methods: list[str] | None = None
    expert_knowledge: list[ExpertRule] = Field(default_factory=list)
    selected_relations: list[tuple[str, str]] | None = None
    ensemble_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    trajectory_index: int | None = Field(default=None, ge=0)
    decomposition_period: int | None = Field(default=None, ge=2)
    panel_evidence: bool = True
    # Registro do objetivo de analise (ver frontend): o efeito real esta em selected_relations.
    objective: dict[str, Any] | None = None

    def to_params(self) -> RunParams:
        return RunParams(
            dataset_id=self.dataset_id,
            columns=self.columns,
            max_lag=self.max_lag,
            make_stationary=self.make_stationary,
            normalize=self.normalize,
            quick_mode=self.quick_mode,
            n_bootstrap=self.n_bootstrap,
            parallel_jobs=self.parallel_jobs,
            methods=self.methods,
            expert_knowledge=[rule.model_dump() for rule in self.expert_knowledge],
            selected_relations=self.selected_relations,
            ensemble_threshold=self.ensemble_threshold,
            trajectory_index=self.trajectory_index,
            decomposition_period=self.decomposition_period,
            panel_evidence=self.panel_evidence,
            objective=self.objective,
        )


class RunView(BaseModel):
    id: str
    kind: Literal["pipeline", "benchmark", "replicated_validation", "atlas_experiment", "atlas_chat"]
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    params: dict[str, Any]
    created_at: str
    started_at: str | None
    finished_at: str | None
    error: str | None
    progress: dict[str, Any] | None

    @classmethod
    def from_run(cls, run: Run) -> "RunView":
        return cls(
            id=run.id,
            kind=run.kind,
            status=run.status.value,
            params=run.params,
            created_at=run.created_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
            error=run.error,
            progress=run.progress,
        )


class MethodView(BaseModel):
    name: str
    default_weight: float


class BenchmarkRequest(BaseModel):
    n_samples: int = Field(default=500, ge=100, le=5000)
    noise_multiplier: float = Field(default=3.0, ge=0.0)
    index_change: int = Field(default=250, ge=0)
    n_bootstrap: int = Field(default=20, ge=1, le=100)
    max_lag: int = Field(default=2, ge=1, le=10)
    methods: list[str] | None = None


class ReplicatedValidationRequest(BaseModel):
    dataset_id: str
    columns: list[str] | None = None
    n_replicates: int = Field(default=10, ge=2, le=100)
    replicate_seed: int = 2029
    statistical_bootstraps: int = Field(default=10_000, ge=100, le=100_000)
    significance_level: float = Field(default=0.05, gt=0.0, lt=1.0)
    minimum_precision_gain: float = Field(default=0.05, ge=0.0, le=1.0)
    minimum_win_rate: float = Field(default=0.70, ge=0.0, le=1.0)
    min_confirmatory_replicates: int = Field(default=10, ge=2)
    max_lag: int = Field(default=2, ge=1, le=20)
    ensemble_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    quick_mode: bool = False
    n_bootstrap: int | None = Field(default=None, ge=1, le=100)
    methods: list[str] | None = None
    decomposition_period: int | None = Field(default=None, ge=2)
    parallel_replicas: int | None = Field(default=None, ge=1, le=8)  # None = automatico


class AtlasExperimentRequest(BaseModel):
    dataset_id: str
    columns: list[str] | None = None
    max_rows: int | None = Field(default=None, ge=50)
    n_bootstrap: int = Field(default=3, ge=1, le=100)
    max_methods: int | None = Field(default=3, ge=2, le=20)
    max_lag: int = Field(default=1, ge=1, le=20)
    methods: list[str] | None = None
    use_assumption_soft_filter: bool = True
    declared_causal_sufficiency: bool | None = None


class AtlasChatRequest(BaseModel):
    dataset_id: str
    columns: list[str] | None = None
    model: str | None = None
    max_retries: int = Field(default=1, ge=0, le=3)
    parallel_calls: int = Field(default=4, ge=1, le=8)  # chamadas simultaneas ao Ollama
    declared_causal_sufficiency: bool | None = None


class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    k: int = Field(default=4, ge=1, le=10)
    generate: bool = False  # so gera resposta com o Ollama local quando pedido
    model: str | None = None
