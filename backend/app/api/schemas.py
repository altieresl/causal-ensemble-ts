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
        )


class RunView(BaseModel):
    id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    params: dict[str, Any]
    created_at: str
    started_at: str | None
    finished_at: str | None
    error: str | None

    @classmethod
    def from_run(cls, run: Run) -> "RunView":
        return cls(
            id=run.id,
            status=run.status.value,
            params=run.params.to_dict(),
            created_at=run.created_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
            error=run.error,
        )


class MethodView(BaseModel):
    name: str
    default_weight: float
