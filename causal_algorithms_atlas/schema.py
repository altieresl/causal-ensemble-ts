from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SchemaError(ValueError):
    """Raised when a parsed algorithm card violates the atlas schema."""


class AlgorithmFamily(str, Enum):
    CONSTRAINT_BASED = "constraint-based"
    SCORE_BASED = "score-based"
    FUNCTIONAL_CAUSAL_MODEL = "functional-causal-model"
    GRANGER_BASED = "granger-based"
    CONTINUOUS_OPTIMIZATION = "continuous-optimization"
    INFORMATION_THEORETIC = "information-theoretic"
    HYBRID = "hybrid"


class TemporalHandling(str, Enum):
    NATIVE = "native"
    WINDOWED_UNROLLING = "windowed-unrolling"
    SEGMENTATION = "segmentation"
    NONE = "none"


class OutputType(str, Enum):
    DAG = "dag"
    PAG = "pag"
    SIGNED_GRAPH = "signed-graph"
    UNSIGNED_GRAPH = "unsigned-graph"
    PARTIAL_GRAPH = "partial-graph"


class SampleType(str, Enum):
    SINGLE_SERIES = "single-series"
    PANEL = "panel"
    BOTH = "both"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    DRAFT = "draft"


KNOWN_ASSUMPTIONS: frozenset[str] = frozenset(
    {
        "causal_sufficiency",
        "stationarity",
        "linearity",
        "acyclicity_instantaneous",
        "non_gaussian_errors",
        "faithfulness",
        "markov_condition",
        "no_selection_bias",
    }
)


@dataclass(frozen=True)
class Assumption:
    id: str
    required: bool
    statement: str

    def __post_init__(self) -> None:
        if self.id not in KNOWN_ASSUMPTIONS:
            raise SchemaError(
                f"Assumption id {self.id!r} nao esta em KNOWN_ASSUMPTIONS: "
                f"{sorted(KNOWN_ASSUMPTIONS)}."
            )


@dataclass(frozen=True)
class DataRequirements:
    min_variables: int | None
    min_timepoints: int | None
    sample_type: SampleType | None


@dataclass(frozen=True)
class AlgorithmCard:
    id: str
    name: str
    aliases: tuple[str, ...]
    family: AlgorithmFamily
    temporal_handling: TemporalHandling
    output_type: OutputType
    assumptions: tuple[Assumption, ...]
    handles_latent_confounders: bool
    handles_nonlinearity: bool
    handles_contemporaneous_effects: bool
    data_requirements: DataRequirements
    implemented_in_framework: bool
    framework_method_name: str | None
    references: tuple[str, ...]
    verification: VerificationStatus
    verified_by: str | None
    last_reviewed: str
    sections: dict[str, str] = field(default_factory=dict)
    source_path: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise SchemaError("O campo 'id' da ficha nao pode ser vazio.")
        if not self.name.strip():
            raise SchemaError("O campo 'name' da ficha nao pode ser vazio.")
        if self.verification is VerificationStatus.VERIFIED and not (
            self.verified_by and self.verified_by.strip()
        ):
            raise SchemaError(
                f"Ficha {self.id!r}: verification=verified exige 'verified_by' preenchido."
            )
        if not self.references:
            raise SchemaError(f"Ficha {self.id!r}: precisa de ao menos uma referencia.")
