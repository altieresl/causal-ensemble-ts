from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class EvidenceError(ValueError):
    """Raised when an evidence record is malformed or dangles a reference."""


@dataclass(frozen=True)
class EvidenceRecord:
    algorithm_id: str
    dataset: str
    source: str
    metrics: dict[str, float]
    notes: str
    source_path: str = ""


def load_evidence_records(directory: str | Path) -> list[EvidenceRecord]:
    directory = Path(directory)
    records: list[EvidenceRecord] = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            records.append(
                EvidenceRecord(
                    algorithm_id=raw["algorithm_id"],
                    dataset=raw["dataset"],
                    source=raw["source"],
                    metrics=dict(raw.get("metrics") or {}),
                    notes=str(raw.get("notes") or ""),
                    source_path=str(path),
                )
            )
        except (KeyError, yaml.YAMLError) as exc:
            raise EvidenceError(f"{path}: registro de evidencia invalido ({exc}).") from exc
    return records


def validate_evidence_algorithm_ids(
    records: list[EvidenceRecord], known_ids: frozenset[str]
) -> None:
    unknown = [r for r in records if r.algorithm_id not in known_ids]
    if unknown:
        bad = [(r.algorithm_id, r.source_path) for r in unknown]
        raise EvidenceError(f"Registros de evidencia com algorithm_id desconhecido: {bad}.")
