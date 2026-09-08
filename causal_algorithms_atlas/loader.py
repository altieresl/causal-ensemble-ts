from __future__ import annotations

import re
from pathlib import Path

import yaml

from causal_algorithms_atlas.schema import (
    AlgorithmCard,
    AlgorithmFamily,
    Assumption,
    DataRequirements,
    OutputType,
    SampleType,
    SchemaError,
    TemporalHandling,
    VerificationStatus,
)

REQUIRED_SECTIONS: tuple[str, ...] = (
    "Ideia central",
    "Premissas",
    "Quando usar",
    "Quando evitar",
    "Relação com outros métodos",
)

_FRONTMATTER_PATTERN = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)
_HEADING_PATTERN = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


class LoaderError(ValueError):
    """Raised when an algorithm card file cannot be parsed."""


def _split_sections(body: str) -> dict[str, str]:
    matches = list(_HEADING_PATTERN.finditer(body))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[match.group(1)] = body[start:end].strip()
    return sections


def _build_assumptions(raw: list[dict]) -> tuple[Assumption, ...]:
    return tuple(
        Assumption(id=item["id"], required=bool(item["required"]), statement=item["statement"])
        for item in raw
    )


def _build_data_requirements(raw: dict) -> DataRequirements:
    sample_type = raw.get("sample_type")
    return DataRequirements(
        min_variables=raw.get("min_variables"),
        min_timepoints=raw.get("min_timepoints"),
        sample_type=SampleType(sample_type) if sample_type is not None else None,
    )


def parse_algorithm_card(text: str, *, source_path: str) -> AlgorithmCard:
    match = _FRONTMATTER_PATTERN.match(text)
    if not match:
        raise LoaderError(
            f"{source_path}: frontmatter YAML delimitado por '---' nao encontrado."
        )
    frontmatter_text, body = match.group(1), match.group(2)
    try:
        raw = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as exc:
        raise LoaderError(f"{source_path}: YAML invalido no frontmatter ({exc}).") from exc

    sections = _split_sections(body)
    missing = [name for name in REQUIRED_SECTIONS if name not in sections]
    if missing:
        raise LoaderError(f"{source_path}: secoes obrigatorias ausentes: {missing}.")

    try:
        card = AlgorithmCard(
            id=raw["id"],
            name=raw["name"],
            aliases=tuple(raw.get("aliases") or ()),
            family=AlgorithmFamily(raw["family"]),
            temporal_handling=TemporalHandling(raw["temporal_handling"]),
            output_type=OutputType(raw["output_type"]),
            assumptions=_build_assumptions(raw.get("assumptions") or []),
            handles_latent_confounders=bool(raw["handles_latent_confounders"]),
            handles_nonlinearity=bool(raw["handles_nonlinearity"]),
            handles_contemporaneous_effects=bool(raw["handles_contemporaneous_effects"]),
            data_requirements=_build_data_requirements(raw["data_requirements"]),
            implemented_in_framework=bool(raw["implemented_in_framework"]),
            framework_method_name=raw.get("framework_method_name"),
            references=tuple(raw.get("references") or ()),
            verification=VerificationStatus(raw["verification"]),
            verified_by=raw.get("verified_by"),
            last_reviewed=str(raw["last_reviewed"]),
            sections=sections,
            source_path=source_path,
        )
    except (KeyError, ValueError, SchemaError) as exc:
        raise LoaderError(f"{source_path}: {exc}") from exc
    return card


def load_algorithm_cards(directory: str | Path) -> dict[str, AlgorithmCard]:
    directory = Path(directory)
    cards: dict[str, AlgorithmCard] = {}
    for path in sorted(directory.glob("*.md")):
        card = parse_algorithm_card(path.read_text(encoding="utf-8"), source_path=str(path))
        if card.id in cards:
            raise LoaderError(
                f"Id de algoritmo duplicado {card.id!r}: {cards[card.id].source_path} e {path}."
            )
        cards[card.id] = card
    return cards
