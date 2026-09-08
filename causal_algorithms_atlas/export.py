from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from causal_algorithms_atlas.schema import AlgorithmCard

SUMMARY_SECTION_NAME = "Resumo estruturado"


@dataclass(frozen=True)
class Chunk:
    id: str
    algorithm_id: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)


def _card_metadata(card: AlgorithmCard) -> dict:
    return {
        "family": card.family.value,
        "temporal_handling": card.temporal_handling.value,
        "output_type": card.output_type.value,
        "handles_latent_confounders": card.handles_latent_confounders,
        "handles_nonlinearity": card.handles_nonlinearity,
        "handles_contemporaneous_effects": card.handles_contemporaneous_effects,
        "implemented_in_framework": card.implemented_in_framework,
        "verification": card.verification.value,
    }


def _summary_text(card: AlgorithmCard) -> str:
    confundidor = "tolera" if card.handles_latent_confounders else "nao tolera"
    nao_linear = "captura relacoes nao lineares" if card.handles_nonlinearity else "assume relacoes lineares"
    instantaneo = (
        "modela efeitos instantaneos (lag 0)"
        if card.handles_contemporaneous_effects
        else "nao modela efeitos instantaneos (lag 0)"
    )
    assumption_ids = ", ".join(a.id for a in card.assumptions) or "nenhuma premissa declarada"
    return (
        f"{card.name} e um metodo da familia {card.family.value}, com tratamento temporal "
        f"{card.temporal_handling.value}, que {confundidor} confundidores latentes, "
        f"{nao_linear} e {instantaneo}. Saida do tipo {card.output_type.value}. "
        f"Premissas declaradas: {assumption_ids}."
    )


def cards_to_chunks(cards: dict[str, AlgorithmCard]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for card in cards.values():
        metadata = _card_metadata(card)
        chunks.append(
            Chunk(
                id=f"{card.id}#{SUMMARY_SECTION_NAME}",
                algorithm_id=card.id,
                section=SUMMARY_SECTION_NAME,
                text=_summary_text(card),
                metadata=metadata,
            )
        )
        for section_name, section_text in card.sections.items():
            chunks.append(
                Chunk(
                    id=f"{card.id}#{section_name}",
                    algorithm_id=card.id,
                    section=section_name,
                    text=section_text,
                    metadata=metadata,
                )
            )
    return chunks


def write_chunks_jsonl(chunks: list[Chunk], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            record = {
                "id": chunk.id,
                "algorithm_id": chunk.algorithm_id,
                "section": chunk.section,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
