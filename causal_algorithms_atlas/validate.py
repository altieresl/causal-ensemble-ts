from __future__ import annotations

from causal_algorithms_atlas.schema import AlgorithmCard


class ValidationError(ValueError):
    """Raised when cross-referenced atlas content is inconsistent."""


def validate_references(
    cards: dict[str, AlgorithmCard], references: dict[str, dict]
) -> None:
    for card in cards.values():
        missing = [key for key in card.references if key not in references]
        if missing:
            raise ValidationError(
                f"Ficha {card.id!r} referencia chaves ausentes em references.yaml: {missing}."
            )


def validate_framework_alignment(
    cards: dict[str, AlgorithmCard], framework_method_names: frozenset[str]
) -> None:
    for card in cards.values():
        if card.implemented_in_framework and not card.framework_method_name:
            raise ValidationError(
                f"Ficha {card.id!r}: implemented_in_framework=true exige framework_method_name."
            )
        if (
            card.framework_method_name is not None
            and card.framework_method_name not in framework_method_names
        ):
            raise ValidationError(
                f"Ficha {card.id!r}: framework_method_name={card.framework_method_name!r} "
                f"nao esta registrado no framework ({sorted(framework_method_names)})."
            )
