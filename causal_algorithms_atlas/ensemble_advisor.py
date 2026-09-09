from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from causal_algorithms_atlas.dataset_profile import DatasetProfile
from causal_algorithms_atlas.loader import load_algorithm_cards
from causal_algorithms_atlas.schema import VerificationStatus

_ALGORITHMS_DIR = Path(__file__).resolve().parent / "algorithms"


@dataclass(frozen=True)
class MethodRecommendation:
    """Decisao de incluir ou nao um metodo do framework, com justificativa.

    A decisao usa apenas premissas declaradas nas fichas verificadas do atlas e
    caracteristicas objetivas do dataset (estacionariedade, linearidade) -- nunca
    ground truth. Isso preserva a validade de qualquer comparacao posterior do
    ensemble contra o melhor metodo individual: a composicao do ensemble nao foi
    escolhida olhando para a resposta certa.
    """

    framework_method_name: str
    algorithm_id: str
    included: bool
    reasons: tuple[str, ...]


def recommend_framework_methods(
    profile: DatasetProfile,
    *,
    algorithms_dir: str | Path = _ALGORITHMS_DIR,
) -> list[MethodRecommendation]:
    cards = load_algorithm_cards(algorithms_dir)
    recommendations: list[MethodRecommendation] = []

    for card in sorted(cards.values(), key=lambda c: c.id):
        if not card.implemented_in_framework or not card.framework_method_name:
            continue
        if card.verification is not VerificationStatus.VERIFIED:
            continue

        reasons: list[str] = []
        included = True

        requires_stationarity = any(
            a.id == "stationarity" and a.required for a in card.assumptions
        )
        if requires_stationarity:
            if profile.mostly_stationary:
                reasons.append(
                    "Exige estacionariedade; "
                    f"{profile.stationary_fraction:.0%} das series testadas passaram no ADF."
                )
            else:
                included = False
                reasons.append(
                    "Exige estacionariedade, mas apenas "
                    f"{profile.stationary_fraction:.0%} das series testadas passaram no ADF "
                    "(considere pre-processar com CausalPreprocessor.make_stationary)."
                )

        requires_linearity = any(
            a.id == "linearity" and a.required for a in card.assumptions
        )
        if requires_linearity:
            if profile.mostly_linear:
                reasons.append(
                    "Assume relacoes lineares; "
                    f"{profile.linear_fraction:.0%} das series testadas nao rejeitaram "
                    "linearidade no teste RESET."
                )
            else:
                included = False
                reasons.append(
                    "Assume relacoes lineares, mas "
                    f"{1.0 - profile.linear_fraction:.0%} das series testadas mostraram "
                    "nao linearidade significativa no teste RESET."
                )
        elif card.handles_nonlinearity and not profile.mostly_linear:
            reasons.append(
                "Nao assume linearidade -- compativel com o perfil predominantemente nao linear."
            )

        if not reasons:
            reasons.append(
                "Nenhuma premissa de estacionariedade/linearidade desta ficha se aplicou "
                "como filtro para este perfil."
            )

        recommendations.append(
            MethodRecommendation(
                framework_method_name=card.framework_method_name,
                algorithm_id=card.id,
                included=included,
                reasons=tuple(reasons),
            )
        )

    return recommendations


def select_candidate_methods(
    recommendations: list[MethodRecommendation],
) -> dict[str, Callable]:
    """Traduz recomendacoes incluidas em callables prontos para o ensemble."""
    from causal_discovery import get_registered_methods

    registered = get_registered_methods()
    return {
        rec.framework_method_name: registered[rec.framework_method_name]
        for rec in recommendations
        if rec.included and rec.framework_method_name in registered
    }
