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
                    "Assume relacoes lineares; em "
                    f"{profile.linear_fraction:.0%} das series testadas, permitir termos "
                    "nao lineares nao reduziu o erro de previsao fora da amostra o "
                    "suficiente para importar na pratica."
                )
            else:
                included = False
                reasons.append(
                    "Assume relacoes lineares, mas em "
                    f"{1.0 - profile.linear_fraction:.0%} das series testadas um modelo "
                    "com termos nao lineares previu bem melhor fora da amostra."
                )
        elif card.handles_nonlinearity and not profile.mostly_linear:
            reasons.append(
                "Nao assume linearidade -- compativel com o perfil predominantemente nao linear."
            )

        requires_non_gaussian = any(
            a.id == "non_gaussian_errors" and a.required for a in card.assumptions
        )
        if requires_non_gaussian:
            if profile.mostly_non_gaussian:
                reasons.append(
                    "Exige residuos nao gaussianos; em "
                    f"{profile.non_gaussian_fraction:.0%} das series testadas, skewness ou "
                    "curtose em excesso dos residuos de um VAR(1) passaram do limiar calibrado."
                )
            else:
                included = False
                reasons.append(
                    "Exige residuos nao gaussianos, mas em apenas "
                    f"{profile.non_gaussian_fraction:.0%} das series testadas skewness/curtose "
                    "dos residuos passaram do limiar calibrado (identificabilidade da estrutura "
                    "instantanea fica comprometida com residuos proximos de gaussianos)."
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
    """Traduz recomendacoes incluidas em callables prontos para o ensemble.

    Filtro RIGIDO: so devolve metodos que passaram em todas as premissas
    estatisticas verificaveis. Esta e a versao usada como gabarito determinístico
    (a coluna "Estatistico (correto)" usada pra avaliar o chat) -- precisa
    continuar binaria e reprodutivel para essa comparacao continuar fazendo
    sentido. Para alimentar o ensemble de verdade, considerando tambem
    candidatos que violam uma premissa mas podem se provar robustos sob
    bootstrap, use ``select_candidate_methods_with_assumption_flags``.
    """
    from causal_discovery import get_registered_methods

    registered = get_registered_methods()
    return {
        rec.framework_method_name: registered[rec.framework_method_name]
        for rec in recommendations
        if rec.included and rec.framework_method_name in registered
    }


def select_candidate_methods_with_assumption_flags(
    recommendations: list[MethodRecommendation],
) -> dict[str, tuple[Callable, tuple[str, ...]]]:
    """Pool AMPLIADO de candidatos: todo metodo verified+implementado entra,
    mesmo violando uma premissa estatistica -- devolvido junto com as razoes
    da violacao (tupla vazia se nao violou nada).

    Isso existe porque uma premissa estatistica violada ("residuos gaussianos
    demais para VAR-LiNGAM identificar a estrutura instantanea") nao significa
    necessariamente que o metodo desempenha mal na pratica -- ja observamos o
    oposto (VAR-LiNGAM com F1=1.0 num dataset onde a premissa falha, ver
    ``.local/apresentacao_09-09.md`` Secao 2.4). Em vez de decidir isso de
    antemao (o que exigiria uma regra ad-hoc tipo "ignore essa premissa"),
    devolve-se o metodo como candidato E a violacao como um flag visivel; quem
    decide se ele sobrevive e a metrica cega de estabilidade sob bootstrap de
    ``causal_discovery.select_robust_ensemble_combination`` (nunca
    ``ground_truth``) em ``experiment_runner.run_experiment``.

    Isto NAO substitui ``select_candidate_methods``: a versao rigida continua
    sendo o gabarito usado pra avaliar o chat, exatamente porque uma metrica de
    avaliacao precisa ser deterministica -- ver a secao correspondente na
    apresentacao sobre por que esse filtro rigido continua necessario ali.
    """
    from causal_discovery import get_registered_methods

    registered = get_registered_methods()
    return {
        rec.framework_method_name: (
            registered[rec.framework_method_name],
            () if rec.included else rec.reasons,
        )
        for rec in recommendations
        if rec.framework_method_name in registered
    }


def explain_recommendation(
    profile: DatasetProfile,
    recommendations: list[MethodRecommendation],
    *,
    algorithms_dir: str | Path = _ALGORITHMS_DIR,
    model: str = "qwen2.5:7b",
) -> str:
    """Pede ao Llama local um resumo em prosa da composicao do ensemble.

    A decisao de inclusao/exclusao ja foi tomada por ``recommend_framework_methods``
    (deterministica, sem ground truth); esta funcao so traduz essa decisao e seus
    motivos ja calculados para um paragrafo legivel, citando apenas o que foi
    fornecido no prompt.
    """
    from causal_algorithms_atlas import rag_chat

    cards = load_algorithm_cards(algorithms_dir)
    included = [rec for rec in recommendations if rec.included]
    excluded = [rec for rec in recommendations if not rec.included]

    def _describe(rec: MethodRecommendation) -> str:
        idea = cards[rec.algorithm_id].sections.get("Core idea", "")
        motivo = " ".join(rec.reasons)
        return f"- {rec.framework_method_name}: {idea}\n  Motivo: {motivo}"

    included_block = "\n".join(_describe(rec) for rec in included) or "(nenhum metodo incluido)"
    excluded_block = "\n".join(_describe(rec) for rec in excluded) or "(nenhum metodo excluido)"

    prompt = (
        "Voce e um assistente que explica, em portugues e de forma direta, por que "
        "um subconjunto de algoritmos de causal discovery foi selecionado para um "
        "ensemble, com base no perfil do dataset abaixo. Baseie-se exclusivamente "
        "nas informacoes fornecidas; nao invente premissas nao mencionadas.\n\n"
        f"Perfil do dataset: {profile.to_query_text()}\n\n"
        f"Metodos incluidos:\n{included_block}\n\n"
        f"Metodos excluidos:\n{excluded_block}\n\n"
        "Escreva um paragrafo curto resumindo a composicao do ensemble e a razao "
        "de cada exclusao."
    )
    return rag_chat.call_ollama(prompt, model=model)
