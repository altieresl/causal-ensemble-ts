from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from causal_algorithms_atlas.dataset_profile import DatasetProfile
from causal_algorithms_atlas.loader import load_algorithm_cards
from causal_algorithms_atlas.schema import VerificationStatus

_ALGORITHMS_DIR = Path(__file__).resolve().parent / "algorithms"


@dataclass(frozen=True)
class MethodDecision:
    """Decisao do chat sobre um unico algoritmo, com o prompt/resposta que a geraram."""

    name: str
    include: bool
    reason: str
    prompt: str
    raw_response: str


@dataclass(frozen=True)
class ChatMethodSelection:
    """Selecao de metodos feita pelo chat local, um algoritmo por vez.

    Assim como ``recommend_framework_methods``, a decisao do chat nunca consulta
    ``ground_truth`` -- ele so recebe o perfil estatistico (agregado e por
    variavel) e a ficha de um algoritmo por chamada. Perguntar sobre um metodo
    de cada vez, em vez de pedir a lista dos 8 em um unico JSON, evitou boa
    parte das autocontradicoes observadas na versao em lote (ex.: justificar a
    exclusao de X por exigir premissa P e, no mesmo JSON, incluir X mesmo
    assim). Como cada resposta so pode dizer sim/nao sobre o metodo perguntado,
    nao ha nome inventado possivel -- por isso nao existe mais um campo
    ``hallucinated``.
    """

    included: tuple[str, ...]
    excluded: tuple[str, ...]
    justification: str
    language: str
    decisions: tuple[MethodDecision, ...]


def _known_methods(algorithms_dir: str | Path) -> list[str]:
    cards = load_algorithm_cards(algorithms_dir)
    return sorted(
        card.framework_method_name
        for card in cards.values()
        if card.implemented_in_framework
        and card.framework_method_name
        and card.verification is VerificationStatus.VERIFIED
    )


def _catalog_text(algorithms_dir: str | Path, names: list[str], language: str) -> str:
    """Monta a descricao de um ou mais algoritmos a partir das fichas reais.

    As fichas em ``algorithms/*.md`` sao a fonte de verdade unica das premissas
    (``loader.load_algorithm_cards``), a mesma usada por
    ``recommend_framework_methods``. O idioma so muda os *rotulos* das secoes
    no prompt (para portugues, quando ``language="pt"``); o conteudo (Core
    idea/Assumptions) nunca e duplicado ou traduzido a mao.
    """
    idea_label = "Ideia central" if language == "pt" else "Core idea"
    premises_label = "Premissas" if language == "pt" else "Assumptions"
    cards = load_algorithm_cards(algorithms_dir)
    by_method = {c.framework_method_name: c for c in cards.values()}
    blocks = []
    for name in names:
        card = by_method[name]
        idea = card.sections.get("Core idea", "").strip()
        premissas = card.sections.get("Assumptions", "").strip()
        blocks.append(f'### "{name}"\n{idea_label}: {idea}\n{premises_label}: {premissas}')
    return "\n\n".join(blocks)


def _profile_text_en(profile: DatasetProfile) -> str:
    stationarity_txt = "stationary" if profile.mostly_stationary else "non-stationary"
    stationarity_pct = (
        profile.stationary_fraction
        if profile.mostly_stationary
        else 1.0 - profile.stationary_fraction
    )
    linearity_txt = "linear" if profile.mostly_linear else "non-linear"
    linearity_pct = (
        profile.linear_fraction if profile.mostly_linear else 1.0 - profile.linear_fraction
    )
    return (
        f"Dataset with {profile.n_variables} variables and {profile.n_timepoints} "
        f"observations. {stationarity_pct:.0%} of the tested series are "
        f"{stationarity_txt} (ADF test, alpha=0.05). {linearity_pct:.0%} of "
        f"the tested series have an approximately {linearity_txt} relationship with "
        "lag 1 of themselves and the other variables (comparing out-of-sample forecast "
        "error between a linear model and a model with quadratic terms). Latent "
        "confounders cannot be verified from observational data alone (an "
        "identifiability limit, not a measurement of this profile). Which causal "
        "discovery algorithms are best suited to this profile?"
    )


def _variable_detail_text(profile: DatasetProfile, language: str) -> str:
    """Detalha estacionariedade/linearidade por variavel, nao so o agregado.

    ``recommend_framework_methods`` decide olhando o agregado (``mostly_*``),
    mas o agregado sozinho ja apagou a informacao de qual variavel especifica
    falha em qual premissa. Dar essa granularidade ao chat reduz a chance de
    ele precisar "adivinhar" a partir de uma unica porcentagem.
    """
    lines = []
    for variable in profile.variables:
        if language == "pt":
            if variable.stationary is None:
                stat = "estacionariedade nao testavel (serie curta demais)"
            else:
                verdict = "estacionaria" if variable.stationary else "nao estacionaria"
                stat = f"{verdict} (ADF p={variable.adf_p_value:.3f})"
            if variable.linear is None:
                lin = "linearidade nao testavel"
            else:
                verdict = "linear" if variable.linear else "nao linear"
                lin = f"{verdict} (efeito de nao linearidade={variable.nonlinearity_effect_size:.4f})"
            lines.append(f"- {variable.name}: {stat}; {lin}")
        else:
            if variable.stationary is None:
                stat = "stationarity not testable (series too short)"
            else:
                verdict = "stationary" if variable.stationary else "non-stationary"
                stat = f"{verdict} (ADF p={variable.adf_p_value:.3f})"
            if variable.linear is None:
                lin = "linearity not testable"
            else:
                verdict = "linear" if variable.linear else "non-linear"
                lin = f"{verdict} (nonlinearity effect size={variable.nonlinearity_effect_size:.4f})"
            lines.append(f"- {variable.name}: {stat}; {lin}")
    return "\n".join(lines)


_METHOD_PROMPT_PT = (
    "Voce e um assistente que decide se UM algoritmo de causal discovery deve "
    "entrar em um ensemble, dado o perfil do dataset abaixo. Voce nao tem acesso "
    "ao grafo causal real (ground truth) e nao deve supor nada alem do que foi "
    "fornecido.\n\n"
    "Perfil agregado do dataset: {profile_text}\n\n"
    "Perfil por variavel:\n{variable_detail}\n\n"
    "Algoritmo em avaliacao:\n{catalog_text}\n\n"
    "Tarefa: decida se este algoritmo deve ser incluido no ensemble para este "
    "perfil, respeitando estritamente as premissas obrigatorias listadas acima "
    "(se a premissa disser que estacionariedade e obrigatoria, so inclua se o "
    "perfil for predominantemente estacionario; o mesmo vale para linearidade). "
    "Responda SOMENTE com um objeto JSON, sem nenhum texto antes ou depois, no "
    'formato exato: {{"include": true ou false, "reason": "uma frase curta"}}'
)

_METHOD_PROMPT_EN = (
    "You are an assistant that decides whether ONE causal discovery algorithm "
    "should be included in an ensemble, given the dataset profile below. You do "
    "not have access to the true causal graph (ground truth) and must not "
    "assume anything beyond what is provided.\n\n"
    "Aggregate dataset profile: {profile_text}\n\n"
    "Per-variable profile:\n{variable_detail}\n\n"
    "Algorithm under evaluation:\n{catalog_text}\n\n"
    "Task: decide whether this algorithm should be included in the ensemble for "
    "this profile, strictly respecting the mandatory assumptions listed above "
    "(if an assumption says stationarity is mandatory, only include it if the "
    "profile is mostly stationary; the same applies to linearity). Respond ONLY "
    "with a JSON object, with no text before or after, in this exact format: "
    '{{"include": true or false, "reason": "one short sentence"}}'
)


def _build_method_prompt(
    profile: DatasetProfile, algorithms_dir: str | Path, language: str, method_name: str
) -> str:
    if language == "pt":
        profile_text = profile.to_query_text()
        template = _METHOD_PROMPT_PT
    elif language == "en":
        profile_text = _profile_text_en(profile)
        template = _METHOD_PROMPT_EN
    else:
        raise ValueError(f"Idioma nao suportado: {language!r} (use 'pt' ou 'en').")
    variable_detail = _variable_detail_text(profile, language)
    catalog_text = _catalog_text(algorithms_dir, [method_name], language)
    return template.format(
        profile_text=profile_text, variable_detail=variable_detail, catalog_text=catalog_text
    )


def _parse_method_response(raw_response: str) -> tuple[bool, str]:
    text = raw_response.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {}
    include = bool(parsed.get("include", False))
    reason = str(parsed.get("reason", ""))
    return include, reason


def recommend_methods_via_chat(
    profile: DatasetProfile,
    *,
    algorithms_dir: str | Path = _ALGORITHMS_DIR,
    model: str = "qwen2.5:7b",
    language: str = "pt",
) -> ChatMethodSelection:
    """Pede ao chat local (Ollama) para decidir, um algoritmo por vez, quais
    entram no ensemble.

    Diferente de ``recommend_framework_methods`` (regra deterministica sobre as
    premissas declaradas nas fichas), aqui quem decide inclusao/exclusao e o
    LLM -- para medir se ele consegue reproduzir uma selecao estatisticamente
    correta. Cada algoritmo e avaliado em uma chamada separada (em vez de pedir
    a lista dos 8 em um unico JSON) porque decidir tudo de uma vez foi a maior
    fonte de autocontradicao observada nos testes. Nunca recebe ``ground_truth``.
    """
    from causal_algorithms_atlas import rag_chat

    names = _known_methods(algorithms_dir)
    decisions: list[MethodDecision] = []
    for name in names:
        prompt = _build_method_prompt(profile, algorithms_dir, language, name)
        raw_response = rag_chat.call_ollama(prompt, model=model, format="json")
        include, reason = _parse_method_response(raw_response)
        decisions.append(
            MethodDecision(
                name=name, include=include, reason=reason, prompt=prompt, raw_response=raw_response
            )
        )

    included = tuple(d.name for d in decisions if d.include)
    excluded = tuple(d.name for d in decisions if not d.include)
    justification = " ".join(f"{d.name}: {d.reason}" for d in decisions if d.reason)
    return ChatMethodSelection(
        included=included,
        excluded=excluded,
        justification=justification,
        language=language,
        decisions=tuple(decisions),
    )
