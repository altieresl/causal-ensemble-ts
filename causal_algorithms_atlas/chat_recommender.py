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
    """Decisao do chat sobre um unico algoritmo, com o prompt/resposta que a geraram.

    Os quatro campos booleanos/numericos antes de ``include`` sao preenchidos
    pelo proprio LLM, forcados pelo JSON Schema em ``_METHOD_JSON_SCHEMA`` --
    nao sao calculados em Python. A ideia e obrigar o modelo a *reescrever* a
    porcentagem relevante e compara-la a 50% explicitamente, em vez de reagir
    so a palavra "nao estacionario"/"nao linear" aparecendo em algum lugar do
    prompt (foi essa confusao que causou exclusoes em massa incorretas na
    versao anterior, sem esses campos).
    """

    name: str
    stationary_fraction_pct: float | None
    dataset_is_majority_stationary: bool | None
    linear_fraction_pct: float | None
    dataset_is_majority_linear: bool | None
    algorithm_requires_stationarity: bool | None
    algorithm_requires_linearity: bool | None
    include: bool
    reason: str
    prompt: str
    raw_response: str
    retried: bool


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
    # Mirrors DatasetProfile.to_query_text(): always state both fractions explicitly
    # (stationary and non-stationary, linear and non-linear) so a reader -- human or
    # LLM -- never has to compute "100 - X" to know the number for the "not" case.
    return (
        f"Dataset with {profile.n_variables} variables and {profile.n_timepoints} "
        f"observations. {profile.stationary_fraction:.0%} of the tested series are "
        f"stationary and {1.0 - profile.stationary_fraction:.0%} are not (ADF test, "
        f"alpha=0.05). {profile.linear_fraction:.0%} of the tested series have an "
        f"approximately linear relationship with lag 1 of themselves and the other "
        f"variables, and {1.0 - profile.linear_fraction:.0%} do not (comparing "
        "out-of-sample forecast error between a linear model and a model with "
        "quadratic terms). Latent confounders cannot be verified from observational "
        "data alone (an identifiability limit, not a measurement of this profile). "
        "Which causal discovery algorithms are best suited to this profile?"
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


_METHOD_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "stationary_fraction_pct": {
            "type": "number",
            "description": "The exact percentage of series reported as stationary in the aggregate dataset profile above (0-100).",
        },
        "dataset_is_majority_stationary": {
            "type": "boolean",
            "description": "True only if stationary_fraction_pct >= 50.",
        },
        "linear_fraction_pct": {
            "type": "number",
            "description": "The exact percentage of series reported as linear in the aggregate dataset profile above (0-100).",
        },
        "dataset_is_majority_linear": {
            "type": "boolean",
            "description": "True only if linear_fraction_pct >= 50.",
        },
        "algorithm_requires_stationarity": {
            "type": "boolean",
            "description": "True only if the Assumptions section for this specific algorithm states stationarity is REQUIRED.",
        },
        "algorithm_requires_linearity": {
            "type": "boolean",
            "description": "True only if the Assumptions section for this specific algorithm states linearity is REQUIRED.",
        },
        "include": {
            "type": "boolean",
            "description": "False if (algorithm_requires_stationarity and not dataset_is_majority_stationary) or (algorithm_requires_linearity and not dataset_is_majority_linear); true otherwise.",
        },
        "reason": {"type": "string"},
    },
    "required": [
        "stationary_fraction_pct",
        "dataset_is_majority_stationary",
        "linear_fraction_pct",
        "dataset_is_majority_linear",
        "algorithm_requires_stationarity",
        "algorithm_requires_linearity",
        "include",
        "reason",
    ],
}

_METHOD_PROMPT_PT = (
    "Voce e um assistente que decide se UM algoritmo de causal discovery deve "
    "entrar em um ensemble, dado o perfil do dataset abaixo. Voce nao tem acesso "
    "ao grafo causal real (ground truth) e nao deve supor nada alem do que foi "
    "fornecido.\n\n"
    "Perfil agregado do dataset: {profile_text}\n\n"
    "Perfil por variavel:\n{variable_detail}\n\n"
    "Algoritmo em avaliacao:\n{catalog_text}\n\n"
    "Tarefa: preencha cada campo do JSON na ordem abaixo, um de cada vez:\n"
    "1. Copie a porcentagem exata de series estacionarias do perfil agregado acima.\n"
    "2. Diga se essa porcentagem e >= 50 (maioria estacionaria).\n"
    "3. Copie a porcentagem exata de series lineares do perfil agregado acima.\n"
    "4. Diga se essa porcentagem e >= 50 (maioria linear).\n"
    "5. Olhando SO a secao de premissas do algoritmo acima, diga se ele exige "
    "estacionariedade obrigatoriamente.\n"
    "6. Olhando SO a secao de premissas do algoritmo acima, diga se ele exige "
    "linearidade obrigatoriamente.\n"
    "7. Inclua o algoritmo apenas se nenhuma premissa obrigatoria dele for "
    "violada pela maioria calculada nos passos 2 e 4 (uma premissa so e violada "
    "se o algoritmo a exige E a maioria correspondente for falsa).\n"
    "8. Justifique em uma frase curta.\n\n"
    "Responda SOMENTE com um objeto JSON, sem nenhum texto antes ou depois."
)

_METHOD_PROMPT_EN = (
    "You are an assistant that decides whether ONE causal discovery algorithm "
    "should be included in an ensemble, given the dataset profile below. You do "
    "not have access to the true causal graph (ground truth) and must not "
    "assume anything beyond what is provided.\n\n"
    "Aggregate dataset profile: {profile_text}\n\n"
    "Per-variable profile:\n{variable_detail}\n\n"
    "Algorithm under evaluation:\n{catalog_text}\n\n"
    "Task: fill in each JSON field below, one at a time, in order:\n"
    "1. Copy the exact percentage of stationary series from the aggregate "
    "profile above.\n"
    "2. State whether that percentage is >= 50 (majority stationary).\n"
    "3. Copy the exact percentage of linear series from the aggregate profile "
    "above.\n"
    "4. State whether that percentage is >= 50 (majority linear).\n"
    "5. Looking ONLY at the algorithm's Assumptions section above, state "
    "whether it mandatorily requires stationarity.\n"
    "6. Looking ONLY at the algorithm's Assumptions section above, state "
    "whether it mandatorily requires linearity.\n"
    "7. Include the algorithm only if none of its mandatory assumptions are "
    "violated by the majorities computed in steps 2 and 4 (an assumption is "
    "only violated if the algorithm requires it AND the corresponding majority "
    "is false).\n"
    "8. Justify in one short sentence.\n\n"
    "Respond ONLY with a JSON object, with no text before or after."
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


def _parse_method_response(raw_response: str) -> dict:
    text = raw_response.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {}
    return {
        "stationary_fraction_pct": parsed.get("stationary_fraction_pct"),
        "dataset_is_majority_stationary": parsed.get("dataset_is_majority_stationary"),
        "linear_fraction_pct": parsed.get("linear_fraction_pct"),
        "dataset_is_majority_linear": parsed.get("dataset_is_majority_linear"),
        "algorithm_requires_stationarity": parsed.get("algorithm_requires_stationarity"),
        "algorithm_requires_linearity": parsed.get("algorithm_requires_linearity"),
        "include": bool(parsed.get("include", False)),
        "reason": str(parsed.get("reason", "")),
    }


def _is_trustworthy(parsed: dict, profile: DatasetProfile, *, tolerance_pct: float = 5.0) -> bool:
    """Confere a resposta do chat contra fatos que o Python ja sabe com certeza.

    Duas checagens independentes, ambas sem envolver julgamento -- so aritmetica:
    (1) as porcentagens que o modelo *reescreveu* batem com o perfil real (ele as
    recebeu prontas no prompt; se divergirem, ele leu/copiou errado); (2) o
    ``include`` final e consistente com os proprios booleanos que ele declarou
    (formula explicada em ``_METHOD_JSON_SCHEMA["properties"]["include"]``). Uma
    resposta que falha aqui e re-tentada uma vez em vez de aceita silenciosamente.
    """
    true_stat_pct = profile.stationary_fraction * 100
    true_lin_pct = profile.linear_fraction * 100

    stat_pct = parsed["stationary_fraction_pct"]
    lin_pct = parsed["linear_fraction_pct"]
    if stat_pct is None or abs(stat_pct - true_stat_pct) > tolerance_pct:
        return False
    if lin_pct is None or abs(lin_pct - true_lin_pct) > tolerance_pct:
        return False

    requires_stat = parsed["algorithm_requires_stationarity"]
    requires_lin = parsed["algorithm_requires_linearity"]
    majority_stat = parsed["dataset_is_majority_stationary"]
    majority_lin = parsed["dataset_is_majority_linear"]
    if None in (requires_stat, requires_lin, majority_stat, majority_lin):
        return False
    expected_include = not (
        (requires_stat and not majority_stat) or (requires_lin and not majority_lin)
    )
    return parsed["include"] == expected_include


def recommend_methods_via_chat(
    profile: DatasetProfile,
    *,
    algorithms_dir: str | Path = _ALGORITHMS_DIR,
    model: str = "qwen2.5:7b",
    language: str = "pt",
    max_retries: int = 1,
) -> ChatMethodSelection:
    """Pede ao chat local (Ollama) para decidir, um algoritmo por vez, quais
    entram no ensemble.

    Diferente de ``recommend_framework_methods`` (regra deterministica sobre as
    premissas declaradas nas fichas), aqui quem decide inclusao/exclusao e o
    LLM -- para medir se ele consegue reproduzir uma selecao estatisticamente
    correta. Cada algoritmo e avaliado em uma chamada separada (em vez de pedir
    a lista dos 8 em um unico JSON) porque decidir tudo de uma vez foi a maior
    fonte de autocontradicao observada nos testes. Nunca recebe ``ground_truth``.

    ``max_retries`` (default 1): quantas vezes re-tentar a chamada de um metodo
    quando ``_is_trustworthy`` encontra uma inconsistencia verificavel (numero
    copiado errado ou ``include`` que nao bate com os proprios booleanos
    declarados). Isso nunca "corrige" o julgamento do modelo -- so pede de novo
    quando a resposta contradiz fatos que o Python ja sabe com certeza.
    """
    from causal_algorithms_atlas import rag_chat

    names = _known_methods(algorithms_dir)
    decisions: list[MethodDecision] = []
    for name in names:
        prompt = _build_method_prompt(profile, algorithms_dir, language, name)
        retried = False
        for attempt in range(max_retries + 1):
            raw_response = rag_chat.call_ollama(prompt, model=model, format=_METHOD_JSON_SCHEMA)
            parsed = _parse_method_response(raw_response)
            if _is_trustworthy(parsed, profile) or attempt == max_retries:
                break
            retried = True
        decisions.append(
            MethodDecision(
                name=name,
                stationary_fraction_pct=parsed["stationary_fraction_pct"],
                dataset_is_majority_stationary=parsed["dataset_is_majority_stationary"],
                linear_fraction_pct=parsed["linear_fraction_pct"],
                dataset_is_majority_linear=parsed["dataset_is_majority_linear"],
                algorithm_requires_stationarity=parsed["algorithm_requires_stationarity"],
                algorithm_requires_linearity=parsed["algorithm_requires_linearity"],
                include=parsed["include"],
                reason=parsed["reason"],
                prompt=prompt,
                raw_response=raw_response,
                retried=retried,
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
