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

    Os campos booleanos/numericos antes de ``include`` sao preenchidos pelo
    proprio LLM, forcados pelo JSON Schema em ``_METHOD_JSON_SCHEMA`` -- nao sao
    calculados em Python. A ideia e obrigar o modelo a *reescrever* a porcentagem
    relevante e compara-la a 50% explicitamente, em vez de reagir so a palavra
    "non-stationary"/"non-linear" aparecendo em algum lugar do prompt (foi essa
    confusao que causou exclusoes em massa incorretas na versao anterior, sem
    esses campos).

    ``synthesis_corrected=True`` significa que o modelo declarou todos os seis
    booleanos corretamente (percentuais batem com o perfil real, premissas do
    algoritmo lidas certo) mas ERROU a combinacao logica final dos tres "E/OU"
    em ``include`` -- um erro de sintese, nao de leitura. Quando isso persiste
    apos as tentativas, ``include`` e recalculado em Python a partir dos
    proprios booleanos que o modelo declarou (formula em
    ``_expected_include_from_booleans``, a mesma que ``_is_trustworthy`` ja usa
    para validar) em vez de usar o veredito do LLM. Isso NUNCA usa
    ``ground_truth`` nem informacao que o modelo nao tenha ele mesmo declarado --
    e so aritmetica booleana sobre fatos que o proprio modelo ja disse serem
    verdadeiros, exatamente a mesma checagem que ``_is_trustworthy`` ja fazia
    para decidir se a resposta era confiavel.
    """

    name: str
    stationary_fraction_pct: float | None
    dataset_is_majority_stationary: bool | None
    linear_fraction_pct: float | None
    dataset_is_majority_linear: bool | None
    non_gaussian_fraction_pct: float | None
    dataset_is_majority_non_gaussian: bool | None
    algorithm_requires_stationarity: bool | None
    algorithm_requires_linearity: bool | None
    algorithm_requires_non_gaussian_errors: bool | None
    include: bool
    reason: str
    prompt: str
    raw_response: str
    retried: bool
    synthesis_corrected: bool


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


def _catalog_text(algorithms_dir: str | Path, names: list[str]) -> str:
    """Monta a descricao de um ou mais algoritmos a partir das fichas reais.

    As fichas em ``algorithms/*.md`` sao a fonte de verdade unica das premissas
    (``loader.load_algorithm_cards``), a mesma usada por
    ``recommend_framework_methods``. Nada e duplicado ou traduzido a mao.

    So mostra estacionariedade/linearidade/nao-gaussianidade (computadas com a
    mesma logica de ``recommend_framework_methods``), nao a secao "Assumptions"
    inteira. Um teste de ablacao mostrou que mostrar outras premissas da ficha
    que o schema nunca usa (``causal_sufficiency``, ``faithfulness``, ...) faz o
    modelo excluir o metodo mesmo quando os booleanos que de fato importam saem
    corretos -- ex.: o LPCMCI (nao exige causal_sufficiency) era excluido de
    forma deterministica ate essa linha ser removida do prompt; nem o nome do
    metodo nem o texto da "Core idea" tinham qualquer efeito no teste.
    """
    cards = load_algorithm_cards(algorithms_dir)
    by_method = {c.framework_method_name: c for c in cards.values()}
    blocks = []
    for name in names:
        card = by_method[name]
        idea = card.sections.get("Core idea", "").strip()
        requires_stationarity = any(
            a.id == "stationarity" and a.required for a in card.assumptions
        )
        requires_linearity = any(a.id == "linearity" and a.required for a in card.assumptions)
        requires_non_gaussian = any(
            a.id == "non_gaussian_errors" and a.required for a in card.assumptions
        )
        stat_verdict = "REQUIRED" if requires_stationarity else "NOT required"
        lin_verdict = "REQUIRED" if requires_linearity else "NOT required"
        gauss_verdict = "REQUIRED" if requires_non_gaussian else "NOT required"
        blocks.append(
            f'### "{name}"\n'
            f"Core idea: {idea}\n"
            f"Assumptions relevant to this decision:\n"
            f"- Stationarity: {stat_verdict}\n"
            f"- Linearity: {lin_verdict}\n"
            f"- Non-Gaussian errors: {gauss_verdict}"
        )
    return "\n\n".join(blocks)


def _profile_text(profile: DatasetProfile) -> str:
    """Descreve o perfil agregado em formato ``chave = valor``.

    Uma versao anterior usava uma frase em prosa ("43% ... are stationary and
    57% are not"). Mesmo declarando as duas fracoes explicitamente, o modelo
    (qwen2.5:7b) trocava sistematicamente qual dos dois numeros copiar para o
    campo ``stationary_fraction_pct`` do schema -- persistente mesmo apos
    retry, em datasets onde a fracao estava perto de 50% (ver
    toy_e_boundary_mixed no historico de testes). O formato ``chave = valor``
    abaixo usa os MESMOS nomes dos campos do JSON Schema, eliminando a
    necessidade de o modelo re-interpretar uma frase em linguagem natural para
    saber qual numero pertence a qual rotulo.
    """
    stationary_pct = profile.stationary_fraction * 100
    linear_pct = profile.linear_fraction * 100
    non_gaussian_pct = profile.non_gaussian_fraction * 100
    return (
        f"Dataset with {profile.n_variables} variables and {profile.n_timepoints} "
        "observations. Stationarity tested via ADF (alpha=0.05); linearity tested "
        "via out-of-sample forecast effect size (linear vs. quadratic model on lag "
        "1 of all variables); non-Gaussianity of errors tested via effect size "
        "(skewness/excess kurtosis of the residuals of a VAR(1) fit per variable, "
        "above a calibrated threshold) rather than a significance test. Latent "
        "confounders cannot be verified from observational data alone (an "
        "identifiability limit, not a measurement of this profile).\n"
        f"stationary_percentage = {stationary_pct:.0f}\n"
        f"non_stationary_percentage = {100.0 - stationary_pct:.0f}\n"
        f"linear_percentage = {linear_pct:.0f}\n"
        f"non_linear_percentage = {100.0 - linear_pct:.0f}\n"
        f"non_gaussian_percentage = {non_gaussian_pct:.0f}\n"
        f"gaussian_percentage = {100.0 - non_gaussian_pct:.0f}"
    )


def _variable_detail_text(profile: DatasetProfile) -> str:
    """Detalha estacionariedade/linearidade por variavel, nao so o agregado.

    ``recommend_framework_methods`` decide olhando o agregado (``mostly_*``),
    mas o agregado sozinho ja apagou a informacao de qual variavel especifica
    falha em qual premissa. Dar essa granularidade ao chat reduz a chance de
    ele precisar "adivinhar" a partir de uma unica porcentagem.
    """
    lines = []
    for variable in profile.variables:
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
        if variable.non_gaussian is None:
            gauss = "non-Gaussianity not testable"
        else:
            verdict = "non-Gaussian" if variable.non_gaussian else "Gaussian-compatible"
            gauss = (
                f"{verdict} (residual skewness={variable.residual_skewness:.3f}, "
                f"excess kurtosis={variable.residual_excess_kurtosis:.3f})"
            )
        lines.append(f"- {variable.name}: {stat}; {lin}; {gauss}")
    return "\n".join(lines)


_METHOD_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "stationary_fraction_pct": {
            "type": "number",
            "description": "Copy the value of stationary_percentage exactly as given in the aggregate dataset profile above (0-100). Do NOT copy non_stationary_percentage here.",
        },
        "dataset_is_majority_stationary": {
            "type": "boolean",
            "description": "True only if stationary_fraction_pct >= 50.",
        },
        "linear_fraction_pct": {
            "type": "number",
            "description": "Copy the value of linear_percentage exactly as given in the aggregate dataset profile above (0-100). Do NOT copy non_linear_percentage here.",
        },
        "dataset_is_majority_linear": {
            "type": "boolean",
            "description": "True only if linear_fraction_pct >= 50.",
        },
        "non_gaussian_fraction_pct": {
            "type": "number",
            "description": "Copy the value of non_gaussian_percentage exactly as given in the aggregate dataset profile above (0-100). Do NOT copy gaussian_percentage here.",
        },
        "dataset_is_majority_non_gaussian": {
            "type": "boolean",
            "description": "True only if non_gaussian_fraction_pct >= 50.",
        },
        "algorithm_requires_stationarity": {
            "type": "boolean",
            "description": "True only if the Assumptions section for this specific algorithm states stationarity is REQUIRED.",
        },
        "algorithm_requires_linearity": {
            "type": "boolean",
            "description": "True only if the Assumptions section for this specific algorithm states linearity is REQUIRED.",
        },
        "algorithm_requires_non_gaussian_errors": {
            "type": "boolean",
            "description": "True only if the Assumptions section for this specific algorithm states non-Gaussian errors are REQUIRED.",
        },
        "include": {
            "type": "boolean",
            "description": "False if (algorithm_requires_stationarity and not dataset_is_majority_stationary) or (algorithm_requires_linearity and not dataset_is_majority_linear) or (algorithm_requires_non_gaussian_errors and not dataset_is_majority_non_gaussian); true otherwise.",
        },
        "reason": {"type": "string"},
    },
    "required": [
        "stationary_fraction_pct",
        "dataset_is_majority_stationary",
        "linear_fraction_pct",
        "dataset_is_majority_linear",
        "non_gaussian_fraction_pct",
        "dataset_is_majority_non_gaussian",
        "algorithm_requires_stationarity",
        "algorithm_requires_linearity",
        "algorithm_requires_non_gaussian_errors",
        "include",
        "reason",
    ],
}

_METHOD_PROMPT = (
    "You are an assistant that decides whether ONE causal discovery algorithm "
    "should be included in an ensemble, given the dataset profile below. You do "
    "not have access to the true causal graph (ground truth) and must not "
    "assume anything beyond what is provided.\n\n"
    "Aggregate dataset profile:\n{profile_text}\n\n"
    "Per-variable profile:\n{variable_detail}\n\n"
    "Algorithm under evaluation:\n{catalog_text}\n\n"
    "Task: fill in each JSON field below, one at a time, in order:\n"
    "1. Copy the value of stationary_percentage from the aggregate profile "
    "above.\n"
    "2. State whether that value is >= 50 (majority stationary).\n"
    "3. Copy the value of linear_percentage from the aggregate profile above.\n"
    "4. State whether that value is >= 50 (majority linear).\n"
    "5. Copy the value of non_gaussian_percentage from the aggregate profile "
    "above.\n"
    "6. State whether that value is >= 50 (majority non-Gaussian).\n"
    "7. Looking ONLY at the algorithm's Assumptions section above, state "
    "whether it mandatorily requires stationarity.\n"
    "8. Looking ONLY at the algorithm's Assumptions section above, state "
    "whether it mandatorily requires linearity.\n"
    "9. Looking ONLY at the algorithm's Assumptions section above, state "
    "whether it mandatorily requires non-Gaussian errors.\n"
    "10. Include the algorithm only if none of its mandatory assumptions are "
    "violated by the majorities computed in steps 2, 4 and 6 (an assumption is "
    "only violated if the algorithm requires it AND the corresponding majority "
    "is false).\n"
    "11. Justify in one short sentence.\n\n"
    "Respond ONLY with a JSON object, with no text before or after."
)


def _build_method_prompt(profile: DatasetProfile, algorithms_dir: str | Path, method_name: str) -> str:
    profile_text = _profile_text(profile)
    variable_detail = _variable_detail_text(profile)
    catalog_text = _catalog_text(algorithms_dir, [method_name])
    return _METHOD_PROMPT.format(
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
        "non_gaussian_fraction_pct": parsed.get("non_gaussian_fraction_pct"),
        "dataset_is_majority_non_gaussian": parsed.get("dataset_is_majority_non_gaussian"),
        "algorithm_requires_stationarity": parsed.get("algorithm_requires_stationarity"),
        "algorithm_requires_linearity": parsed.get("algorithm_requires_linearity"),
        "algorithm_requires_non_gaussian_errors": parsed.get(
            "algorithm_requires_non_gaussian_errors"
        ),
        "include": bool(parsed.get("include", False)),
        "reason": str(parsed.get("reason", "")),
    }


def _percentages_match_profile(
    parsed: dict, profile: DatasetProfile, *, tolerance_pct: float = 5.0
) -> bool:
    """Confere se as porcentagens que o modelo *reescreveu* batem com o perfil real.

    Ele as recebeu prontas no prompt; se divergirem, ele leu/copiou errado (o bug
    de troca de numeros documentado em ``_profile_text``).
    """
    true_stat_pct = profile.stationary_fraction * 100
    true_lin_pct = profile.linear_fraction * 100
    true_non_gaussian_pct = profile.non_gaussian_fraction * 100

    stat_pct = parsed["stationary_fraction_pct"]
    lin_pct = parsed["linear_fraction_pct"]
    non_gaussian_pct = parsed["non_gaussian_fraction_pct"]
    if stat_pct is None or abs(stat_pct - true_stat_pct) > tolerance_pct:
        return False
    if lin_pct is None or abs(lin_pct - true_lin_pct) > tolerance_pct:
        return False
    if non_gaussian_pct is None or abs(non_gaussian_pct - true_non_gaussian_pct) > tolerance_pct:
        return False
    return True


def _expected_include_from_booleans(parsed: dict) -> bool | None:
    """Recalcula ``include`` a partir dos booleanos que o PROPRIO modelo declarou.

    Mesma formula de ``_METHOD_JSON_SCHEMA["properties"]["include"]``. Retorna
    ``None`` se algum booleano necessario nao foi declarado (nao da pra calcular).
    """
    requires_stat = parsed["algorithm_requires_stationarity"]
    requires_lin = parsed["algorithm_requires_linearity"]
    requires_non_gaussian = parsed["algorithm_requires_non_gaussian_errors"]
    majority_stat = parsed["dataset_is_majority_stationary"]
    majority_lin = parsed["dataset_is_majority_linear"]
    majority_non_gaussian = parsed["dataset_is_majority_non_gaussian"]
    if None in (
        requires_stat,
        requires_lin,
        requires_non_gaussian,
        majority_stat,
        majority_lin,
        majority_non_gaussian,
    ):
        return None
    return not (
        (requires_stat and not majority_stat)
        or (requires_lin and not majority_lin)
        or (requires_non_gaussian and not majority_non_gaussian)
    )


def _is_trustworthy(parsed: dict, profile: DatasetProfile, *, tolerance_pct: float = 5.0) -> bool:
    """Confere a resposta do chat contra fatos que o Python ja sabe com certeza.

    Duas checagens independentes, ambas sem envolver julgamento -- so aritmetica:
    (1) as porcentagens batem com o perfil real (``_percentages_match_profile``);
    (2) o ``include`` final e consistente com os proprios booleanos declarados
    (``_expected_include_from_booleans``). Uma resposta que falha aqui e
    re-tentada em vez de aceita silenciosamente (ver ``recommend_methods_via_chat``
    para o que acontece se ainda falhar apos as tentativas).
    """
    if not _percentages_match_profile(parsed, profile, tolerance_pct=tolerance_pct):
        return False
    expected_include = _expected_include_from_booleans(parsed)
    if expected_include is None:
        return False
    return parsed["include"] == expected_include


def recommend_methods_via_chat(
    profile: DatasetProfile,
    *,
    algorithms_dir: str | Path = _ALGORITHMS_DIR,
    model: str = "qwen2.5:7b",
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

    Se, apos esgotar as tentativas, os percentuais ainda batem com o perfil real
    (``_percentages_match_profile``) e todos os booleanos foram declarados, mas
    ``include`` ainda contradiz a formula sobre os PROPRIOS booleanos do modelo
    (``_expected_include_from_booleans``) -- um erro de sintese logica final, nao
    de leitura, observado sistematicamente ao acrescentar o terceiro eixo
    (``non_gaussian_errors``): o modelo lia certo os 6 booleanos mas ainda assim
    dizia ``include=false`` para um metodo cujas premissas estavam todas
    satisfeitas -- ``include`` e recalculado em Python a partir desses mesmos
    booleanos (``MethodDecision.synthesis_corrected=True``). Isso e pura
    aritmetica booleana sobre o que o modelo mesmo declarou, nao uma correcao
    usando informacao nova ou ``ground_truth``.

    ``temperature=0`` (decodificacao gulosa) foi testado e descartado: um teste
    isolado (4 metodos, 1 dataset) sugeriu que eliminava ruido de amostragem, mas
    a regressao completa (4 datasets) mostrou o oposto -- mais decisoes erradas
    (7/32) que a amostragem padrao do Ollama (2/32), porque travar na resposta
    "gulosa" do modelo tambem trava em erros deterministicos que a amostragem
    padrao as vezes evita por acaso. Mantido sem `options` (amostragem padrao).
    """
    from causal_algorithms_atlas import rag_chat

    names = _known_methods(algorithms_dir)
    decisions: list[MethodDecision] = []
    for name in names:
        prompt = _build_method_prompt(profile, algorithms_dir, name)
        retried = False
        for attempt in range(max_retries + 1):
            raw_response = rag_chat.call_ollama(prompt, model=model, format=_METHOD_JSON_SCHEMA)
            parsed = _parse_method_response(raw_response)
            if _is_trustworthy(parsed, profile) or attempt == max_retries:
                break
            retried = True

        synthesis_corrected = False
        if _percentages_match_profile(parsed, profile):
            expected_include = _expected_include_from_booleans(parsed)
            if expected_include is not None and parsed["include"] != expected_include:
                synthesis_corrected = True
                parsed = {
                    **parsed,
                    "include": expected_include,
                    "reason": (
                        f"{parsed['reason']} [corrigido: 'include' recalculado a partir dos "
                        "proprios booleanos declarados pelo modelo -- a sintese final do "
                        "modelo os contradizia]"
                    ),
                }

        decisions.append(
            MethodDecision(
                name=name,
                stationary_fraction_pct=parsed["stationary_fraction_pct"],
                dataset_is_majority_stationary=parsed["dataset_is_majority_stationary"],
                linear_fraction_pct=parsed["linear_fraction_pct"],
                dataset_is_majority_linear=parsed["dataset_is_majority_linear"],
                non_gaussian_fraction_pct=parsed["non_gaussian_fraction_pct"],
                dataset_is_majority_non_gaussian=parsed["dataset_is_majority_non_gaussian"],
                algorithm_requires_stationarity=parsed["algorithm_requires_stationarity"],
                algorithm_requires_linearity=parsed["algorithm_requires_linearity"],
                algorithm_requires_non_gaussian_errors=parsed[
                    "algorithm_requires_non_gaussian_errors"
                ],
                include=parsed["include"],
                reason=parsed["reason"],
                prompt=prompt,
                raw_response=raw_response,
                retried=retried,
                synthesis_corrected=synthesis_corrected,
            )
        )

    included = tuple(d.name for d in decisions if d.include)
    excluded = tuple(d.name for d in decisions if not d.include)
    justification = " ".join(f"{d.name}: {d.reason}" for d in decisions if d.reason)
    return ChatMethodSelection(
        included=included,
        excluded=excluded,
        justification=justification,
        decisions=tuple(decisions),
    )
