from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from causal_algorithms_atlas.dataset_profile import DatasetProfile
from causal_algorithms_atlas.loader import load_algorithm_cards
from causal_algorithms_atlas.schema import VerificationStatus

_ALGORITHMS_DIR = Path(__file__).resolve().parent / "algorithms"


@dataclass(frozen=True)
class ChatMethodSelection:
    """Selecao de metodos feita pelo chat local a partir do perfil do dataset.

    Assim como ``recommend_framework_methods``, a decisao do chat nunca consulta
    ``ground_truth`` -- ele so recebe o perfil estatistico e as premissas de cada
    ficha de algoritmo. Ao contrario daquela funcao (deterministica), esta chama um
    LLM local (Ollama), entao ``included``/``excluded`` podem divergir do filtro
    estatistico -- essa divergencia e o que se quer medir.
    """

    included: tuple[str, ...]
    excluded: tuple[str, ...]
    hallucinated: tuple[str, ...]
    justification: str
    raw_response: str
    language: str
    prompt: str


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
    """Monta o catalogo a partir das fichas reais em ``algorithms/*.md``.

    As fichas so existem em portugues -- essa e a fonte de verdade unica das
    premissas (``loader.load_algorithm_cards``), a mesma usada por
    ``recommend_framework_methods``. Para o idioma ingles, so os *rotulos* das
    secoes mudam; o conteudo (Ideia central/Premissas) nunca e duplicado ou
    traduzido a mao, para nao criar uma segunda base que possa divergir da
    ficha verificada se ela for editada.
    """
    idea_label = "Core idea" if language == "en" else "Ideia central"
    premises_label = "Assumptions" if language == "en" else "Premissas"
    cards = load_algorithm_cards(algorithms_dir)
    by_method = {c.framework_method_name: c for c in cards.values()}
    blocks = []
    for name in names:
        card = by_method[name]
        idea = card.sections.get("Ideia central", "").strip()
        premissas = card.sections.get("Premissas", "").strip()
        blocks.append(f'### "{name}"\n{idea_label}: {idea}\n{premises_label}: {premissas}')
    return "\n\n".join(blocks)


def _profile_text_en(profile: DatasetProfile) -> str:
    stationarity_txt = "stationary" if profile.mostly_stationary else "non-stationary"
    linearity_txt = "linear" if profile.mostly_linear else "non-linear"
    return (
        f"Dataset with {profile.n_variables} variables and {profile.n_timepoints} "
        f"observations. {profile.stationary_fraction:.0%} of the tested series are "
        f"{stationarity_txt} (ADF test, alpha=0.05). {profile.linear_fraction:.0%} of "
        f"the tested series have an approximately {linearity_txt} relationship with "
        "lag 1 of themselves and the other variables (comparing out-of-sample forecast "
        "error between a linear model and a model with quadratic terms). Latent "
        "confounders cannot be verified from observational data alone (an "
        "identifiability limit, not a measurement of this profile). Which causal "
        "discovery algorithms are best suited to this profile?"
    )


_INSTRUCTIONS_PT = (
    "Voce e um assistente que recomenda algoritmos de causal discovery em series "
    "temporais para compor um ensemble. Use exclusivamente o perfil do dataset e o "
    "catalogo de algoritmos abaixo -- voce nao tem acesso ao grafo causal real "
    "(ground truth) e nao deve supor nenhuma informacao alem do que foi fornecido.\n\n"
    "Perfil do dataset: {profile_text}\n\n"
    "Catalogo de algoritmos disponiveis no framework:\n{catalog_text}\n\n"
    "Tarefa: decida quais algoritmos do catalogo sao estatisticamente adequados a "
    "este perfil, respeitando as premissas obrigatorias de estacionariedade e "
    "linearidade de cada um (um algoritmo que exige estacionariedade so deve ser "
    "incluido se o perfil for predominantemente estacionario; um algoritmo que exige "
    "linearidade so deve ser incluido se o perfil for predominantemente linear). "
    "Use apenas os nomes exatos entre aspas do catalogo -- nao invente nomes.\n\n"
    "Responda SOMENTE com um objeto JSON, sem nenhum texto antes ou depois, no "
    'formato exato: {{"included": [...], "excluded": [...], "justification": "..."}}'
)

_INSTRUCTIONS_EN = (
    "You are an assistant that recommends time-series causal discovery algorithms "
    "to compose an ensemble. Use only the dataset profile and the algorithm catalog "
    "below -- you do not have access to the true causal graph (ground truth) and "
    "must not assume anything beyond what is provided.\n\n"
    "Dataset profile: {profile_text}\n\n"
    "Catalog of algorithms available in the framework:\n{catalog_text}\n\n"
    "Task: decide which algorithms in the catalog are statistically suitable for "
    "this profile, respecting each one's mandatory stationarity and linearity "
    "assumptions (an algorithm that requires stationarity should only be included "
    "if the profile is mostly stationary; an algorithm that requires linearity "
    "should only be included if the profile is mostly linear). Use only the exact "
    "quoted names from the catalog -- do not invent names.\n\n"
    "Respond ONLY with a JSON object, with no text before or after, in this exact "
    'format: {{"included": [...], "excluded": [...], "justification": "..."}}'
)


def _build_prompt(profile: DatasetProfile, algorithms_dir: str | Path, language: str) -> tuple[str, list[str]]:
    names = _known_methods(algorithms_dir)
    if language == "pt":
        profile_text = profile.to_query_text()
        template = _INSTRUCTIONS_PT
    elif language == "en":
        profile_text = _profile_text_en(profile)
        template = _INSTRUCTIONS_EN
    else:
        raise ValueError(f"Idioma nao suportado: {language!r} (use 'pt' ou 'en').")
    catalog_text = _catalog_text(algorithms_dir, names, language)
    prompt = template.format(profile_text=profile_text, catalog_text=catalog_text)
    return prompt, names


def _parse_response(raw_response: str, known_names: list[str]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], str]:
    text = raw_response.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {}

    known_set = set(known_names)
    included_raw = [str(x) for x in parsed.get("included", []) if isinstance(x, str)]
    excluded_raw = [str(x) for x in parsed.get("excluded", []) if isinstance(x, str)]
    justification = str(parsed.get("justification", ""))

    included = tuple(name for name in included_raw if name in known_set)
    hallucinated = tuple(name for name in included_raw if name not in known_set)
    excluded = tuple(name for name in excluded_raw if name in known_set)
    return included, excluded, hallucinated, justification


def recommend_methods_via_chat(
    profile: DatasetProfile,
    *,
    algorithms_dir: str | Path = _ALGORITHMS_DIR,
    model: str = "llama3.1:8b",
    language: str = "pt",
) -> ChatMethodSelection:
    """Pede ao chat local (Ollama) para selecionar algoritmos a partir do perfil.

    Diferente de ``recommend_framework_methods`` (regra deterministica sobre as
    premissas declaradas nas fichas), aqui quem decide inclusao/exclusao e o LLM,
    a partir do mesmo perfil e do mesmo catalogo de premissas -- para medir se o
    chat consegue reproduzir uma selecao estatisticamente correta. Nunca recebe
    ``ground_truth``.
    """
    from causal_algorithms_atlas import rag_chat

    prompt, known_names = _build_prompt(profile, algorithms_dir, language)
    raw_response = rag_chat.call_ollama(prompt, model=model, format="json")
    included, excluded, hallucinated, justification = _parse_response(raw_response, known_names)
    return ChatMethodSelection(
        included=included,
        excluded=excluded,
        hallucinated=hallucinated,
        justification=justification,
        raw_response=raw_response,
        language=language,
        prompt=prompt,
    )
