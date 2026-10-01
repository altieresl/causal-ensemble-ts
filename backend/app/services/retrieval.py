"""Expansão de consulta para o RAG do atlas.

As fichas misturam português (resumo) com inglês (seções e ids de premissas, ex.: ``stationarity``).
Uma pergunta em português como "quais métodos assumem estacionariedade?" não compartilha nenhum termo
com o vocabulário TF-IDF e recupera trechos ao acaso (score 0). A expansão por um glossário do domínio
é a técnica clássica de *query expansion* por tesauro (Manning, Raghavan & Schütze, *Introduction to
Information Retrieval*, 2008, cap. 9): só acrescenta termos, nunca remove os do usuário.
"""

from __future__ import annotations

import re
import unicodedata

# radical em português (sem acento) -> termos equivalentes no vocabulário das fichas
GLOSSARY: dict[str, str] = {
    "estacionar": "stationarity stationary",
    "linearidade": "linearity linear",
    "nao linear": "nonlinear nonlinearity",
    "naolinear": "nonlinear nonlinearity",
    "gaussian": "gaussian non_gaussian_errors",
    "normalidade": "gaussian non_gaussian_errors",
    "confundidor": "confundidores latent confounders causal_sufficiency",
    "confusor": "confundidores latent confounders causal_sufficiency",
    "suficiencia": "causal_sufficiency sufficiency",
    "latente": "latent latentes confounders",
    "instantane": "instantaneos contemporaneous lag",
    "simultane": "contemporaneous instantaneos",
    "fidelidade": "faithfulness",
    "markov": "markov_condition markov",
    "selecao": "selection selection_bias",
    "aciclic": "acyclicity_instantaneous acyclic",
    "deterministic": "deterministic_dynamics",
    "premissa": "assumptions",
    "pressupost": "assumptions",
    "quando usar": "when to use",
    "quando evitar": "when to avoid",
    "evitar": "avoid",
    "implementa": "implementation notes",
    "relacao com": "relationship to other methods",
    "comparad": "relationship to other methods",
    "ideia": "core idea",
    "como funciona": "core idea",
}


def normalize(text: str) -> str:
    """Minúsculas e sem acentos, para casar 'estacionária' com 'estacionar'."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def expand_query(query: str) -> str:
    plain = normalize(query)
    plain_compact = re.sub(r"\s+", " ", plain)
    extra = [terms for stem, terms in GLOSSARY.items() if stem in plain_compact]
    return " ".join([query, plain, *extra])


def _name_forms(card) -> set[str]:
    names = {card.id, card.name, *card.aliases}
    if card.framework_method_name:
        names.add(card.framework_method_name)
    forms = set()
    for name in names:
        plain = normalize(str(name))
        forms.add(plain)
        forms.add(re.sub(r"[^a-z0-9]+", "", plain))  # "VAR-LiNGAM" -> "varlingam"
    return {f for f in forms if len(f) >= 3}


def mentioned_algorithms(query: str, cards: dict) -> set[str]:
    """Ids das fichas cujo nome/alias aparece na pergunta (com e sem pontuação)."""
    plain = normalize(query)
    compact = re.sub(r"[^a-z0-9]+", "", plain)
    found = set()
    for card in cards.values():
        for form in _name_forms(card):
            if (form.isalnum() and form in compact) or form in plain:
                found.add(card.id)
                break
    return found


def retrieve(retriever, cards: dict, query: str, k: int = 4, boost: float = 0.3):
    """TF-IDF com consulta expandida + reforço para os algoritmos citados pelo nome."""
    mentioned = mentioned_algorithms(query, cards)
    candidates = retriever.top_k(expand_query(query), k=max(k * 4, 12))
    if mentioned:
        # garante que trechos do algoritmo citado entrem mesmo se o TF-IDF os deixou de fora
        extra = [(c, 0.0) for c in retriever.chunks if c.algorithm_id in mentioned]
        seen = {id(c) for c, _ in candidates}
        candidates += [(c, s) for c, s in extra if id(c) not in seen]
    rescored = [(c, float(s) + (boost if c.algorithm_id in mentioned else 0.0)) for c, s in candidates]
    rescored.sort(key=lambda pair: pair[1], reverse=True)
    return rescored[:k]
