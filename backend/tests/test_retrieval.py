import pytest

from backend.app.services.retrieval import expand_query, mentioned_algorithms, normalize, retrieve


@pytest.fixture(scope="module")
def atlas():
    from causal_algorithms_atlas import rag_chat
    from causal_algorithms_atlas.export import cards_to_chunks
    from causal_algorithms_atlas.loader import load_algorithm_cards

    cards = load_algorithm_cards("causal_algorithms_atlas/algorithms")
    return cards, rag_chat.build_retriever(cards_to_chunks(cards))


def test_normalize_and_expand_keep_user_terms():
    assert normalize("Estacionária") == "estacionaria"
    expanded = expand_query("Quais métodos assumem estacionariedade?")
    assert expanded.startswith("Quais métodos assumem estacionariedade?")
    assert "stationarity" in expanded


def test_portuguese_assumption_question_hits_assumption_sections(atlas):
    cards, retriever = atlas
    raw = retriever.top_k("Quais metodos assumem estacionariedade?", k=3)
    assert all(score == 0 for _, score in raw)  # sem expansao: nada casa
    results = retrieve(retriever, cards, "Quais métodos assumem estacionariedade?", k=3)
    assert all(score > 0 for _, score in results)
    assert any(chunk.section == "Assumptions" for chunk, _ in results)


def test_algorithm_named_in_question_is_prioritized(atlas):
    cards, retriever = atlas
    assert "var_lingam" in mentioned_algorithms("Quando evitar o VAR-LiNGAM?", cards)
    results = retrieve(retriever, cards, "Quando evitar o VAR-LiNGAM?", k=3)
    assert all(chunk.algorithm_id == "var_lingam" for chunk, _ in results)
    assert mentioned_algorithms("o que e causalidade?", cards) == set()
