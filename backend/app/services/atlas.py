"""Atlas de algoritmos: fichas, RAG local, experimento (perfil -> selecao -> pos-hoc) e chat.

Nada aqui consulta ground truth para decidir metodos: ele so e usado, quando existe, na avaliacao
pos-hoc do experimento (mesma disciplina de ``causal_algorithms_atlas``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..adapters.serialization import to_jsonable
from ..domain import DomainError, NotFoundError, ServiceUnavailableError

ProgressFn = Callable[[int, int, str], None]


class AtlasService:
    def __init__(self, repo_root: Path, history_path: Path) -> None:
        self._algorithms_dir = repo_root / "causal_algorithms_atlas" / "algorithms"
        self._history_path = history_path
        self._retriever = None

    # -- fichas -----------------------------------------------------------
    def _cards(self) -> dict[str, Any]:
        from causal_algorithms_atlas.loader import load_algorithm_cards

        return load_algorithm_cards(self._algorithms_dir)

    @staticmethod
    def _card_summary(card: Any) -> dict[str, Any]:
        return {
            "id": card.id,
            "name": card.name,
            "aliases": list(card.aliases),
            "family": card.family.value,
            "temporal_handling": card.temporal_handling.value,
            "output_type": card.output_type.value,
            "handles_latent_confounders": card.handles_latent_confounders,
            "handles_nonlinearity": card.handles_nonlinearity,
            "handles_contemporaneous_effects": card.handles_contemporaneous_effects,
            "implemented_in_framework": card.implemented_in_framework,
            "framework_method_name": card.framework_method_name,
            "verification": card.verification.value,
            "assumptions": [asdict(a) for a in card.assumptions],
            "references": list(card.references),
            "data_requirements": to_jsonable(card.data_requirements),
        }

    def list_algorithms(self) -> list[dict[str, Any]]:
        return [self._card_summary(card) for card in sorted(self._cards().values(), key=lambda c: c.id)]

    def get_algorithm(self, algorithm_id: str) -> dict[str, Any]:
        cards = self._cards()
        if algorithm_id not in cards:
            raise NotFoundError(f"Algoritmo '{algorithm_id}' nao existe no atlas.")
        card = cards[algorithm_id]
        return {**self._card_summary(card), "sections": dict(card.sections)}

    # -- RAG --------------------------------------------------------------
    def ask(self, query: str, k: int, generate: bool, model: str | None) -> dict[str, Any]:
        """Recupera trechos por TF-IDF e, opcionalmente, gera a resposta com o Ollama local."""
        from causal_algorithms_atlas import rag_chat
        from causal_algorithms_atlas.export import cards_to_chunks

        if self._retriever is None:
            self._retriever = rag_chat.build_retriever(cards_to_chunks(self._cards()))
        retrieved = self._retriever.top_k(query, k=k)
        answer: str | None = None
        error: str | None = None
        if generate:
            kwargs = {"model": model} if model else {}
            try:
                answer = rag_chat.call_ollama(rag_chat.build_prompt(query, retrieved), **kwargs)
            except rag_chat.RagChatError as exc:
                error = str(exc)
        return {
            "query": query,
            "retrieved": [
                {"algorithm_id": c.algorithm_id, "section": c.section, "text": c.text, "score": float(score)}
                for c, score in retrieved
            ],
            "answer": answer,
            "error": error,
        }

    # -- execucoes (roda em segundo plano) --------------------------------
    def run_experiment(self, bundle: Any, params: dict[str, Any], progress: ProgressFn) -> dict[str, Any]:
        from causal_algorithms_atlas.experiment_runner import InsufficientCandidatesError, run_experiment

        data = bundle.data
        if params.get("max_rows"):
            data = data.iloc[: int(params["max_rows"])].reset_index(drop=True)
        truth = bundle.ground_truth if not bundle.ground_truth.empty else None
        progress(0, 1, "Perfil, recomendacao e selecao")
        try:
            result = run_experiment(
                data,
                ground_truth=truth,
                dataset_name=str(params["dataset_id"]),
                candidate_method_names=set(params["methods"]) if params.get("methods") else None,
                use_assumption_soft_filter=bool(params.get("use_assumption_soft_filter", True)),
                max_lag=int(params["max_lag"]),
                n_bootstrap=int(params["n_bootstrap"]),
                max_methods=params.get("max_methods"),
                random_state=int(params.get("random_state", 42)),
                history_path=self._history_path,
                prob_threshold=float(params.get("prob_threshold", 0.5)),
                declared_causal_sufficiency=params.get("declared_causal_sufficiency"),
            )
            outcome = {"outcome": "completed", **result}
        except InsufficientCandidatesError as error:
            outcome = {"outcome": "insufficient_candidates", "message": str(error)}
        progress(1, 1, "Concluido")
        return to_jsonable(outcome)

    def run_chat_recommendation(self, bundle: Any, params: dict[str, Any], progress: ProgressFn) -> dict[str, Any]:
        """Compara a selecao do LLM local (Ollama) com o filtro estatistico deterministico."""
        from causal_algorithms_atlas import rag_chat
        from causal_algorithms_atlas.chat_recommender import recommend_methods_via_chat
        from causal_algorithms_atlas.dataset_profile import profile_dataset
        from causal_algorithms_atlas.ensemble_advisor import recommend_framework_methods

        profile = profile_dataset(bundle.data)
        recommendations = recommend_framework_methods(
            profile, declared_causal_sufficiency=params.get("declared_causal_sufficiency")
        )
        statistical = sorted(r.framework_method_name for r in recommendations if r.included)
        progress(0, 1, "Consultando o Ollama (um algoritmo por chamada)")
        kwargs = {"model": params["model"]} if params.get("model") else {}
        try:
            chat = recommend_methods_via_chat(profile, max_retries=int(params.get("max_retries", 1)), **kwargs)
        except rag_chat.RagChatError as error:
            raise ServiceUnavailableError(str(error)) from error
        progress(1, 1, "Concluido")
        chat_included = sorted(chat.included)
        return to_jsonable(
            {
                "profile_text": profile.to_query_text(),
                "statistical": {
                    "included": statistical,
                    "recommendations": [
                        {
                            "method": r.framework_method_name,
                            "included": r.included,
                            "reasons": list(r.reasons),
                        }
                        for r in recommendations
                    ],
                },
                "chat": {
                    "included": chat_included,
                    "excluded": sorted(chat.excluded),
                    "justification": chat.justification,
                    "decisions": [
                        {k: v for k, v in asdict(d).items() if k not in {"prompt", "raw_response"}}
                        for d in chat.decisions
                    ],
                },
                "agreement": {
                    "only_statistical": sorted(set(statistical) - set(chat_included)),
                    "only_chat": sorted(set(chat_included) - set(statistical)),
                    "shared": sorted(set(statistical) & set(chat_included)),
                },
            }
        )


def validate_experiment_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "max_rows": None,
        "n_bootstrap": 3,
        "max_methods": 3,
        "max_lag": 1,
        "random_state": 42,
        "prob_threshold": 0.5,
        "use_assumption_soft_filter": True,
        "declared_causal_sufficiency": None,
        "methods": None,
        **{k: v for k, v in params.items() if v is not None or k in {"max_rows", "methods", "declared_causal_sufficiency"}},
    }
    if not 1 <= normalized["n_bootstrap"] <= 100:
        raise DomainError("n_bootstrap deve estar entre 1 e 100.")
    if not 1 <= normalized["max_lag"] <= 20:
        raise DomainError("max_lag deve estar entre 1 e 20.")
    if normalized["max_methods"] is not None and not 2 <= normalized["max_methods"] <= 20:
        raise DomainError("max_methods deve estar entre 2 e 20.")
    if normalized["max_rows"] is not None and normalized["max_rows"] < 50:
        raise DomainError("max_rows deve ser pelo menos 50.")
    return normalized
