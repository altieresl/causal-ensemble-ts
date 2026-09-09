from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from causal_algorithms_atlas.dataset_profile import profile_dataset
from causal_algorithms_atlas.ensemble_advisor import (
    recommend_framework_methods,
    select_candidate_methods,
)
from causal_discovery import (
    compute_undirected_skeleton_metrics,
    evaluate_method_combination,
    select_robust_ensemble_combination,
)


def _post_hoc_metrics(
    evaluation: dict[str, Any],
    ground_truth: pd.DataFrame | None,
    *,
    prob_threshold: float,
) -> dict[str, float] | None:
    if ground_truth is None:
        return None
    summary = evaluation["probabilistic_summary"]
    metrics = compute_undirected_skeleton_metrics(
        summary, ground_truth, prob_threshold=prob_threshold
    )
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "true_positives": metrics["true_positives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
    }


def run_experiment(
    data: pd.DataFrame,
    *,
    ground_truth: pd.DataFrame | None,
    dataset_name: str,
    candidate_method_names: set[str] | None = None,
    max_lag: int = 1,
    n_bootstrap: int = 10,
    min_methods: int = 2,
    max_methods: int | None = None,
    min_votes: int = 2,
    random_state: int = 42,
    history_path: str | Path | None = None,
    prob_threshold: float = 0.5,
    **selection_kwargs: Any,
) -> dict[str, Any]:
    """Roda um ciclo completo: perfilar -> recomendar -> selecionar -> avaliar.

    A escolha de qual combinacao de metodos e "melhor" vem inteiramente de
    ``select_robust_ensemble_combination``, que usa apenas metricas cegas
    (estabilidade sob bootstrap, confianca, densidade) -- nunca ``ground_truth``.
    O ground truth, quando fornecido, so e consultado DEPOIS da selecao, para
    relatar o desempenho (``*_metrics_post_hoc``). Sem essa disciplina, qualquer
    afirmacao de que o ensemble supera o melhor metodo individual seria invalida.

    ``min_methods`` comeca em 2 por padrao porque, sob ``min_votes=2`` (o padrao
    do framework para exigir concordancia entre metodos), uma "combinacao" de um
    unico metodo nunca produz nenhuma aresta -- min_votes=2 exige 2 votos, e um so
    metodo da no maximo 1. O baseline de "melhor metodo sozinho" e calculado
    separadamente aqui, reavaliando cada candidato isolado com ``min_votes=1``
    (proprio de uma leitura solo, nao de um ensemble) e reaproveitando as
    execucoes ja computadas -- sem rodar nenhum metodo de novo.

    ``candidate_method_names``, quando informado, restringe ainda mais os
    candidatos recomendados pelo perfil -- uso tipico para iteracoes rapidas com
    poucos metodos antes de escalar para o conjunto completo.
    """
    profile = profile_dataset(data)
    recommendations = recommend_framework_methods(profile)
    candidates = select_candidate_methods(recommendations)
    if candidate_method_names is not None:
        candidates = {
            name: fn for name, fn in candidates.items() if name in candidate_method_names
        }
    if not candidates:
        raise ValueError(
            "Nenhum metodo candidato disponivel para este dataset apos o filtro de perfil."
        )

    effective_max_methods = max_methods if max_methods is not None else len(candidates)
    method_kwargs = {name: {"max_lag": max_lag} for name in candidates}

    selection = select_robust_ensemble_combination(
        data,
        candidates,
        method_kwargs=method_kwargs,
        min_methods=max(2, min_methods),
        max_methods=max(effective_max_methods, max(2, min_methods)),
        min_votes=min_votes,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        **selection_kwargs,
    )
    ranking = selection["ranking"]
    best_combination = selection["best_combination"]
    best_evaluation = selection["best_evaluation"]

    base_outputs_all = selection["precomputed_outputs"]
    bootstrap_outputs_all = selection["precomputed_bootstrap_outputs"]

    single_evaluations: dict[str, dict[str, Any]] = {}
    for name, fn in candidates.items():
        single_evaluations[name] = evaluate_method_combination(
            data,
            {name: fn},
            method_kwargs={name: {"max_lag": max_lag}},
            precomputed_outputs={name: base_outputs_all[name]},
            precomputed_bootstrap_outputs=[
                {name: iteration[name]} for iteration in bootstrap_outputs_all
            ],
            min_votes=1,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )

    best_single_name = max(
        single_evaluations,
        key=lambda name: single_evaluations[name]["metrics"]["performance_score"],
    )
    best_single_evaluation = single_evaluations[best_single_name]
    best_single_performance_score = float(
        best_single_evaluation["metrics"]["performance_score"]
    )

    best_combination_metrics = _post_hoc_metrics(
        best_evaluation, ground_truth, prob_threshold=prob_threshold
    )
    best_single_metrics = (
        _post_hoc_metrics(best_single_evaluation, ground_truth, prob_threshold=prob_threshold)
        if best_single_evaluation is not None
        else None
    )

    ensemble_beats_best_single_f1 = (
        best_combination_metrics["f1_score"] >= best_single_metrics["f1_score"]
        if best_combination_metrics is not None and best_single_metrics is not None
        else None
    )

    result: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "n_variables": profile.n_variables,
        "n_timepoints": profile.n_timepoints,
        "profile": {
            "stationary_fraction": profile.stationary_fraction,
            "linear_fraction": profile.linear_fraction,
            "mostly_stationary": profile.mostly_stationary,
            "mostly_linear": profile.mostly_linear,
        },
        "recommendations": [
            {
                "framework_method_name": rec.framework_method_name,
                "algorithm_id": rec.algorithm_id,
                "included": rec.included,
                "reasons": list(rec.reasons),
            }
            for rec in recommendations
        ],
        "config": {
            "max_lag": max_lag,
            "n_bootstrap": n_bootstrap,
            "min_methods": max(2, min_methods),
            "max_methods": max(effective_max_methods, max(2, min_methods)),
            "min_votes": min_votes,
            "random_state": random_state,
            "candidate_method_names_filter": (
                sorted(candidate_method_names) if candidate_method_names is not None else None
            ),
        },
        "candidate_methods": sorted(candidates),
        "single_method_performance_scores": {
            name: float(evaluation["metrics"]["performance_score"])
            for name, evaluation in single_evaluations.items()
        },
        "ranking": ranking.to_dict(orient="records"),
        "best_combination": list(best_combination),
        "best_combination_methods": list(best_combination),
        "best_combination_performance_score": float(
            best_evaluation["metrics"]["performance_score"]
        ),
        "best_single_method": best_single_name,
        "best_single_performance_score": best_single_performance_score,
        "best_combination_metrics_post_hoc": best_combination_metrics,
        "best_single_metrics_post_hoc": best_single_metrics,
        "ensemble_beats_best_single_f1": ensemble_beats_best_single_f1,
        "ground_truth_used_only_for_post_hoc_evaluation": True,
    }

    if history_path is not None:
        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

    return result
