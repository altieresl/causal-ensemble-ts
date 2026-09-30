"""Pipeline robusto (ENSEMBLE_AUTO): port do ``pipeline_runner`` de Series_Temporais.ipynb.

Unico modulo do backend que conhece o nucleo cientifico para executar a analise.
A selecao de metodos/combinacao e cega ao ground truth; a validacao estrutural roda
somente depois, em ``validation`` (mesma disciplina do notebook).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pandas as pd

from ..adapters.serialization import frame_records, to_jsonable
from ..domain import RunParams

PipelineFn = Callable[[Any, RunParams], dict[str, Any]]

SOFT_VOTING_SUPPORT_THRESHOLD = 0.0

EDGE_COLUMNS = [
    "source", "target", "lag", "votes", "positive_votes", "negative_votes",
    "sign_consensus", "sign_agreement", "edge_probability", "ensemble_score",
    "support_ratio", "ensemble_selected", "local_expert_score", "consensus_score",
    "dominant_method", "dominant_edge_stability", "confidence", "method",
]
RANKING_COLUMNS = [
    "combination", "performance_score", "mean_stability",
    "mean_edge_probability", "mean_confidence",
]


def build_selection_args(params: RunParams, n_rows: int) -> dict[str, Any]:
    """Hiperparametros do notebook (celula 6); ``quick_mode`` limita a busca a 3 metodos."""
    default_jobs = max(1, min(4, (os.cpu_count() or 2) - 1))
    return {
        "min_methods": 2,
        "min_votes": 1,
        "n_bootstrap": int(params.n_bootstrap or (4 if params.quick_mode else 8)),
        "block_size": max(2, n_rows // 12),
        "stability_threshold": 0.6,
        "selection_probability_threshold": 0.55,
        "prior_edge_probability": 0.1,
        "posterior_weight": 0.7,
        "adaptive_method_weights": True,
        "stability_weight": 0.65,
        "local_expert_weight": 0.60,
        "predictive_validation_weight": 0.75,
        "predictive_validation_max_lag": params.max_lag,
        "predictive_validation_splits": 3,
        "predictive_validation_ridge_alpha": 1.0,
        "method_redundancy_penalty": 0.20,
        "method_stability_power": 1.0,
        "method_diversity_bonus": 0.15,
        "method_density_penalty": 0.5,
        "minimum_method_weight": 0.05,
        "confidence_level": 0.95,
        "random_state": params.random_state,
        "precompute_runs": True,
        "parallel_jobs": int(params.parallel_jobs or default_jobs),
        "max_bootstrap_seconds": 240 if params.quick_mode else 900,
    }


def _restrict_relations(method: Callable, allowed: set[tuple[str, str]]) -> Callable:
    """Mantem so as arestas das relacoes pedidas, sem alterar os dados de entrada."""

    def run_restricted(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        result = method(data, **kwargs)
        if result is None or result.empty:
            return result
        keep = [(s, t) in allowed for s, t in zip(result["source"], result["target"])]
        return result.loc[keep].reset_index(drop=True)

    return run_restricted


def run_pipeline(dataset: Any, params: RunParams) -> dict[str, Any]:
    """Executa pre-processamento + selecao robusta e devolve um resultado JSON-ready."""
    # Imports tardios: carregar o nucleo (torch, tigramite) e caro e opcional nos testes de API.
    from causal_discovery import (
        CausalPreprocessor,
        add_precision_consensus_selection,
        compute_undirected_skeleton_metrics,
        get_registered_method_kwargs,
        get_registered_method_weights,
        get_registered_methods,
    )
    from causal_discovery.ensemble_selection import select_robust_ensemble_combination

    columns = list(params.columns or dataset.selected_columns)
    raw = dataset.data.loc[:, columns].copy()
    preprocessor = CausalPreprocessor(raw, significance_level=0.05, decomposition_period=None)
    processed = preprocessor.fit_transform(
        make_stationary=params.make_stationary,
        normalize=params.normalize,
        remove_trend=False,
        max_diffs=2,
    )

    registered = get_registered_methods()
    names = list(params.methods or registered)
    kwargs_all = get_registered_method_kwargs(params.max_lag)
    weights_all = get_registered_method_weights()

    all_pairs = [(s, t) for s in processed.columns for t in processed.columns if s != t]
    relations = [tuple(r) for r in (params.selected_relations or all_pairs)]
    relation_set = set(relations)
    restrict = relation_set != set(all_pairs)
    methods = {
        name: _restrict_relations(registered[name], relation_set) if restrict else registered[name]
        for name in names
    }

    selection_args = build_selection_args(params, len(processed))
    selection_args["max_methods"] = min(3, len(names)) if params.quick_mode else len(names)
    selection = select_robust_ensemble_combination(
        processed,
        methods,
        method_kwargs={name: kwargs_all[name] for name in names},
        method_weights={name: weights_all[name] for name in names},
        expert_knowledge=list(params.expert_knowledge),
        **selection_args,
    )
    best = selection["best_evaluation"]

    summary = best["probabilistic_summary"].copy()
    summary["ensemble_score"] = summary["pre_validation_ensemble_score"]
    summary = add_precision_consensus_selection(
        summary,
        score_threshold=params.ensemble_threshold,
        method_support_threshold=SOFT_VOTING_SUPPORT_THRESHOLD,
    )

    stability = best["stability"]
    stable = stability.loc[stability["stability_selected"]] if "stability_selected" in stability else stability.iloc[0:0]
    consistency = best["consistency"]

    validation = None
    if not dataset.ground_truth.empty:
        selected_frame = summary
        if "ensemble_selected" in summary.columns:
            selected_frame = summary.loc[summary["ensemble_selected"].fillna(False).astype(bool)]
        validation = compute_undirected_skeleton_metrics(
            selected_frame,
            dataset.ground_truth,
            prob_threshold=0.0,
            nodes=columns,
            evaluated_relations=relations,
        )

    return to_jsonable(
        {
            "columns": columns,
            "best_combination": list(selection["best_combination"]) if not isinstance(selection["best_combination"], str)
            else selection["best_combination"].split(" + "),
            "edges": frame_records(summary, EDGE_COLUMNS),
            "ranking": frame_records(selection["ranking"].head(20), RANKING_COLUMNS),
            "method_weights": best["effective_method_weights"],
            "weight_diagnostics": best["method_weight_diagnostics"],
            "stable_edges": frame_records(stable),
            "consistency": {"labels": list(consistency.index), "matrix": consistency.values.tolist()},
            "metrics": best["metrics"],
            "preprocessing": preprocessor.summary(),
            "selection_args": selection_args,
            "validation": validation,
        }
    )
