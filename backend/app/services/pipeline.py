"""Pipeline robusto (ENSEMBLE_AUTO): port do ``pipeline_runner`` de Series_Temporais.ipynb.

Modulo do backend que conhece o nucleo cientifico para executar a analise principal. A selecao
de metodos/combinacao e cega ao ground truth; a validacao estrutural e a comparacao contra
algoritmos avulsos rodam somente depois (campos ``validation`` e ``comparison``).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from ..adapters.serialization import frame_records, to_jsonable
from ..domain import RunParams
from . import analysis

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


def build_selection_args(
    *,
    quick_mode: bool,
    n_bootstrap: int | None,
    parallel_jobs: int | None,
    max_lag: int,
    random_state: int,
    n_rows: int,
) -> dict[str, Any]:
    """Hiperparametros do notebook (celula 6); ``quick_mode`` limita a busca a 3 metodos."""
    default_jobs = max(1, min(4, (os.cpu_count() or 2) - 1))
    return {
        "min_methods": 2,
        "min_votes": 1,
        "n_bootstrap": int(n_bootstrap or (4 if quick_mode else 8)),
        "block_size": max(2, n_rows // 12),
        "stability_threshold": 0.6,
        "selection_probability_threshold": 0.55,
        "prior_edge_probability": 0.1,
        "posterior_weight": 0.7,
        "adaptive_method_weights": True,
        "stability_weight": 0.65,
        "local_expert_weight": 0.60,
        "predictive_validation_weight": 0.75,
        "predictive_validation_max_lag": max_lag,
        "predictive_validation_splits": 3,
        "predictive_validation_ridge_alpha": 1.0,
        "method_redundancy_penalty": 0.20,
        "method_stability_power": 1.0,
        "method_diversity_bonus": 0.15,
        "method_density_penalty": 0.5,
        "minimum_method_weight": 0.05,
        "confidence_level": 0.95,
        "random_state": random_state,
        "precompute_runs": True,
        "parallel_jobs": int(parallel_jobs or default_jobs),
        "max_bootstrap_seconds": 240 if quick_mode else 900,
    }


def restrict_relations(method: Callable, allowed: set[tuple[str, str]]) -> Callable:
    """Mantem so as arestas das relacoes pedidas, sem alterar os dados de entrada."""

    def run_restricted(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        result = method(data, **kwargs)
        if result is None or result.empty:
            return result
        keep = [(s, t) in allowed for s, t in zip(result["source"], result["target"])]
        return result.loc[keep].reset_index(drop=True)

    return run_restricted


def preprocess(
    raw: pd.DataFrame,
    *,
    make_stationary: bool = True,
    normalize: bool = True,
    decomposition_period: int | None = None,
):
    from causal_discovery import CausalPreprocessor

    preprocessor = CausalPreprocessor(raw, significance_level=0.05, decomposition_period=decomposition_period)
    processed = preprocessor.fit_transform(
        make_stationary=make_stationary, normalize=normalize, remove_trend=False, max_diffs=2
    )
    return processed, preprocessor


def run_robust_selection(
    processed: pd.DataFrame,
    method_names: Sequence[str] | None,
    *,
    quick_mode: bool,
    n_bootstrap: int | None,
    parallel_jobs: int | None,
    max_lag: int,
    random_state: int,
    expert_knowledge: list[dict[str, Any]],
    relations: Sequence[tuple[str, str]] | None,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[tuple[str, str]]]:
    """Selecao robusta de combinacoes. Devolve (selecao, args usados, metodos, relacoes)."""
    from causal_discovery import (
        get_registered_method_kwargs,
        get_registered_method_weights,
        get_registered_methods,
    )
    from causal_discovery.ensemble_selection import select_robust_ensemble_combination

    registered = get_registered_methods()
    names = list(method_names or registered)
    kwargs_all = get_registered_method_kwargs(max_lag)
    weights_all = get_registered_method_weights()

    all_pairs = [(s, t) for s in processed.columns for t in processed.columns if s != t]
    chosen = [tuple(r) for r in (relations or all_pairs)]
    restrict = set(chosen) != set(all_pairs)
    methods = {
        name: restrict_relations(registered[name], set(chosen)) if restrict else registered[name]
        for name in names
    }

    selection_args = build_selection_args(
        quick_mode=quick_mode, n_bootstrap=n_bootstrap, parallel_jobs=parallel_jobs,
        max_lag=max_lag, random_state=random_state, n_rows=len(processed),
    )
    selection_args["max_methods"] = min(3, len(names)) if quick_mode else len(names)
    selection = select_robust_ensemble_combination(
        processed,
        methods,
        method_kwargs={name: kwargs_all[name] for name in names},
        method_weights={name: weights_all[name] for name in names},
        expert_knowledge=list(expert_knowledge),
        **selection_args,
    )
    return selection, selection_args, names, chosen


def soft_voting(summary: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Consenso de precisao em segundo estagio (soft voting), como nos benchmarks do ensemble."""
    from causal_discovery import add_precision_consensus_selection

    frame = summary.copy()
    frame["ensemble_score"] = frame["pre_validation_ensemble_score"]
    return add_precision_consensus_selection(
        frame, score_threshold=threshold, method_support_threshold=SOFT_VOTING_SUPPORT_THRESHOLD
    )


def run_pipeline(dataset: Any, params: RunParams) -> dict[str, Any]:
    """Executa pre-processamento + selecao robusta e devolve um resultado JSON-ready."""
    from causal_discovery import compute_undirected_skeleton_metrics

    columns = list(params.columns or dataset.selected_columns)
    raw = dataset.data.loc[:, columns].copy()
    processed, preprocessor = preprocess(
        raw,
        make_stationary=params.make_stationary,
        normalize=params.normalize,
        decomposition_period=params.decomposition_period,
    )

    selection, selection_args, _names, relations = run_robust_selection(
        processed,
        params.methods,
        quick_mode=params.quick_mode,
        n_bootstrap=params.n_bootstrap,
        parallel_jobs=params.parallel_jobs,
        max_lag=params.max_lag,
        random_state=params.random_state,
        expert_knowledge=params.expert_knowledge,
        relations=params.selected_relations,
    )
    best = selection["best_evaluation"]
    summary = soft_voting(best["probabilistic_summary"], params.ensemble_threshold)

    stability = best["stability"]
    stable = stability.loc[stability["stability_selected"]] if "stability_selected" in stability else stability.iloc[0:0]
    consistency = best["consistency"]

    validation = comparison = panel = None
    ground_truth = dataset.ground_truth
    comparison = analysis.compare_strategies(
        selection, summary, ground_truth=ground_truth, nodes=columns,
        relations=relations, threshold=params.ensemble_threshold,
    )
    if not ground_truth.empty:
        validation = compute_undirected_skeleton_metrics(
            analysis.selected_frame(summary, params.ensemble_threshold),
            ground_truth,
            prob_threshold=0.0,
            nodes=columns,
            evaluated_relations=relations,
        )
        if params.panel_evidence:
            panel = analysis.panel_evidence(
                dataset, relations=relations, max_lag=params.panel_max_lag, ground_truth=ground_truth
            )

    combination = selection["best_combination"]
    return to_jsonable(
        {
            "columns": columns,
            "objective": params.objective,
            "best_combination": combination.split(" + ") if isinstance(combination, str) else list(combination),
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
            "comparison": comparison,
            "panel_evidence": panel,
        }
    )
