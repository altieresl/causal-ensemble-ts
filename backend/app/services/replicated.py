"""Validacao estatistica com replicas independentes (port das celulas 16-19 do notebook).

Reexecuta o ENSEMBLE_AUTO em varias trajetorias independentes e aplica o protocolo pareado
(Wilcoxon + correcao de Holm + IC bootstrap + taxa de vitorias) sobre a precisao. Custo: ~5 min
por replica com a busca completa; por isso roda como execucao em segundo plano, com progresso
e cancelamento entre replicas.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from ..adapters.serialization import frame_records, to_jsonable
from ..domain import DomainError, RunCancelledError
from . import analysis
from .pipeline import preprocess, run_robust_selection, soft_voting

ProgressFn = Callable[[int, int, str], None]

DEFAULTS: dict[str, Any] = {
    "n_replicates": 10,
    "replicate_seed": 2029,
    "statistical_bootstraps": 10_000,
    "significance_level": 0.05,
    "minimum_precision_gain": 0.05,
    "minimum_win_rate": 0.70,
    "min_confirmatory_replicates": 10,
    "max_lag": 2,
    "ensemble_threshold": 0.5,
    "quick_mode": False,
    "n_bootstrap": None,
    "parallel_jobs": None,
    "parallel_replicas": None,
    "methods": None,
    "decomposition_period": None,
    "columns": None,
}
METRIC_COLUMNS = ["precision", "recall", "f1_score", "average_precision", "roc_auc"]


def normalize_params(raw: dict[str, Any]) -> dict[str, Any]:
    params = {**DEFAULTS, **{k: v for k, v in raw.items() if v is not None or k in DEFAULTS}}
    if not 2 <= params["n_replicates"] <= 100:
        raise DomainError("n_replicates deve estar entre 2 e 100.")
    if not 100 <= params["statistical_bootstraps"] <= 100_000:
        raise DomainError("statistical_bootstraps deve estar entre 100 e 100000.")
    if not 0.0 < params["significance_level"] < 1.0:
        raise DomainError("significance_level deve estar entre 0 e 1.")
    if params["methods"] is not None and len(params["methods"]) < 2:
        raise DomainError("Selecione ao menos 2 metodos.")
    return params


def plan_parallelism(params: dict[str, Any], n_replicates: int) -> tuple[int, int]:
    """(replicas simultaneas, jobs por replica): divide o orcamento de CPU sem oversubscription.

    Cada replica ja paraleliza os metodos em threads (``parallel_jobs``); rodar varias replicas
    ao mesmo tempo so ajuda se o orcamento total for dividido entre elas.
    """
    cpus = os.cpu_count() or 2
    workers = params.get("parallel_replicas") or max(1, min(3, cpus // 4))
    workers = max(1, min(int(workers), n_replicates))
    jobs = params.get("parallel_jobs") or max(1, min(4, max(1, cpus - 1) // workers))
    return workers, int(jobs)


def holm_adjust(p_values: Any) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        np.array([(len(values) - rank) * values[index] for rank, index in enumerate(order)])
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def sample_replicate_ids(trajectory_count: int, excluded: tuple[int, ...], n: int, seed: int) -> list[int]:
    available = np.setdiff1d(np.arange(trajectory_count), np.asarray(sorted(set(excluded)), dtype=int))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(available, size=min(n, len(available)), replace=False).tolist())


def _metrics_row(outcome: dict[str, Any], strategy: str, replicate_id: int, runtime: float) -> dict[str, Any]:
    binary, ranked = outcome["binary"], outcome["ranked"]
    return {
        "replicate_id": int(replicate_id),
        "strategy": strategy,
        "precision": binary["precision"],
        "recall": binary["recall"],
        "f1_score": binary["f1_score"],
        "structural_hamming_distance": binary["structural_hamming_distance"],
        "true_positives": binary["true_positives"],
        "false_positives": binary["false_positives"],
        "false_negatives": binary["false_negatives"],
        "average_precision": ranked["average_precision"],
        "roc_auc": ranked["roc_auc"],
        "runtime_seconds": float(runtime),
    }


def _baseline_rows(columns: list[str], truth: pd.DataFrame, replicate_id: int, seed: int, relations) -> list[dict]:
    from causal_discovery import (
        compute_ranked_undirected_skeleton_metrics,
        compute_undirected_skeleton_metrics,
    )

    pairs = [(columns[i], columns[j]) for i in range(len(columns)) for j in range(i + 1, len(columns))]
    all_pairs = pd.DataFrame(
        [{"source": s, "target": t, "lag": 1, "score": 1.0, "p_value": np.nan} for s, t in pairs]
    )
    all_outcome = analysis.evaluate_strategy(
        all_pairs, ground_truth=truth, nodes=columns, relations=relations,
        evidence=analysis.ranking_evidence(all_pairs),
    )
    rng = np.random.default_rng(seed + int(replicate_id))
    scores = rng.random(len(pairs))
    true_count = compute_undirected_skeleton_metrics(
        pd.DataFrame(columns=analysis.EMPTY_EDGES), truth, nodes=columns
    )["ground_truth_pairs"]
    chosen = np.argsort(scores)[-true_count:] if true_count else np.array([], dtype=int)
    random_edges = pd.DataFrame(
        [{"source": pairs[i][0], "target": pairs[i][1], "lag": 1, "score": scores[i], "p_value": np.nan} for i in chosen]
    )
    random_pair_scores = pd.DataFrame([{"source": s, "target": t, "score": v} for (s, t), v in zip(pairs, scores)])
    random_outcome = {
        "binary": compute_undirected_skeleton_metrics(random_edges, truth, nodes=columns),
        "ranked": compute_ranked_undirected_skeleton_metrics(random_pair_scores, truth),
    }
    return [
        _metrics_row(all_outcome, "ALL_PAIRS", replicate_id, 0.0),
        _metrics_row(random_outcome, "RANDOM_DENSITY", replicate_id, 0.0),
    ]


def _run_replicate(bundle: Any, replicate_id: int, params: dict[str, Any], names: list[str], columns: list[str]):
    processed, _ = preprocess(
        bundle.trajectory_frame(replicate_id),
        decomposition_period=params["decomposition_period"],
    )
    started = time.perf_counter()
    selection, _args, _names, relations = run_robust_selection(
        processed, names, quick_mode=params["quick_mode"], n_bootstrap=params["n_bootstrap"],
        parallel_jobs=params["parallel_jobs"], max_lag=params["max_lag"], random_state=42,
        expert_knowledge=[], relations=None,
    )
    runtime = time.perf_counter() - started
    truth = bundle.ground_truth

    base = analysis.base_method_outputs(selection)
    rows = []
    for name in names:
        frame = analysis.drop_self_links(base[name])
        outcome = analysis.evaluate_strategy(
            frame, ground_truth=truth, nodes=columns, relations=relations, evidence=analysis.ranking_evidence(frame)
        )
        rows.append(_metrics_row(outcome, name, replicate_id, float("nan")))

    def ensemble_row(evaluation: dict[str, Any], label: str) -> dict[str, Any]:
        summary = soft_voting(evaluation["probabilistic_summary"], params["ensemble_threshold"])
        evidence = "ensemble_score" if "ensemble_score" in summary.columns else "probability"
        outcome = analysis.evaluate_strategy(
            summary, ground_truth=truth, nodes=columns, relations=relations,
            binary_frame=analysis.selected_frame(summary, params["ensemble_threshold"]), evidence=evidence,
        )
        return _metrics_row(outcome, label, replicate_id, runtime)

    rows.append(ensemble_row(selection["best_evaluation"], "ENSEMBLE_AUTO"))
    full_key = " + ".join(names)
    if full_key in selection["all_evaluations"]:
        rows.append(ensemble_row(selection["all_evaluations"][full_key], "ENSEMBLE"))
    rows.extend(_baseline_rows(columns, truth, replicate_id, params["replicate_seed"], relations))

    combination = selection["best_combination"]
    return rows, {
        "replicate_id": int(replicate_id),
        "auto_combination": combination if isinstance(combination, str) else " + ".join(combination),
        "combinations_evaluated": int(len(selection["ranking"])),
    }


def _statistics(metrics: pd.DataFrame, names: list[str], params: dict[str, Any]) -> list[dict[str, Any]]:
    from causal_discovery import compute_paired_superiority_statistics

    if "ENSEMBLE" in set(metrics["strategy"]):
        baselines = [*names, "ENSEMBLE"]
    else:
        baselines = list(names)
    frame = metrics.rename(columns={"replicate_id": "trajectory_index"})
    rows = []
    for baseline in baselines:
        try:
            rows.append(
                compute_paired_superiority_statistics(
                    frame, candidate="ENSEMBLE_AUTO", baseline=baseline, metric="precision",
                    higher_is_better=True, n_bootstrap=params["statistical_bootstraps"], random_state=42,
                )
            )
        except ValueError as error:
            rows.append({"candidate": "ENSEMBLE_AUTO", "baseline": baseline, "error": str(error)})
    table = pd.DataFrame(rows)
    if "wilcoxon_p_value" not in table.columns:
        return frame_records(table)
    valid = table["wilcoxon_p_value"].notna()
    table["holm_p_value"] = np.nan
    table.loc[valid, "holm_p_value"] = holm_adjust(table.loc[valid, "wilcoxon_p_value"])
    table["confirmatory_sample_available"] = table["paired_trajectories"] >= params["min_confirmatory_replicates"]
    table["superiority_criterion_met"] = (
        table["confirmatory_sample_available"]
        & (table["confidence_interval_low"] > params["minimum_precision_gain"])
        & (table["holm_p_value"] < params["significance_level"])
        & (table["win_rate"] >= params["minimum_win_rate"])
    )
    return frame_records(table.sort_values("baseline"))


def run_replicated_validation(
    bundle: Any,
    excluded_trajectories: tuple[int, ...],
    raw_params: dict[str, Any],
    progress: ProgressFn,
) -> dict[str, Any]:
    from causal_discovery import get_registered_methods

    params = normalize_params(raw_params)
    if bundle.trajectory_count <= 1:
        raise DomainError("O dataset nao possui replicas independentes.")
    if bundle.ground_truth.empty:
        raise DomainError("A validacao pareada de precisao exige um grafo verdadeiro conhecido.")
    registered = list(get_registered_methods())
    names = list(params["methods"] or registered)
    unknown = sorted(set(names) - set(registered))
    if unknown:
        raise DomainError(f"Metodos desconhecidos: {unknown}")
    columns = list(bundle.selected_columns)

    ids = sample_replicate_ids(
        bundle.trajectory_count, excluded_trajectories, params["n_replicates"], params["replicate_seed"]
    )
    workers, jobs_per_replicate = plan_parallelism(params, len(ids))
    replicate_params = {**params, "parallel_jobs": jobs_per_replicate}

    def work(replicate_id: int):
        try:
            return replicate_id, _run_replicate(bundle, replicate_id, replicate_params, names, columns), None
        except Exception as error:  # noqa: BLE001 - uma replica com falha nao derruba as demais
            return replicate_id, None, {
                "replicate_id": int(replicate_id), "error_type": type(error).__name__, "error": str(error)
            }

    outcomes: dict[int, tuple[Any, Any]] = {}
    failures: list[dict[str, Any]] = []
    progress(0, len(ids), f"Iniciando {len(ids)} replicas ({workers} em paralelo, {jobs_per_replicate} jobs cada)")
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="replica")
    try:
        futures = [pool.submit(work, replicate_id) for replicate_id in ids]
        for finished, future in enumerate(as_completed(futures), start=1):
            replicate_id, outcome, failure = future.result()
            if failure is not None:
                failures.append(failure)
            else:
                outcomes[replicate_id] = outcome
            progress(finished, len(ids), f"Replicas concluidas: {finished}/{len(ids)} (ultima: {replicate_id})")
    except RunCancelledError:
        pool.shutdown(wait=False, cancel_futures=True)  # replicas em andamento terminam sozinhas
        raise
    finally:
        pool.shutdown(wait=True)
    # Ordem estavel por id de replica, independente da ordem de termino.
    metric_rows = [row for rid in sorted(outcomes) for row in outcomes[rid][0]]
    selection_rows = [outcomes[rid][1] for rid in sorted(outcomes)]
    failures.sort(key=lambda f: f["replicate_id"])
    progress(len(ids), len(ids), "Concluido")

    metrics = pd.DataFrame(metric_rows)
    descriptive: list[dict[str, Any]] = []
    comparison: list[dict[str, Any]] = []
    if not metrics.empty and "ENSEMBLE_AUTO" in set(metrics["strategy"]):
        main = metrics.loc[metrics["strategy"].isin([*names, "ENSEMBLE", "ENSEMBLE_AUTO"])]
        grouped = main.groupby("strategy")[METRIC_COLUMNS].agg(["mean", "std"])
        for strategy, row in grouped.iterrows():
            entry: dict[str, Any] = {"strategy": strategy}
            for metric in METRIC_COLUMNS:
                entry[f"{metric}_mean"] = row[(metric, "mean")]
                entry[f"{metric}_std"] = row[(metric, "std")]
            descriptive.append(entry)
        comparison = _statistics(metrics, names, params)

    return to_jsonable(
        {
            "params": params,
            "columns": columns,
            "replicate_ids": ids,
            "completed_replicates": len(selection_rows),
            "metrics": metric_rows,
            "selections": selection_rows,
            "selection_counts": (
                pd.Series([r["auto_combination"] for r in selection_rows]).value_counts().to_dict()
                if selection_rows else {}
            ),
            "failures": failures,
            "descriptive": descriptive,
            "comparison": comparison,
        }
    )
