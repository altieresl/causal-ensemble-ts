from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .ensemble import run_method_suite, summarize_ensemble, summarize_probabilistic_ensemble
from .expert_knowledge import apply_expert_knowledge_to_summary, extract_method_weights

MetricScoreFn = Callable[[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame], float]


def _run_method_suite_fast(
    data: pd.DataFrame,
    methods: Mapping[str, Callable[..., pd.DataFrame]],
    *,
    method_kwargs: Mapping[str, dict] | None = None,
    parallel_jobs: int = 1,
) -> dict[str, pd.DataFrame]:
    method_kwargs = method_kwargs or {}
    if parallel_jobs <= 1 or len(methods) <= 1:
        return run_method_suite(data, methods, method_kwargs=method_kwargs)

    outputs: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(parallel_jobs))) as executor:
        futures = {
            executor.submit(method, data, **method_kwargs.get(name, {})): name
            for name, method in methods.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            outputs[name] = future.result()

    return outputs


def _empty_stability_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source",
            "target",
            "lag",
            "selected_count",
            "selection_frequency",
            "appearance_frequency",
            "mean_edge_probability",
            "mean_confidence",
            "stability_selected",
        ]
    )


def _to_edge_key(source: object, target: object, lag: object) -> tuple[str, str, int] | None:
    if pd.isna(source) or pd.isna(target) or pd.isna(lag):
        return None
    try:
        lag_int = int(lag)
    except Exception:
        return None
    return str(source), str(target), lag_int


def _extract_edge_set(frame: pd.DataFrame) -> set[tuple[str, str, int]]:
    if frame is None or frame.empty:
        return set()
    required = {"source", "target", "lag"}
    if not required.issubset(frame.columns):
        return set()

    keys: set[tuple[str, str, int]] = set()
    for _, row in frame[["source", "target", "lag"]].iterrows():
        key = _to_edge_key(row["source"], row["target"], row["lag"])
        if key is not None:
            keys.add(key)
    return keys


def compute_method_consistency(results_by_method: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    methods = list(results_by_method.keys())
    if not methods:
        return pd.DataFrame()

    edge_sets = {name: _extract_edge_set(frame) for name, frame in results_by_method.items()}
    matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)

    for left in methods:
        for right in methods:
            set_left = edge_sets[left]
            set_right = edge_sets[right]
            union = set_left | set_right
            if not union:
                score = 1.0
            else:
                score = len(set_left & set_right) / len(union)
            matrix.at[left, right] = float(score)

    return matrix


def _moving_block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)

    block = max(1, min(int(block_size), n))
    n_blocks = math.ceil(n / block)

    # Em series temporais, usamos blocos contiguos para preservar dependencia local.
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    indices = [np.arange(start, start + block) for start in starts]
    merged = np.concatenate(indices)[:n]
    return merged.astype(int)


def _bootstrap_sample(data: pd.DataFrame, *, block_size: int, rng: np.random.Generator) -> pd.DataFrame:
    indices = _moving_block_bootstrap_indices(len(data), block_size, rng)
    sampled = data.iloc[indices].copy()
    sampled.reset_index(drop=True, inplace=True)
    return sampled


def run_bootstrap_stability_selection(
    data: pd.DataFrame,
    methods: Mapping[str, Callable[..., pd.DataFrame]],
    *,
    method_kwargs: Mapping[str, dict] | None = None,
    method_weights: Mapping[str, float] | None = None,
    expert_knowledge: pd.DataFrame | list[dict] | None = None,
    precomputed_bootstrap_outputs: list[Mapping[str, pd.DataFrame]] | None = None,
    selected_method_names: list[str] | None = None,
    parallel_jobs: int = 1,
    max_bootstrap_seconds: float | None = None,
    n_bootstrap: int = 30,
    block_size: int | None = None,
    min_votes: int = 2,
    use_probabilistic: bool = True,
    selection_probability_threshold: float = 0.5,
    stability_threshold: float = 0.6,
    prior_edge_probability: float = 0.1,
    posterior_weight: float = 0.7,
    confidence_level: float = 0.95,
    random_state: int | None = 42,
) -> pd.DataFrame:
    if n_bootstrap <= 0 and not precomputed_bootstrap_outputs:
        raise ValueError("n_bootstrap deve ser maior que zero.")

    if (data is None or data.empty) and not precomputed_bootstrap_outputs:
        return _empty_stability_frame()

    method_kwargs = method_kwargs or {}
    weight_map = extract_method_weights(method_weights)
    if precomputed_bootstrap_outputs:
        iterations = list(precomputed_bootstrap_outputs)
    else:
        block = block_size if block_size is not None else max(2, len(data) // 10)
        rng = np.random.default_rng(random_state)
        iterations = []
        start_time = time.perf_counter()
        for _ in range(n_bootstrap):
            if (
                max_bootstrap_seconds is not None
                and len(iterations) > 0
                and (time.perf_counter() - start_time) >= float(max_bootstrap_seconds)
            ):
                break
            sampled_data = _bootstrap_sample(data, block_size=block, rng=rng)
            outputs = _run_method_suite_fast(
                sampled_data,
                methods,
                method_kwargs=method_kwargs,
                parallel_jobs=parallel_jobs,
            )
            iterations.append(outputs)

    if not iterations:
        return _empty_stability_frame()

    n_iterations = len(iterations)
    rng = np.random.default_rng(random_state)

    stats: dict[tuple[str, str, int], dict[str, float]] = {}

    for outputs_full in iterations:
        if selected_method_names is not None:
            outputs = {
                name: outputs_full[name]
                for name in selected_method_names
                if name in outputs_full
            }
        else:
            outputs = dict(outputs_full)

        if not outputs:
            continue

        results = list(outputs.values())

        if use_probabilistic:
            summary = summarize_probabilistic_ensemble(
                results,
                min_votes=min_votes,
                prior_edge_probability=prior_edge_probability,
                posterior_weight=posterior_weight,
                confidence_level=confidence_level,
                method_weights=weight_map,
            )
            summary = apply_expert_knowledge_to_summary(summary, expert_knowledge, hard_filter=True)
            selected = summary[summary["edge_probability"] >= selection_probability_threshold]
        else:
            summary = summarize_ensemble(results, min_votes=min_votes)
            selected = summary

        selected_keys = _extract_edge_set(selected)
        for _, row in summary.iterrows():
            key = _to_edge_key(row.get("source"), row.get("target"), row.get("lag"))
            if key is None:
                continue

            bucket = stats.setdefault(
                key,
                {
                    "selected_count": 0.0,
                    "appearance_count": 0.0,
                    "prob_sum": 0.0,
                    "prob_count": 0.0,
                    "confidence_sum": 0.0,
                    "confidence_count": 0.0,
                },
            )

            bucket["appearance_count"] += 1.0
            if key in selected_keys:
                bucket["selected_count"] += 1.0

            edge_probability = row.get("edge_probability", np.nan)
            if pd.notna(edge_probability):
                bucket["prob_sum"] += float(edge_probability)
                bucket["prob_count"] += 1.0

            confidence = row.get("confidence", np.nan)
            if pd.notna(confidence):
                bucket["confidence_sum"] += float(confidence)
                bucket["confidence_count"] += 1.0

    records: list[dict[str, Any]] = []
    for (source, target, lag), bucket in stats.items():
        selected_count = int(bucket["selected_count"])
        selection_frequency = selected_count / n_iterations
        appearance_frequency = bucket["appearance_count"] / n_iterations
        mean_edge_probability = (
            bucket["prob_sum"] / bucket["prob_count"] if bucket["prob_count"] > 0 else np.nan
        )
        mean_confidence = (
            bucket["confidence_sum"] / bucket["confidence_count"]
            if bucket["confidence_count"] > 0
            else np.nan
        )

        records.append(
            {
                "source": source,
                "target": target,
                "lag": lag,
                "selected_count": selected_count,
                "selection_frequency": float(selection_frequency),
                "appearance_frequency": float(appearance_frequency),
                "mean_edge_probability": float(mean_edge_probability)
                if pd.notna(mean_edge_probability)
                else np.nan,
                "mean_confidence": float(mean_confidence) if pd.notna(mean_confidence) else np.nan,
                "stability_selected": bool(selection_frequency >= stability_threshold),
            }
        )

    if not records:
        return _empty_stability_frame()

    return (
        pd.DataFrame(records)
        .sort_values(["selection_frequency", "appearance_frequency"], ascending=[False, False])
        .reset_index(drop=True)
    )


def _mean_upper_triangle(matrix: pd.DataFrame) -> float:
    if matrix is None or matrix.empty or len(matrix) <= 1:
        return 1.0
    values = matrix.to_numpy(dtype=float)
    upper = values[np.triu_indices_from(values, k=1)]
    if upper.size == 0:
        return 1.0
    return float(np.nanmean(upper))


def _default_performance_score(metrics: dict[str, float]) -> float:
    # Pesos priorizam estabilidade sem ignorar confianca e concordancia.
    return float(
        0.35 * metrics["mean_stability"]
        + 0.25 * metrics["mean_confidence"]
        + 0.25 * metrics["mean_edge_probability"]
        + 0.15 * metrics["mean_method_agreement"]
    )


def evaluate_method_combination(
    data: pd.DataFrame,
    methods: Mapping[str, Callable[..., pd.DataFrame]],
    *,
    method_kwargs: Mapping[str, dict] | None = None,
    method_weights: Mapping[str, float] | None = None,
    expert_knowledge: pd.DataFrame | list[dict] | None = None,
    precomputed_outputs: Mapping[str, pd.DataFrame] | None = None,
    precomputed_bootstrap_outputs: list[Mapping[str, pd.DataFrame]] | None = None,
    parallel_jobs: int = 1,
    max_bootstrap_seconds: float | None = None,
    min_votes: int = 2,
    n_bootstrap: int = 30,
    block_size: int | None = None,
    stability_threshold: float = 0.6,
    selection_probability_threshold: float = 0.5,
    prior_edge_probability: float = 0.1,
    posterior_weight: float = 0.7,
    confidence_level: float = 0.95,
    random_state: int | None = 42,
    score_fn: MetricScoreFn | None = None,
) -> dict[str, Any]:
    method_kwargs = method_kwargs or {}
    weight_map = extract_method_weights(method_weights)
    if precomputed_outputs is not None:
        outputs = dict(precomputed_outputs)
    else:
        outputs = _run_method_suite_fast(
            data,
            methods,
            method_kwargs=method_kwargs,
            parallel_jobs=parallel_jobs,
        )
    results = list(outputs.values())

    probabilistic_summary = summarize_probabilistic_ensemble(
        results,
        min_votes=min_votes,
        prior_edge_probability=prior_edge_probability,
        posterior_weight=posterior_weight,
        confidence_level=confidence_level,
        method_weights=weight_map,
    )
    probabilistic_summary = apply_expert_knowledge_to_summary(
        probabilistic_summary,
        expert_knowledge,
        hard_filter=True,
    )

    stability = run_bootstrap_stability_selection(
        data,
        methods,
        method_kwargs=method_kwargs,
        method_weights=weight_map,
        expert_knowledge=expert_knowledge,
        precomputed_bootstrap_outputs=precomputed_bootstrap_outputs,
        selected_method_names=list(methods.keys()),
        parallel_jobs=parallel_jobs,
        max_bootstrap_seconds=max_bootstrap_seconds,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        min_votes=min_votes,
        use_probabilistic=True,
        selection_probability_threshold=selection_probability_threshold,
        stability_threshold=stability_threshold,
        prior_edge_probability=prior_edge_probability,
        posterior_weight=posterior_weight,
        confidence_level=confidence_level,
        random_state=random_state,
    )

    consistency = compute_method_consistency(outputs)

    metrics: dict[str, float] = {
        "num_methods": float(len(methods)),
        "num_edges": float(len(probabilistic_summary)),
        "mean_edge_probability": float(probabilistic_summary["edge_probability"].mean())
        if not probabilistic_summary.empty
        else 0.0,
        "mean_confidence": float(probabilistic_summary["confidence"].mean())
        if not probabilistic_summary.empty
        else 0.0,
        "mean_uncertainty": float(probabilistic_summary["uncertainty"].mean())
        if not probabilistic_summary.empty
        else 1.0,
        "mean_stability": float(stability["selection_frequency"].mean()) if not stability.empty else 0.0,
        "stable_edge_ratio": float(stability["stability_selected"].mean()) if not stability.empty else 0.0,
        "mean_method_agreement": _mean_upper_triangle(consistency),
    }

    scorer = score_fn or (lambda current, *_: _default_performance_score(current))
    performance_score = float(scorer(metrics, probabilistic_summary, stability, consistency))
    metrics["performance_score"] = performance_score

    return {
        "methods": list(methods.keys()),
        "outputs": outputs,
        "probabilistic_summary": probabilistic_summary,
        "stability": stability,
        "consistency": consistency,
        "metrics": metrics,
    }


def select_robust_ensemble_combination(
    data: pd.DataFrame,
    methods: Mapping[str, Callable[..., pd.DataFrame]],
    *,
    method_kwargs: Mapping[str, dict] | None = None,
    method_weights: Mapping[str, float] | None = None,
    expert_knowledge: pd.DataFrame | list[dict] | None = None,
    precompute_runs: bool = True,
    parallel_jobs: int = 1,
    max_bootstrap_seconds: float | None = None,
    min_methods: int = 2,
    max_methods: int | None = None,
    min_votes: int = 2,
    n_bootstrap: int = 30,
    block_size: int | None = None,
    stability_threshold: float = 0.6,
    selection_probability_threshold: float = 0.5,
    prior_edge_probability: float = 0.1,
    posterior_weight: float = 0.7,
    confidence_level: float = 0.95,
    random_state: int | None = 42,
    score_fn: MetricScoreFn | None = None,
) -> dict[str, Any]:
    if not methods:
        raise ValueError("methods nao pode ser vazio.")

    names = list(methods.keys())
    max_size = len(names) if max_methods is None else min(max_methods, len(names))
    min_size = max(1, min_methods)
    if min_size > max_size:
        raise ValueError("min_methods nao pode ser maior que max_methods.")

    method_kwargs = method_kwargs or {}
    evaluations: dict[str, dict[str, Any]] = {}
    ranking_rows: list[dict[str, Any]] = []

    base_outputs_all: dict[str, pd.DataFrame] | None = None
    precomputed_bootstrap_outputs: list[dict[str, pd.DataFrame]] | None = None

    if precompute_runs:
        # Reuso de resultados evita executar métodos pesados repetidamente por combinação.
        base_outputs_all = _run_method_suite_fast(
            data,
            methods,
            method_kwargs=method_kwargs,
            parallel_jobs=parallel_jobs,
        )
        block = block_size if block_size is not None else max(2, len(data) // 10)
        rng = np.random.default_rng(random_state)
        precomputed_bootstrap_outputs = []
        start_time = time.perf_counter()
        for _ in range(n_bootstrap):
            if (
                max_bootstrap_seconds is not None
                and len(precomputed_bootstrap_outputs) > 0
                and (time.perf_counter() - start_time) >= float(max_bootstrap_seconds)
            ):
                break
            sampled_data = _bootstrap_sample(data, block_size=block, rng=rng)
            precomputed_bootstrap_outputs.append(
                _run_method_suite_fast(
                    sampled_data,
                    methods,
                    method_kwargs=method_kwargs,
                    parallel_jobs=parallel_jobs,
                )
            )

    for size in range(min_size, max_size + 1):
        for combo in combinations(names, size):
            combo_methods = {name: methods[name] for name in combo}
            combo_kwargs = {name: method_kwargs.get(name, {}) for name in combo}

            evaluation = evaluate_method_combination(
                data,
                combo_methods,
                method_kwargs=combo_kwargs,
                method_weights={name: method_weights[name] for name in combo if method_weights and name in method_weights},
                expert_knowledge=expert_knowledge,
                precomputed_outputs={name: base_outputs_all[name] for name in combo} if base_outputs_all else None,
                precomputed_bootstrap_outputs=precomputed_bootstrap_outputs,
                parallel_jobs=parallel_jobs,
                max_bootstrap_seconds=max_bootstrap_seconds,
                min_votes=min_votes,
                n_bootstrap=n_bootstrap,
                block_size=block_size,
                stability_threshold=stability_threshold,
                selection_probability_threshold=selection_probability_threshold,
                prior_edge_probability=prior_edge_probability,
                posterior_weight=posterior_weight,
                confidence_level=confidence_level,
                random_state=random_state,
                score_fn=score_fn,
            )

            key = " + ".join(combo)
            evaluations[key] = evaluation
            metrics = evaluation["metrics"]
            ranking_rows.append(
                {
                    "combination": key,
                    "num_methods": int(metrics["num_methods"]),
                    "num_edges": int(metrics["num_edges"]),
                    "mean_stability": float(metrics["mean_stability"]),
                    "stable_edge_ratio": float(metrics["stable_edge_ratio"]),
                    "mean_edge_probability": float(metrics["mean_edge_probability"]),
                    "mean_confidence": float(metrics["mean_confidence"]),
                    "mean_method_agreement": float(metrics["mean_method_agreement"]),
                    "performance_score": float(metrics["performance_score"]),
                }
            )

    ranking = pd.DataFrame(ranking_rows)
    if ranking.empty:
        raise RuntimeError("Nenhuma combinacao foi avaliada.")

    ranking = ranking.sort_values(
        ["performance_score", "mean_stability", "mean_confidence"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best_key = str(ranking.iloc[0]["combination"])
    best_eval = evaluations[best_key]

    return {
        "best_combination": best_eval["methods"],
        "best_evaluation": best_eval,
        "ranking": ranking,
        "all_evaluations": evaluations,
    }
