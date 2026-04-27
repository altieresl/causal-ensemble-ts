from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from .probabilistic import (
    bayes_factor_from_p_value,
    combine_p_values_fisher,
    posterior_probability_from_bayes_factor,
    score_to_probability,
    wilson_support_interval,
)


def run_method_suite(
    data: pd.DataFrame,
    methods: Mapping[str, Callable[..., pd.DataFrame]],
    *,
    method_kwargs: Mapping[str, dict] | None = None,
) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    method_kwargs = method_kwargs or {}

    for name, method in methods.items():
        outputs[name] = method(data, **method_kwargs.get(name, {}))

    return outputs


def summarize_ensemble(results: list[pd.DataFrame], min_votes: int = 2) -> pd.DataFrame:
    non_empty = [result for result in results if result is not None and not result.empty]
    if not non_empty:
        return pd.DataFrame(columns=["source", "target", "lag", "method", "votes", "mean_score"])

    ensemble = pd.concat(non_empty, ignore_index=True)
    summary = (
        ensemble.groupby(["source", "target", "lag"], as_index=False)
        .agg(
            method=("method", list),
            votes=("method", "count"),
            mean_score=("score", "mean"),
        )
        .sort_values(["votes", "mean_score"], ascending=[False, False])
    )
    return summary[summary["votes"] >= min_votes].reset_index(drop=True)


def summarize_probabilistic_ensemble(
    results: list[pd.DataFrame],
    *,
    min_votes: int = 2,
    prior_edge_probability: float = 0.1,
    posterior_weight: float = 0.7,
    confidence_level: float = 0.95,
    method_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    non_empty = [result for result in results if result is not None and not result.empty]
    if not non_empty:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "lag",
                "method",
                "votes",
                "weighted_votes",
                "support_ratio",
                "weighted_support_ratio",
                "support_ci_low",
                "support_ci_high",
                "mean_score",
                "combined_p_value",
                "bayes_factor_10",
                "posterior_probability",
                "edge_probability",
                "uncertainty",
                "confidence",
            ]
        )

    ensemble = pd.concat(non_empty, ignore_index=True)
    method_weights = method_weights or {}

    # Pesos iguais quando nenhum peso explicito for fornecido para o metodo.
    method_weight_lookup = {
        str(method): max(float(method_weights.get(str(method), 1.0)), 0.0)
        for method in ensemble.get("method", pd.Series(dtype=object)).dropna().tolist()
    }

    methods_available = {
        str(method)
        for method in ensemble.get("method", pd.Series(dtype=object)).dropna().tolist()
    }
    total_methods = max(len(methods_available), 1)
    total_method_weight = sum(method_weight_lookup.get(name, 1.0) for name in methods_available)
    if total_method_weight <= 0.0:
        total_method_weight = float(total_methods)

    scores = pd.to_numeric(ensemble.get("score", pd.Series(dtype=float)), errors="coerce").abs()
    scale = float(scores.median()) if not scores.empty else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    posterior_weight = min(max(float(posterior_weight), 0.0), 1.0)
    rows: list[dict] = []

    grouped = ensemble.groupby(["source", "target", "lag"], as_index=False)
    for (source, target, lag), group in grouped:
        methods = sorted({str(name) for name in group.get("method", pd.Series(dtype=object)).dropna()})
        votes = len(methods)
        if votes < min_votes:
            continue

        weighted_votes = sum(method_weight_lookup.get(name, 1.0) for name in methods)
        support_ratio = votes / total_methods
        weighted_support_ratio = weighted_votes / total_method_weight
        support_ci_low, support_ci_high = wilson_support_interval(
            votes,
            total_methods,
            confidence_level=confidence_level,
        )

        mean_score = float(pd.to_numeric(group.get("score", pd.Series(dtype=float)), errors="coerce").mean())

        if "p_value" in group.columns:
            combined_p_value = combine_p_values_fisher(group["p_value"])
        else:
            combined_p_value = float("nan")

        bayes_factor_10 = bayes_factor_from_p_value(combined_p_value)
        if np.isfinite(combined_p_value):
            posterior_probability = posterior_probability_from_bayes_factor(
                bayes_factor_10,
                prior_edge_probability=prior_edge_probability,
            )
        else:
            posterior_probability = score_to_probability(mean_score, scale=scale)

        edge_probability = (
            posterior_weight * posterior_probability
            + (1.0 - posterior_weight) * weighted_support_ratio
        )
        edge_probability = min(max(float(edge_probability), 0.0), 1.0)
        uncertainty = 1.0 - edge_probability
        confidence = 1.0 - (support_ci_high - support_ci_low)

        rows.append(
            {
                "source": source,
                "target": target,
                "lag": lag,
                "method": methods,
                "votes": votes,
                "weighted_votes": float(weighted_votes),
                "support_ratio": float(support_ratio),
                "weighted_support_ratio": float(weighted_support_ratio),
                "support_ci_low": float(support_ci_low),
                "support_ci_high": float(support_ci_high),
                "mean_score": mean_score,
                "combined_p_value": float(combined_p_value),
                "bayes_factor_10": float(bayes_factor_10) if np.isfinite(bayes_factor_10) else np.nan,
                "posterior_probability": float(posterior_probability),
                "edge_probability": float(edge_probability),
                "uncertainty": float(uncertainty),
                "confidence": float(confidence),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "lag",
                "method",
                "votes",
                "weighted_votes",
                "support_ratio",
                "weighted_support_ratio",
                "support_ci_low",
                "support_ci_high",
                "mean_score",
                "combined_p_value",
                "bayes_factor_10",
                "posterior_probability",
                "edge_probability",
                "uncertainty",
                "confidence",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["edge_probability", "votes", "mean_score"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
