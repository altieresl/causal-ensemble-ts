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
from .registry import get_signed_score_methods
from .types import validate_method_output


SIGNED_SCORE_METHODS = frozenset(
    {
        "PCMCI",
        "LPCMCI",
        "ClassicalGranger",
        "HeterogeneousTemporalGranger",
        "ScoreBasedBIC",
        "VARLiNGAM",
        "DYNOTEARS",
    }
)

SIGN_COLUMNS = [
    "signed_methods",
    "positive_votes",
    "negative_votes",
    "sign_consensus",
    "sign_agreement",
]

ENSEMBLE_COLUMNS = [
    "source",
    "target",
    "lag",
    "method",
    "votes",
    "mean_score",
    *SIGN_COLUMNS,
]

PROBABILISTIC_COLUMNS = [
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
    *SIGN_COLUMNS,
    "combined_p_value",
    "bayes_factor_10",
    "posterior_probability",
    "edge_probability",
    "uncertainty",
    "confidence",
]


def run_method_suite(
    data: pd.DataFrame,
    methods: Mapping[str, Callable[..., pd.DataFrame]],
    *,
    method_kwargs: Mapping[str, dict] | None = None,
) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    method_kwargs = method_kwargs or {}

    for name, method in methods.items():
        outputs[name] = _label_method_output(
            name,
            method(data, **method_kwargs.get(name, {})),
            data_columns=data.columns,
        )

    return outputs


def _label_method_output(
    name: str,
    output: object,
    *,
    data_columns: object | None = None,
) -> pd.DataFrame:
    """Alinha a coluna canonica ``method`` ao nome usado no registro do ensemble."""
    frame = validate_method_output(
        output,
        method_name=str(name),
        data_columns=data_columns,
    )
    frame["method"] = str(name)
    return frame


def _empty_ensemble_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ENSEMBLE_COLUMNS)


def _empty_probabilistic_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PROBABILISTIC_COLUMNS)


def _normalize_method_names(
    ensemble: pd.DataFrame,
    *,
    method_names: list[str] | None = None,
    method_weights: Mapping[str, float] | None = None,
) -> list[str]:
    names: list[str] = []
    if method_names is not None:
        names.extend(str(name) for name in method_names)
    else:
        names.extend(
            str(method)
            for method in ensemble.get("method", pd.Series(dtype=object)).dropna().tolist()
        )

    if method_weights:
        names.extend(str(name) for name in method_weights.keys())

    seen: set[str] = set()
    unique_names: list[str] = []
    for name in names:
        if name not in seen:
            unique_names.append(name)
            seen.add(name)
    return unique_names


def _deduplicate_method_edges(ensemble: pd.DataFrame) -> pd.DataFrame:
    if ensemble.empty:
        return ensemble

    frame = ensemble.copy()
    if {"source", "target"}.issubset(frame.columns):
        frame = frame.loc[frame["source"].ne(frame["target"])].copy()
    if frame.empty:
        return frame

    if "method" not in frame.columns:
        frame["method"] = "unknown"
    if "score" not in frame.columns:
        frame["score"] = np.nan
    if "p_value" not in frame.columns:
        frame["p_value"] = np.nan

    frame["method"] = frame["method"].astype(str)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["p_value"] = pd.to_numeric(frame["p_value"], errors="coerce")
    return (
        frame.groupby(["source", "target", "lag", "method"], as_index=False, dropna=False)
        .agg(
            score=("score", "mean"),
            p_value=("p_value", "min"),
        )
    )


def _summarize_score_signs(group: pd.DataFrame) -> dict[str, object]:
    methods = group.get("method", pd.Series(dtype=object)).astype(str)
    scores = pd.to_numeric(
        group.get("score", pd.Series(index=group.index, dtype=float)),
        errors="coerce",
    )
    signed_method_names = SIGNED_SCORE_METHODS | get_signed_score_methods(discover=False)
    signed = group.loc[methods.isin(signed_method_names)].copy()
    signed["_signed_score"] = scores.loc[signed.index]
    directional = signed.loc[
        np.isfinite(signed["_signed_score"]) & signed["_signed_score"].ne(0.0)
    ]

    positive_votes = int(directional["_signed_score"].gt(0.0).sum())
    negative_votes = int(directional["_signed_score"].lt(0.0).sum())
    sign_votes = positive_votes + negative_votes

    if positive_votes and negative_votes:
        sign_consensus = "mixed"
    elif positive_votes:
        sign_consensus = "positive"
    elif negative_votes:
        sign_consensus = "negative"
    else:
        sign_consensus = "unknown"

    sign_agreement = (
        max(positive_votes, negative_votes) / sign_votes
        if sign_votes
        else float("nan")
    )
    return {
        "signed_methods": sorted(set(directional["method"].astype(str))),
        "positive_votes": positive_votes,
        "negative_votes": negative_votes,
        "sign_consensus": sign_consensus,
        "sign_agreement": float(sign_agreement),
    }


def _summarize_signs_by_edge(ensemble: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (source, target, lag), group in ensemble.groupby(
        ["source", "target", "lag"],
        dropna=False,
    ):
        rows.append(
            {
                "source": source,
                "target": target,
                "lag": lag,
                **_summarize_score_signs(group),
            }
        )
    return pd.DataFrame(rows, columns=["source", "target", "lag", *SIGN_COLUMNS])


def summarize_ensemble(
    results: list[pd.DataFrame],
    min_votes: int = 2,
) -> pd.DataFrame:
    non_empty = [result for result in results if result is not None and not result.empty]
    if not non_empty:
        return _empty_ensemble_frame()

    ensemble = _deduplicate_method_edges(pd.concat(non_empty, ignore_index=True))
    if ensemble.empty:
        return _empty_ensemble_frame()

    sign_summary = _summarize_signs_by_edge(ensemble)
    summary = (
        ensemble.groupby(["source", "target", "lag"], as_index=False)
        .agg(
            method=("method", lambda values: sorted(set(values.astype(str)))),
            votes=("method", "nunique"),
            mean_score=("score", "mean"),
        )
        .merge(sign_summary, on=["source", "target", "lag"], how="left")
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
    method_names: list[str] | None = None,
) -> pd.DataFrame:
    non_empty = [result for result in results if result is not None and not result.empty]
    if not non_empty:
        return _empty_probabilistic_frame()

    ensemble = _deduplicate_method_edges(pd.concat(non_empty, ignore_index=True))
    if ensemble.empty:
        return _empty_probabilistic_frame()

    method_weights = method_weights or {}
    all_method_names = _normalize_method_names(
        ensemble,
        method_names=method_names,
        method_weights=method_weights,
    )
    if not all_method_names:
        all_method_names = _normalize_method_names(ensemble)

    # Pesos iguais quando nenhum peso explicito for fornecido para o metodo.
    method_weight_lookup = {
        name: max(float(method_weights.get(name, 1.0)), 0.0)
        for name in all_method_names
    }

    total_methods = max(len(all_method_names), 1)
    total_method_weight = sum(method_weight_lookup.get(name, 1.0) for name in all_method_names)
    if total_method_weight <= 0.0:
        total_method_weight = float(total_methods)

    score_values = pd.to_numeric(
        ensemble.get("score", pd.Series(dtype=float)),
        errors="coerce",
    ).abs()
    score_scales = (
        ensemble.assign(_abs_score=score_values)
        .groupby("method", dropna=False)["_abs_score"]
        .median()
        .to_dict()
    )

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
        sign_summary = _summarize_score_signs(group)

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
            score_probabilities: list[float] = []
            for _, method_row in group.iterrows():
                score = pd.to_numeric(method_row.get("score"), errors="coerce")
                if not np.isfinite(score):
                    continue
                scale = float(score_scales.get(method_row.get("method"), 1.0))
                if not np.isfinite(scale) or scale <= 0.0:
                    scale = 1.0
                score_probabilities.append(
                    score_to_probability(float(score), scale=scale)
                )
            posterior_probability = (
                float(np.mean(score_probabilities)) if score_probabilities else 0.5
            )

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
                **sign_summary,
                "combined_p_value": float(combined_p_value),
                "bayes_factor_10": float(bayes_factor_10) if np.isfinite(bayes_factor_10) else np.nan,
                "posterior_probability": float(posterior_probability),
                "edge_probability": float(edge_probability),
                "uncertainty": float(uncertainty),
                "confidence": float(confidence),
            }
        )

    if not rows:
        return _empty_probabilistic_frame()

    return (
        pd.DataFrame(rows)
        .sort_values(["edge_probability", "votes", "mean_score"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
