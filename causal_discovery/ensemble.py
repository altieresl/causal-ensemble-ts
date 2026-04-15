from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd


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