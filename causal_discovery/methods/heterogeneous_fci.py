from __future__ import annotations

import numpy as np
import pandas as pd

from ..probabilistic import combine_p_values_fisher
from ..types import canonical_links_to_dataframe
from ..utils import validate_numeric_dataframe
from .classical_granger import run_classical_granger


def _split_contiguous_segments(data: pd.DataFrame, n_segments: int) -> list[pd.DataFrame]:
    if n_segments <= 0:
        raise ValueError("n_segments deve ser maior que zero.")
    index_groups = np.array_split(np.arange(len(data)), min(n_segments, len(data)))
    return [data.iloc[indices].copy() for indices in index_groups if len(indices) > 0]


def run_heterogeneous_causal_discovery(
    data: pd.DataFrame,
    max_lag: int,
    *,
    alpha: float = 0.05,
    n_segments: int = 4,
    min_segment_votes: int = 2,
) -> pd.DataFrame:
    """Aggregate lag-specific Granger links across contiguous regimes.

    This is a project-specific segmented-consensus heuristic, not FCI.
    """
    validated = validate_numeric_dataframe(data, min_rows=max((max_lag + 5) * min_segment_votes, max_lag + 5))
    segments = [
        segment
        for segment in _split_contiguous_segments(validated, n_segments)
        if len(segment) >= max_lag + 5
    ]
    if not segments:
        return canonical_links_to_dataframe([])

    segment_results = []
    for segment_index, segment in enumerate(segments, start=1):
        result = run_classical_granger(segment, max_lag=max_lag, alpha=alpha)
        if result.empty:
            continue
        result = result.copy()
        result["segment"] = segment_index
        segment_results.append(result)

    if not segment_results:
        return canonical_links_to_dataframe([])

    combined = pd.concat(segment_results, ignore_index=True)
    summary = (
        combined.groupby(["source", "target", "lag"], as_index=False)
        .agg(
            score=("score", "mean"),
            p_value=("p_value", combine_p_values_fisher),
            segment_votes=("segment", "nunique"),
            segments=("segment", lambda values: sorted(set(values))),
        )
    )
    summary = summary[summary["segment_votes"] >= min_segment_votes].copy()
    if summary.empty:
        return canonical_links_to_dataframe([])

    summary["method"] = "HeterogeneousTemporalGranger"
    summary["support_ratio"] = summary["segment_votes"] / max(len(segments), 1)
    return canonical_links_to_dataframe(summary.to_dict("records"))


def run_heterogeneous_fci(data: pd.DataFrame, max_lag: int, **kwargs: object) -> pd.DataFrame:
    from .causal_learn import run_fci

    return run_fci(data, max_lag, **kwargs)
