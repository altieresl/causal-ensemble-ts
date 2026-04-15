from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from ..types import canonical_links_to_dataframe
from ..utils import validate_numeric_dataframe


def run_classical_granger(
    data: pd.DataFrame,
    max_lag: int,
    *,
    alpha: float = 0.05,
    test_name: str = "ssr_ftest",
    include_self_links: bool = False,
) -> pd.DataFrame:
    validated = validate_numeric_dataframe(data, min_rows=max_lag + 5)
    records: list[dict] = []

    for target in validated.columns:
        for source in validated.columns:
            if not include_self_links and source == target:
                continue

            pair = validated[[target, source]].dropna()
            if len(pair) <= max_lag + 2:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    result = grangercausalitytests(pair, maxlag=max_lag, verbose=False)
            except Exception:
                continue

            for lag, tests in result.items():
                statistic, p_value, *_ = tests[0][test_name]
                if p_value > alpha:
                    continue

                records.append(
                    {
                        "source": source,
                        "target": target,
                        "lag": lag,
                        "score": float(statistic),
                        "p_value": float(p_value),
                        "method": "ClassicalGranger",
                    }
                )

    frame = canonical_links_to_dataframe(records)
    if frame.empty:
        return frame
    return frame.sort_values(["target", "source", "lag"]).reset_index(drop=True)