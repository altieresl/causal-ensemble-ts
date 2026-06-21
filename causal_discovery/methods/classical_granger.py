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
    """Run pairwise linear Granger tests and identify significant exact lags.

    The joint test first checks whether all source lags up to ``max_lag`` are
    zero. Exact lag rows are then taken from the unrestricted autoregression,
    avoiding the incorrect interpretation of a tested lag order as one exact
    causal delay.
    """
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
                    result = grangercausalitytests(pair, maxlag=[max_lag], verbose=False)
            except Exception:
                continue

            tests, fitted_models = result[max_lag]
            joint_statistic, joint_p_value, *_ = tests[test_name]
            if joint_p_value > alpha:
                continue

            unrestricted = fitted_models[1]
            source_start = max_lag
            for lag in range(1, max_lag + 1):
                parameter_index = source_start + lag - 1
                coefficient = float(unrestricted.params[parameter_index])
                p_value = float(unrestricted.pvalues[parameter_index])
                if p_value > alpha:
                    continue

                records.append(
                    {
                        "source": source,
                        "target": target,
                        "lag": lag,
                        "score": coefficient,
                        "p_value": p_value,
                        "test_statistic": float(unrestricted.tvalues[parameter_index]),
                        "joint_statistic": float(joint_statistic),
                        "joint_p_value": float(joint_p_value),
                        "lag_order": int(max_lag),
                        "method": "ClassicalGranger",
                    }
                )

    frame = canonical_links_to_dataframe(records)
    if frame.empty:
        return frame
    return frame.sort_values(["target", "source", "lag"]).reset_index(drop=True)
