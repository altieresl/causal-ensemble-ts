from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import lingam

from ..types import canonical_links_to_dataframe
from ..utils import validate_numeric_dataframe


def run_var_lingam(
    data: pd.DataFrame,
    max_lag: int,
    *,
    min_abs_score: float = 1e-10,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Run the lingam package implementation of VAR-LiNGAM.

    Reference: Hyvarinen, Zhang, Shimizu, and Hoyer (2010), JMLR.
    """
    validated = validate_numeric_dataframe(data, min_rows=max_lag + 5)
    var_names = validated.columns.tolist()

    try:
        model = lingam.VARLiNGAM(lags=max_lag, random_state=random_state)
        model.fit(validated.to_numpy())
    except Exception as exc:
        warnings.warn(
            f"VARLiNGAM nao convergiu ou rejeitou os dados: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return canonical_links_to_dataframe([])

    records: list[dict] = []
    for lag, adjacency in enumerate(model.adjacency_matrices_):
        for target_idx, target in enumerate(var_names):
            for source_idx, source in enumerate(var_names):
                strength = float(adjacency[target_idx, source_idx])
                if abs(strength) <= min_abs_score:
                    continue

                records.append(
                    {
                        "source": source,
                        "target": target,
                        "lag": lag,
                        "score": strength,
                        "p_value": np.nan,
                        "method": "VARLiNGAM",
                        "edge_type": "instantaneous" if lag == 0 else "lagged",
                    }
                )

    return canonical_links_to_dataframe(records)
