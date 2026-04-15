from __future__ import annotations

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
) -> pd.DataFrame:
    validated = validate_numeric_dataframe(data, min_rows=max_lag + 5)
    var_names = validated.columns.tolist()

    try:
        model = lingam.VARLiNGAM(lags=max_lag)
        model.fit(validated.to_numpy())
    except Exception:
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
                    }
                )

    return canonical_links_to_dataframe(records)