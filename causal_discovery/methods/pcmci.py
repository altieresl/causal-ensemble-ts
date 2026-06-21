from __future__ import annotations

import numpy as np
import pandas as pd
import tigramite.data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

from ..types import canonical_links_to_dataframe
from ..utils import validate_numeric_dataframe


def _is_forward_directed_edge(edge: object) -> bool:
    """Return whether Tigramite reports a definite source-to-target edge."""
    return str(edge).strip() == "-->"


def run_pcmci(
    data: pd.DataFrame,
    max_lag: int,
    *,
    pc_alpha: float = 0.05,
    alpha_level: float = 0.05,
) -> pd.DataFrame:
    """Run Tigramite PCMCI and return definite lagged directed links.

    PCMCI zero-lag links are undirected, so they are not represented as directed
    canonical edges. Reference: Runge et al. (2019), Science Advances.
    """
    if max_lag < 1:
        return canonical_links_to_dataframe([])

    validated = validate_numeric_dataframe(data, min_rows=max_lag + 5)
    var_names = validated.columns.tolist()
    dataframe = pp.DataFrame(validated.to_numpy(), var_names=var_names)

    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(),
        verbosity=0,
    )
    results = pcmci.run_pcmci(
        tau_min=1,
        tau_max=max_lag,
        pc_alpha=pc_alpha,
        alpha_level=alpha_level,
    )

    graph = results["graph"]
    val_matrix = results["val_matrix"]
    p_matrix = results.get("p_matrix")
    q_matrix = results.get("q_matrix")

    records: list[dict] = []
    for target_idx, target in enumerate(var_names):
        for source_idx, source in enumerate(var_names):
            for lag in range(1, max_lag + 1):
                edge_type = str(graph[source_idx, target_idx, lag]).strip()
                if not _is_forward_directed_edge(edge_type):
                    continue

                records.append(
                    {
                        "source": source,
                        "target": target,
                        "lag": lag,
                        "score": float(val_matrix[source_idx, target_idx, lag]),
                        "p_value": float(p_matrix[source_idx, target_idx, lag]) if p_matrix is not None else np.nan,
                        "q_value": float(q_matrix[source_idx, target_idx, lag]) if q_matrix is not None else np.nan,
                        "method": "PCMCI",
                        "edge_type": edge_type,
                    }
                )

    return canonical_links_to_dataframe(records)
