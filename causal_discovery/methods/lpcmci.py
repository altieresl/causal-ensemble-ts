from __future__ import annotations

import numpy as np
import pandas as pd
import tigramite.data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.lpcmci import LPCMCI

from ..types import canonical_links_to_dataframe
from ..utils import validate_numeric_dataframe


def _is_definite_forward_edge(edge: object) -> bool:
    """Return only fully oriented source-to-target DPAG edges."""
    return str(edge).strip() == "-->"


def run_lpcmci(
    data: pd.DataFrame,
    max_lag: int,
    *,
    pc_alpha: float = 0.05,
    n_preliminary_iterations: int = 1,
    max_cond_px: int = 0,
    max_p_global: float = float("inf"),
    max_p_non_ancestral: float = float("inf"),
    max_q_global: float = float("inf"),
    max_pds_set: int = float("inf"),
) -> pd.DataFrame:
    """Run Tigramite LPCMCI and return definite directed links.

    Ambiguous or bidirected DPAG marks are intentionally excluded because the
    canonical edge table represents a directed causal claim. Reference:
    Gerhardus and Runge (2020), NeurIPS.
    """
    validated = validate_numeric_dataframe(data, min_rows=max_lag + 5)
    var_names = validated.columns.tolist()
    dataframe = pp.DataFrame(validated.to_numpy(), var_names=var_names)

    model = LPCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(),
        verbosity=0,
    )
    results = model.run_lpcmci(
        tau_min=0,
        tau_max=max_lag,
        pc_alpha=pc_alpha,
        n_preliminary_iterations=n_preliminary_iterations,
        max_cond_px=max_cond_px,
        max_p_global=max_p_global,
        max_p_non_ancestral=max_p_non_ancestral,
        max_q_global=max_q_global,
        max_pds_set=max_pds_set,
    )

    graph = results["graph"]
    val_matrix = results["val_matrix"]
    p_matrix = results.get("p_matrix")
    q_matrix = results.get("q_matrix")

    records: list[dict] = []
    for target_idx, target in enumerate(var_names):
        for source_idx, source in enumerate(var_names):
            for lag in range(max_lag + 1):
                edge_type = str(graph[source_idx, target_idx, lag]).strip()
                if not _is_definite_forward_edge(edge_type):
                    continue

                records.append(
                    {
                        "source": source,
                        "target": target,
                        "lag": lag,
                        "score": float(val_matrix[source_idx, target_idx, lag]),
                        "p_value": float(p_matrix[source_idx, target_idx, lag]) if p_matrix is not None else np.nan,
                        "q_value": float(q_matrix[source_idx, target_idx, lag]) if q_matrix is not None else np.nan,
                        "method": "LPCMCI",
                        "edge_type": edge_type,
                    }
                )

    return canonical_links_to_dataframe(records)
