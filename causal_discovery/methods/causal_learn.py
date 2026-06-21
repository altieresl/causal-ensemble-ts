from __future__ import annotations

import contextlib
import io
import math
import threading
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..types import canonical_links_to_dataframe
from ..utils import (
    build_temporal_design_matrix,
    parse_current_name,
    parse_lagged_name,
    validate_numeric_dataframe,
)

_GES_COMPATIBILITY_LOCK = threading.Lock()


def _standardize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    std = frame.std(axis=0, ddof=0).replace(0.0, 1.0)
    return (frame - frame.mean(axis=0)) / std


def _fixed_bic_score(data, target: int, parents: list[int], parameters=None) -> float:
    """NumPy 2 compatible form of causal-learn's official local BIC score."""
    covariance, sample_size = data
    lambda_value = 0.5 if parameters is None else parameters.get("lambda_value", 0.5)
    sigma = float(covariance[target, target])
    if parents:
        target_cov = covariance[np.ix_([target], parents)]
        parent_cov = covariance[np.ix_(parents, parents)]
        parent_inverse = np.linalg.pinv(parent_cov)
        reduction = (target_cov @ parent_inverse @ target_cov.T).item()
        sigma = float(covariance[target, target] - reduction)
    sigma = max(sigma, np.finfo(float).eps)
    likelihood = -0.5 * sample_size * (1.0 + math.log(sigma))
    penalty = lambda_value * (len(parents) + 1) * math.log(sample_size)
    return float(likelihood - penalty)


_fixed_bic_score.__name__ = "local_score_BIC_from_cov"


def _endpoint_name(endpoint: object) -> str:
    return str(getattr(endpoint, "name", endpoint))


def _edge_endpoints_for_names(edge, left_name: str, right_name: str) -> tuple[str, str]:
    node1 = edge.get_node1().get_name()
    endpoint1 = _endpoint_name(edge.get_endpoint1())
    endpoint2 = _endpoint_name(edge.get_endpoint2())
    if node1 == left_name:
        return endpoint1, endpoint2
    return endpoint2, endpoint1


def _temporal_pair(left_name: str, right_name: str):
    left_lagged = parse_lagged_name(left_name)
    right_lagged = parse_lagged_name(right_name)
    left_current = parse_current_name(left_name)
    right_current = parse_current_name(right_name)

    if left_lagged is not None and right_current is not None:
        return left_lagged[0], right_current, left_lagged[1], left_name, right_name
    if right_lagged is not None and left_current is not None:
        return right_lagged[0], left_current, right_lagged[1], right_name, left_name
    return None


def _iter_temporal_edges(graph) -> Iterable[tuple[object, tuple]]:
    for edge in graph.get_graph_edges():
        left_name = edge.get_node1().get_name()
        right_name = edge.get_node2().get_name()
        temporal = _temporal_pair(left_name, right_name)
        if temporal is not None:
            yield edge, temporal


def run_score_based_ges(
    data: pd.DataFrame,
    max_lag: int,
    *,
    score_func: str = "local_score_BIC",
    max_parents: int | None = None,
    penalty_discount: float = 0.5,
    standardize: bool = True,
) -> pd.DataFrame:
    """Run causal-learn GES on a time-unrolled design matrix.

    CPDAG adjacencies between lagged and current variables are oriented by the
    known temporal order. Edges explicitly directed from current to past are
    rejected instead of reversed.
    """
    validated = validate_numeric_dataframe(data, min_rows=max_lag + 10)
    design = build_temporal_design_matrix(validated, max_lag)
    if standardize:
        design = _standardize_frame(design)

    try:
        import causallearn.search.ScoreBased.GES as ges_module
    except Exception as exc:  # pragma: no cover
        raise ImportError("Instale causal-learn para usar GES.") from exc

    # causal-learn 0.1.4.7 uses float(array) in its BIC score, which fails
    # with NumPy 2. The compatibility function uses the same score formula.
    with _GES_COMPATIBILITY_LOCK:
        globals_dict = ges_module.ges.__globals__
        original_bic = globals_dict.get("local_score_BIC_from_cov")
        try:
            if score_func in {"local_score_BIC", "local_score_BIC_from_cov"}:
                globals_dict["local_score_BIC_from_cov"] = _fixed_bic_score
            result = ges_module.ges(
                design.to_numpy(dtype=float),
                score_func=score_func,
                maxP=max_parents,
                parameters={"lambda_value": float(penalty_discount)}
                if score_func in {"local_score_BIC", "local_score_BIC_from_cov"}
                else None,
                node_names=list(design.columns),
            )
        finally:
            if original_bic is not None:
                globals_dict["local_score_BIC_from_cov"] = original_bic
    graph = result["G"]
    graph_score = float(result["score"])
    records: list[dict] = []

    for edge, temporal in _iter_temporal_edges(graph):
        source, target, lag, lagged_name, current_name = temporal
        lag_endpoint, current_endpoint = _edge_endpoints_for_names(
            edge, lagged_name, current_name
        )

        if lag_endpoint == "ARROW" and current_endpoint == "TAIL":
            continue
        if (lag_endpoint, current_endpoint) not in {
            ("TAIL", "ARROW"),
            ("TAIL", "TAIL"),
        }:
            continue

        records.append(
            {
                "source": source,
                "target": target,
                "lag": lag,
                "score": 1.0,
                "p_value": np.nan,
                "method": "GES",
                "edge_pattern": f"{lag_endpoint}-{current_endpoint}",
                "orientation_source": "cpdag"
                if current_endpoint == "ARROW"
                else "temporal_order",
                "graph_score": graph_score,
                "score_function": score_func,
            }
        )

    return canonical_links_to_dataframe(records)


def run_fci(
    data: pd.DataFrame,
    max_lag: int,
    *,
    alpha: float = 0.05,
    independence_test_method: str = "fisherz",
    depth: int = -1,
    max_path_length: int = -1,
    include_partial: bool = False,
    standardize: bool = True,
) -> pd.DataFrame:
    """Run causal-learn FCI on a time-unrolled design matrix.

    Temporal background knowledge forbids current variables from causing past
    variables. By default only definite tail-to-arrow PAG edges are returned.
    """
    validated = validate_numeric_dataframe(data, min_rows=max_lag + 10)
    design = build_temporal_design_matrix(validated, max_lag)
    if standardize:
        design = _standardize_frame(design)

    try:
        from causallearn.graph.GraphNode import GraphNode
        from causallearn.search.ConstraintBased.FCI import fci
        from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    except Exception as exc:  # pragma: no cover
        raise ImportError("Instale causal-learn para usar FCI.") from exc

    names = list(design.columns)
    current_names = [name for name in names if parse_current_name(name) is not None]
    lagged_names = [name for name in names if parse_lagged_name(name) is not None]
    background = BackgroundKnowledge()
    for current_name in current_names:
        for lagged_name in lagged_names:
            background.add_forbidden_by_node(
                GraphNode(current_name),
                GraphNode(lagged_name),
            )

    # causal-learn prints background-knowledge progress independently of verbose.
    with contextlib.redirect_stdout(io.StringIO()):
        graph, _ = fci(
            design.to_numpy(dtype=float),
            independence_test_method=independence_test_method,
            alpha=alpha,
            depth=depth,
            max_path_length=max_path_length,
            verbose=False,
            background_knowledge=background,
            show_progress=False,
            node_names=names,
        )

    records: list[dict] = []
    for edge, temporal in _iter_temporal_edges(graph):
        source, target, lag, lagged_name, current_name = temporal
        lag_endpoint, current_endpoint = _edge_endpoints_for_names(
            edge, lagged_name, current_name
        )
        definite = (lag_endpoint, current_endpoint) == ("TAIL", "ARROW")
        partial = (lag_endpoint, current_endpoint) == ("CIRCLE", "ARROW")
        if not definite and not (include_partial and partial):
            continue

        properties = sorted(
            getattr(prop, "name", str(prop)) for prop in getattr(edge, "properties", [])
        )
        records.append(
            {
                "source": source,
                "target": target,
                "lag": lag,
                "score": 1.0 if definite else 0.5,
                "p_value": np.nan,
                "method": "FCI",
                "edge_pattern": f"{lag_endpoint}-{current_endpoint}",
                "orientation_certainty": "definite" if definite else "partial",
                "edge_properties": properties,
                "possible_latent_confounding": "pl" in properties,
                "possibly_direct": "pd" in properties,
                "independence_test": independence_test_method,
            }
        )

    return canonical_links_to_dataframe(records)
