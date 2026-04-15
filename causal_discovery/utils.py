from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def validate_numeric_dataframe(data: pd.DataFrame, *, min_rows: int = 3) -> pd.DataFrame:
    if data is None or data.empty:
        raise ValueError("O DataFrame de entrada está vazio.")

    numeric_data = data.select_dtypes(include=[np.number]).copy()
    if numeric_data.shape[1] != data.shape[1]:
        dropped = sorted(set(data.columns) - set(numeric_data.columns))
        raise ValueError(f"Todas as colunas devem ser numéricas. Colunas inválidas: {dropped}")

    cleaned = numeric_data.dropna()
    if len(cleaned) < min_rows:
        raise ValueError(
            f"Dados insuficientes após remover NaNs. Linhas disponíveis: {len(cleaned)}"
        )
    return cleaned


def build_target_dataset(
    data: pd.DataFrame,
    target_col: str,
    max_lag: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_lag < 1:
        raise ValueError("max_lag deve ser maior ou igual a 1.")

    target = data[target_col].rename(target_col)
    lagged_features = []
    for lag in range(1, max_lag + 1):
        for column in data.columns:
            lagged_features.append(data[column].shift(lag).rename(format_lagged_name(column, lag)))

    design = pd.concat([target, *lagged_features], axis=1).dropna()
    return design.drop(columns=[target_col]), design[target_col]


def build_temporal_design_matrix(data: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    if max_lag < 1:
        raise ValueError("max_lag deve ser maior ou igual a 1.")

    current = data.add_suffix("__t")
    lagged = [
        data.shift(lag).rename(columns=lambda col: format_lagged_name(col, lag))
        for lag in range(1, max_lag + 1)
    ]
    return pd.concat([current, *lagged], axis=1).dropna()


def format_lagged_name(name: str, lag: int) -> str:
    return f"{name}__lag_{lag}"


def parse_lagged_name(name: str) -> Optional[tuple[str, int]]:
    if "__lag_" not in name:
        return None
    source, lag = name.rsplit("__lag_", 1)
    return source, int(lag)


def parse_current_name(name: str) -> Optional[str]:
    if not name.endswith("__t"):
        return None
    return name[: -len("__t")]


def compute_pairwise_score(source: pd.Series, target: pd.Series) -> tuple[float, float]:
    aligned = pd.concat([source, target], axis=1).dropna()
    if len(aligned) < 3:
        return np.nan, np.nan

    try:
        corr, p_value = pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    except Exception:
        return np.nan, np.nan
    return float(corr), float(p_value)


def describe_graph_edge(matrix: np.ndarray, source_idx: int, target_idx: int) -> str:
    left = matrix[source_idx, target_idx]
    right = matrix[target_idx, source_idx]
    return f"{left}/{right}"


def extract_temporal_links_from_graph(
    graph: object,
    design_matrix: pd.DataFrame,
    *,
    method: str,
) -> list[dict]:
    matrix = np.asarray(getattr(graph, "graph", graph))
    columns = list(design_matrix.columns)
    records: list[dict] = []

    for source_idx, source_column in enumerate(columns):
        lagged = parse_lagged_name(source_column)
        if lagged is None:
            continue

        source_name, lag = lagged
        for target_idx, target_column in enumerate(columns):
            target_name = parse_current_name(target_column)
            if target_name is None:
                continue

            if matrix[source_idx, target_idx] == 0 and matrix[target_idx, source_idx] == 0:
                continue

            score, p_value = compute_pairwise_score(
                design_matrix[source_column],
                design_matrix[target_column],
            )
            records.append(
                {
                    "source": source_name,
                    "target": target_name,
                    "lag": lag,
                    "score": score,
                    "p_value": p_value,
                    "method": method,
                    "edge_pattern": describe_graph_edge(matrix, source_idx, target_idx),
                }
            )

    return records