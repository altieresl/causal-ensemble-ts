from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt

from ..types import canonical_links_to_dataframe
from ..utils import validate_numeric_dataframe


@dataclass(frozen=True)
class _DynoTearsData:
    current: np.ndarray
    lagged_blocks: list[np.ndarray]


def _standardize_array(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (values - mean) / std


def _build_problem(data: pd.DataFrame, max_lag: int, standardize: bool) -> tuple[_DynoTearsData, list[str]]:
    validated = validate_numeric_dataframe(data, min_rows=max_lag + 10)
    values = validated.to_numpy(dtype=float)
    if standardize:
        values = _standardize_array(values)

    current = values[max_lag:]
    lagged_blocks = [values[max_lag - lag : -lag] for lag in range(1, max_lag + 1)]
    return _DynoTearsData(current=current, lagged_blocks=lagged_blocks), validated.columns.tolist()


def _unpack_doubled_variables(vector: np.ndarray, dimension: int, max_lag: int) -> tuple[np.ndarray, list[np.ndarray]]:
    single_matrix_size = dimension * dimension
    matrices = []
    offset = 0
    for _ in range(max_lag + 1):
        pos = vector[offset : offset + single_matrix_size].reshape(dimension, dimension)
        neg = vector[offset + single_matrix_size : offset + 2 * single_matrix_size].reshape(dimension, dimension)
        matrices.append(pos - neg)
        offset += 2 * single_matrix_size
    return matrices[0], matrices[1:]


def _pack_bounds(dimension: int, max_lag: int) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for lag in range(max_lag + 1):
        for component in range(2):
            for row in range(dimension):
                for col in range(dimension):
                    if lag == 0 and row == col:
                        bounds.append((0.0, 0.0))
                    else:
                        bounds.append((0.0, None))
    return bounds


def _objective_and_gradient(
    vector: np.ndarray,
    problem: _DynoTearsData,
    dimension: int,
    max_lag: int,
    lambda1: float,
    rho: float,
    alpha: float,
) -> tuple[float, np.ndarray]:
    weights, lag_weights = _unpack_doubled_variables(vector, dimension, max_lag)

    residual = problem.current - problem.current @ weights
    for lag_index, lag_matrix in enumerate(lag_weights):
        residual -= problem.lagged_blocks[lag_index] @ lag_matrix

    sample_size = problem.current.shape[0]
    loss = 0.5 / sample_size * np.square(residual).sum()

    expm_term = slin.expm(weights * weights)
    h = np.trace(expm_term) - dimension

    objective = loss + 0.5 * rho * h * h + alpha * h + lambda1 * vector.sum()

    grad_weights = -problem.current.T @ residual / sample_size
    grad_weights += (rho * h + alpha) * (expm_term.T * weights * 2.0)
    grad_lags = [-(block.T @ residual) / sample_size for block in problem.lagged_blocks]

    gradients: list[np.ndarray] = []
    for grad_matrix in [grad_weights, *grad_lags]:
        gradients.append(grad_matrix + lambda1)
        gradients.append(-grad_matrix + lambda1)

    gradient = np.concatenate([matrix.ravel() for matrix in gradients])
    return float(objective), gradient


def run_dynotears(
    data: pd.DataFrame,
    max_lag: int,
    *,
    lambda1: float = 0.01,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    w_threshold: float = 0.05,
    standardize: bool = True,
) -> pd.DataFrame:
    problem, var_names = _build_problem(data, max_lag, standardize=standardize)
    dimension = len(var_names)
    if problem.current.size == 0:
        return canonical_links_to_dataframe([])

    vector_size = 2 * (max_lag + 1) * dimension * dimension
    vector = np.zeros(vector_size, dtype=float)
    bounds = _pack_bounds(dimension, max_lag)

    rho = 1.0
    alpha = 0.0
    h_value = np.inf

    for _ in range(max_iter):

        def objective(current_vector: np.ndarray) -> tuple[float, np.ndarray]:
            return _objective_and_gradient(
                current_vector,
                problem,
                dimension,
                max_lag,
                lambda1,
                rho,
                alpha,
            )

        while rho < rho_max:
            result = sopt.minimize(objective, vector, method="L-BFGS-B", jac=True, bounds=bounds)
            vector = result.x
            weights, _ = _unpack_doubled_variables(vector, dimension, max_lag)
            expm_term = slin.expm(weights * weights)
            h_new = np.trace(expm_term) - dimension
            if h_new > 0.25 * h_value:
                rho *= 10.0
            else:
                h_value = h_new
                break

        alpha += rho * h_value
        if h_value <= h_tol or rho >= rho_max:
            break

    weights, lag_weights = _unpack_doubled_variables(vector, dimension, max_lag)
    weights[np.abs(weights) < w_threshold] = 0.0
    for lag_matrix in lag_weights:
        lag_matrix[np.abs(lag_matrix) < w_threshold] = 0.0

    records: list[dict] = []

    for source_idx, source in enumerate(var_names):
        for target_idx, target in enumerate(var_names):
            value = float(weights[source_idx, target_idx])
            if value == 0.0:
                continue
            records.append(
                {
                    "source": source,
                    "target": target,
                    "lag": 0,
                    "score": value,
                    "p_value": np.nan,
                    "method": "DYNOTEARS",
                }
            )

    for lag, lag_matrix in enumerate(lag_weights, start=1):
        for source_idx, source in enumerate(var_names):
            for target_idx, target in enumerate(var_names):
                value = float(lag_matrix[source_idx, target_idx])
                if value == 0.0:
                    continue
                records.append(
                    {
                        "source": source,
                        "target": target,
                        "lag": lag,
                        "score": value,
                        "p_value": np.nan,
                        "method": "DYNOTEARS",
                    }
                )

    return canonical_links_to_dataframe(records)
