from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor

from ..types import canonical_links_to_dataframe
from ..utils import build_target_dataset, parse_lagged_name, validate_numeric_dataframe


def run_neural_granger(
    data: pd.DataFrame,
    max_lag: int,
    *,
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    max_iter: int = 200,
    perm_repeats: int = 10,
    score_threshold: float = 0.0,
) -> pd.DataFrame:
    validated = validate_numeric_dataframe(data, min_rows=max_lag + 5)
    records: list[dict] = []

    for target_name in validated.columns:
        predictors, target = build_target_dataset(validated, target_name, max_lag)
        if predictors.empty or target.empty:
            continue

        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42,
        )

        try:
            model.fit(predictors, target)
        except Exception:
            continue

        importance = permutation_importance(
            model,
            predictors,
            target,
            n_repeats=perm_repeats,
            random_state=42,
            n_jobs=-1,
        )

        for feature_name, score in zip(predictors.columns, importance.importances_mean):
            if score <= score_threshold:
                continue

            parsed = parse_lagged_name(feature_name)
            if parsed is None:
                continue

            source_name, lag = parsed
            records.append(
                {
                    "source": source_name,
                    "target": target_name,
                    "lag": lag,
                    "score": float(score),
                    "p_value": np.nan,
                    "method": "NeuralGrangerMLP",
                }
            )

    return canonical_links_to_dataframe(records)