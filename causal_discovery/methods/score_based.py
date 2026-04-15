from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from ..types import canonical_links_to_dataframe
from ..utils import build_target_dataset, parse_lagged_name, validate_numeric_dataframe


def _compute_bic(y: pd.Series, predictors: pd.DataFrame) -> float:
    design = sm.add_constant(predictors, has_constant="add")
    model = sm.OLS(y, design).fit()
    return float(model.bic)


def run_score_based_search(
    data: pd.DataFrame,
    max_lag: int,
    *,
    min_bic_improvement: float = 1.0,
) -> pd.DataFrame:
    validated = validate_numeric_dataframe(data, min_rows=max_lag + 5)
    records: list[dict] = []

    for target_name in validated.columns:
        predictors, target = build_target_dataset(validated, target_name, max_lag)
        if predictors.empty:
            continue

        selected_features: list[str] = []
        remaining_features = list(predictors.columns)
        current_bic = _compute_bic(target, predictors.iloc[:, 0:0])

        while remaining_features:
            best_feature = None
            best_bic = current_bic

            for candidate in remaining_features:
                candidate_features = selected_features + [candidate]
                bic = _compute_bic(target, predictors[candidate_features])
                if bic < best_bic - min_bic_improvement:
                    best_bic = bic
                    best_feature = candidate

            if best_feature is None:
                break

            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            current_bic = best_bic

        if not selected_features:
            continue

        final_design = sm.add_constant(predictors[selected_features], has_constant="add")
        final_model = sm.OLS(target, final_design).fit()

        for feature_name in selected_features:
            parsed = parse_lagged_name(feature_name)
            if parsed is None:
                continue

            source_name, lag = parsed
            records.append(
                {
                    "source": source_name,
                    "target": target_name,
                    "lag": lag,
                    "score": float(final_model.params[feature_name]),
                    "p_value": float(final_model.pvalues[feature_name]),
                    "method": "ScoreBasedBIC",
                    "model_bic": float(final_model.bic),
                    "model_r2": float(final_model.rsquared),
                }
            )

    return canonical_links_to_dataframe(records)


def run_score_based_ges(data: pd.DataFrame, max_lag: int, **kwargs: object) -> pd.DataFrame:
    return run_score_based_search(data, max_lag, **kwargs)