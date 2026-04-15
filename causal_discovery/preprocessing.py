from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


class CausalPreprocessor:
    def __init__(
        self,
        data: pd.DataFrame,
        significance_level: float = 0.05,
        decomposition_period: int | None = None,
    ) -> None:
        self.original_data = data.copy()
        self.processed_data = data.copy()
        self.significance_level = significance_level
        self.decomposition_period = decomposition_period
        self.scalers = {column: StandardScaler() for column in data.columns}
        self.differencing_orders = {column: 0 for column in data.columns}

    def _adf_p_value(self, series: pd.Series) -> float:
        cleaned = series.dropna()
        if len(cleaned) < 5 or cleaned.nunique() <= 1:
            return 1.0
        try:
            return float(adfuller(cleaned)[1])
        except Exception:
            return 1.0

    def _difference_series(self, series: pd.Series, order: int) -> pd.Series:
        differentiated = series.copy()
        for _ in range(order):
            differentiated = differentiated.diff()
        return differentiated

    def make_stationary(self, max_diffs: int = 2) -> pd.DataFrame:
        processed = self.processed_data.copy()

        for column in processed.columns:
            order = 0
            series = processed[column]
            p_value = self._adf_p_value(series)

            while p_value > self.significance_level and order < max_diffs:
                order += 1
                series = self._difference_series(processed[column], order)
                if series.dropna().empty:
                    break
                p_value = self._adf_p_value(series)

            self.differencing_orders[column] = order
            processed[column] = self._difference_series(processed[column], order)

        self.processed_data = processed.dropna()
        return self.processed_data

    def remove_trend_seasonality(self, model: str = "additive") -> pd.DataFrame:
        if self.decomposition_period is None:
            return self.processed_data

        residuals: dict[str, pd.Series] = {}
        for column in self.processed_data.columns:
            series = self.processed_data[column].dropna()
            try:
                result = seasonal_decompose(
                    series,
                    model=model,
                    period=self.decomposition_period,
                    extrapolate_trend="freq",
                )
                residuals[column] = result.resid
            except Exception:
                residuals[column] = series

        self.processed_data = pd.DataFrame(residuals).dropna()
        return self.processed_data

    def normalize_per_series(self) -> pd.DataFrame:
        normalized = self.processed_data.copy()
        for column in normalized.columns:
            values = normalized[column].to_numpy().reshape(-1, 1)
            normalized[column] = self.scalers[column].fit_transform(values)

        self.processed_data = normalized
        return self.processed_data

    def fit_transform(
        self,
        *,
        make_stationary: bool = True,
        normalize: bool = True,
        remove_trend: bool = False,
        max_diffs: int = 2,
        decomposition_model: str = "additive",
    ) -> pd.DataFrame:
        if remove_trend:
            self.remove_trend_seasonality(model=decomposition_model)
        if make_stationary:
            self.make_stationary(max_diffs=max_diffs)
        if normalize:
            self.normalize_per_series()
        return self.get_processed_data()

    def get_processed_data(self) -> pd.DataFrame:
        return self.processed_data.copy()

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "column": list(self.differencing_orders.keys()),
                "differencing_order": list(self.differencing_orders.values()),
            }
        )