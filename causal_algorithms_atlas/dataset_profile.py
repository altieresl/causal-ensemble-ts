from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.tsa.stattools import adfuller

_STATIONARITY_ALPHA = 0.05
_LINEARITY_ALPHA = 0.05
_MIN_OBSERVATIONS_FOR_ADF = 8
_MIN_OBSERVATIONS_FOR_RESET = 20


@dataclass(frozen=True)
class VariableProfile:
    """Perfil objetivo de uma unica serie, com None quando o teste nao pode ser feito."""

    name: str
    stationary: bool | None
    adf_p_value: float | None
    linear: bool | None
    reset_p_value: float | None


@dataclass(frozen=True)
class DatasetProfile:
    """Caracteristicas objetivas de um dataset, para filtrar algoritmos compativeis.

    Confundidores latentes deliberadamente nao aparecem aqui: nao sao verificaveis
    a partir apenas dos dados observados (limite de identificabilidade), entao esse
    perfil nao afirma nada sobre eles.
    """

    n_variables: int
    n_timepoints: int
    variables: tuple[VariableProfile, ...]

    @property
    def _testable_stationarity(self) -> list[bool]:
        return [v.stationary for v in self.variables if v.stationary is not None]

    @property
    def _testable_linearity(self) -> list[bool]:
        return [v.linear for v in self.variables if v.linear is not None]

    @property
    def stationary_fraction(self) -> float:
        values = self._testable_stationarity
        return float(np.mean(values)) if values else 0.0

    @property
    def linear_fraction(self) -> float:
        values = self._testable_linearity
        return float(np.mean(values)) if values else 0.0

    @property
    def mostly_stationary(self) -> bool:
        return self.stationary_fraction >= 0.5

    @property
    def mostly_linear(self) -> bool:
        return self.linear_fraction >= 0.5

    def to_query_text(self) -> str:
        stationarity_txt = "estacionarias" if self.mostly_stationary else "nao estacionarias"
        linearity_txt = "lineares" if self.mostly_linear else "nao lineares"
        return (
            f"Dataset com {self.n_variables} variaveis e {self.n_timepoints} observacoes. "
            f"{self.stationary_fraction:.0%} das series testadas sao {stationarity_txt} "
            "(teste ADF, alfa=0.05). "
            f"{self.linear_fraction:.0%} das series testadas tem dinamica autorregressiva "
            f"aproximadamente {linearity_txt} (teste RESET de Ramsey sobre um AR(1), alfa=0.05). "
            "Confundidores latentes nao sao verificaveis apenas a partir dos dados observados "
            "(limite de identificabilidade, nao uma medida deste perfil). "
            "Quais algoritmos de causal discovery sao mais adequados para este perfil?"
        )


def _adf_p_value(series: pd.Series) -> float | None:
    cleaned = series.dropna()
    if len(cleaned) < _MIN_OBSERVATIONS_FOR_ADF or cleaned.nunique() <= 1:
        return None
    try:
        return float(adfuller(cleaned)[1])
    except Exception:
        return None


def _reset_p_value(series: pd.Series) -> float | None:
    cleaned = series.dropna().to_numpy(dtype=float)
    if len(cleaned) < _MIN_OBSERVATIONS_FOR_RESET:
        return None
    target = cleaned[1:]
    lagged = cleaned[:-1]
    if np.std(lagged) < 1e-12 or np.std(target) < 1e-12:
        return None
    design = sm.add_constant(lagged)
    try:
        model = sm.OLS(target, design).fit()
        result = linear_reset(model, power=2, use_f=True)
        return float(result.pvalue)
    except Exception:
        return None


def profile_dataset(data: pd.DataFrame) -> DatasetProfile:
    """Extrai um perfil objetivo (estacionariedade, linearidade) de um dataset.

    Nao infere nada que nao seja diretamente testavel a partir dos dados: uma serie
    curta demais para o teste ADF ou RESET fica com o campo correspondente em None,
    em vez de assumir um valor default.
    """
    variables: list[VariableProfile] = []
    for column in data.columns:
        series = pd.to_numeric(data[column], errors="coerce")
        adf_p_value = _adf_p_value(series)
        stationary = adf_p_value < _STATIONARITY_ALPHA if adf_p_value is not None else None
        reset_p_value = _reset_p_value(series)
        linear = reset_p_value >= _LINEARITY_ALPHA if reset_p_value is not None else None
        variables.append(
            VariableProfile(
                name=str(column),
                stationary=stationary,
                adf_p_value=adf_p_value,
                linear=linear,
                reset_p_value=reset_p_value,
            )
        )

    return DatasetProfile(
        n_variables=len(data.columns),
        n_timepoints=len(data),
        variables=tuple(variables),
    )
