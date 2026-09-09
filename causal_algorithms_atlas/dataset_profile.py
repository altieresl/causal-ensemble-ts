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
            f"{self.linear_fraction:.0%} das series testadas tem relacao com o lag 1 de si "
            f"mesma e das demais variaveis aproximadamente {linearity_txt} "
            "(teste RESET de Ramsey, alfa=0.05). "
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


def _reset_p_value_for_target(
    target_column: str, numeric_data: pd.DataFrame
) -> float | None:
    """RESET de Ramsey sobre target[t] ~ const + todas_as_variaveis[t-1].

    Usar apenas a autorregressao da propria variavel (target[t] ~ target[t-1])
    perderia nao linearidade que so aparece na relacao causal entre variaveis
    diferentes -- exatamente o que importa para compatibilidade de algoritmo.
    Por isso o preditor inclui o lag 1 de todas as colunas, nao so da propria.

    ``power=3`` (nao 2): o RESET com power=2 so adiciona o quadrado do valor
    ajustado como regressor extra, o que testa apenas desvios de grau par. Uma
    nao linearidade impar e simetrica como tanh(x) nao tem termo quadratico na
    expansao de Taylor e passa completamente despercebida em power=2 mesmo com
    milhares de observacoes -- confirmado empiricamente no toy_b_nonlinear deste
    projeto (p=0.75 em power=2, p=2e-52 em power=3 para a mesma relacao X1->Y).
    power=3 adiciona tambem o cubo do valor ajustado, cobrindo esse caso.
    """
    lagged = numeric_data.shift(1).add_suffix("_lag1")
    frame = pd.concat([numeric_data[[target_column]], lagged], axis=1).dropna()
    if len(frame) < _MIN_OBSERVATIONS_FOR_RESET:
        return None

    target = frame[target_column].to_numpy(dtype=float)
    predictors = frame.drop(columns=[target_column]).to_numpy(dtype=float)
    varying_columns = np.std(predictors, axis=0) > 1e-12
    predictors = predictors[:, varying_columns]
    if predictors.shape[1] == 0 or np.std(target) < 1e-12:
        return None

    design = sm.add_constant(predictors)
    try:
        model = sm.OLS(target, design).fit()
        result = linear_reset(model, power=3, use_f=True)
        return float(result.pvalue)
    except Exception:
        return None


def profile_dataset(data: pd.DataFrame) -> DatasetProfile:
    """Extrai um perfil objetivo (estacionariedade, linearidade) de um dataset.

    Nao infere nada que nao seja diretamente testavel a partir dos dados: uma serie
    curta demais para o teste ADF ou RESET fica com o campo correspondente em None,
    em vez de assumir um valor default.
    """
    numeric_data = data.apply(pd.to_numeric, errors="coerce")

    variables: list[VariableProfile] = []
    for column in data.columns:
        series = numeric_data[column]
        adf_p_value = _adf_p_value(series)
        stationary = adf_p_value < _STATIONARITY_ALPHA if adf_p_value is not None else None
        reset_p_value = _reset_p_value_for_target(column, numeric_data)
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
