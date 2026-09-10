from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

_STATIONARITY_ALPHA = 0.05
_MIN_OBSERVATIONS_FOR_ADF = 8
_MIN_OBSERVATIONS_FOR_EFFECT_SIZE = 40
# Calibrado empiricamente (nao e um valor convencional como o alfa=0.05 de
# significancia): em toy_a_linear (relacoes lineares conhecidas), o "chao de
# ruido" do ganho preditivo fica abaixo de 0.001. Em toy_b_nonlinear (relacoes
# tanh conhecidas via ground truth), Y e X1 -- as variaveis que de fato recebem
# uma entrada transformada por tanh -- ficam em 0.009-0.012. 0.005 separa os
# dois grupos com margem razoavel dos dois lados. Ainda e uma escolha de
# calibracao, nao uma verdade estatistica -- revisar se novos datasets
# mostrarem uma zona cinzenta diferente.
_NONLINEARITY_EFFECT_SIZE_THRESHOLD = 0.005
_N_VALIDATION_SPLITS = 4


@dataclass(frozen=True)
class VariableProfile:
    """Perfil objetivo de uma unica serie, com None quando o teste nao pode ser feito."""

    name: str
    stationary: bool | None
    adf_p_value: float | None
    linear: bool | None
    nonlinearity_effect_size: float | None


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
        # Declara as duas fracoes (estacionaria/nao, linear/nao) sempre explicitamente,
        # em vez de rotular condicionalmente uma so -- uma versao anterior imprimia
        # sempre stationary_fraction mas trocava o rotulo conforme mostly_stationary,
        # invertendo o sentido da frase sempre que a maioria era "nao estacionaria"
        # (ex.: dizia "33% sao nao estacionarias" quando na verdade eram 67%). Declarar
        # os dois lados remove essa ambiguidade tanto para leitura humana quanto para
        # um LLM que precise copiar o numero certo sem fazer a conta de "100 - X".
        return (
            f"Dataset com {self.n_variables} variaveis e {self.n_timepoints} observacoes. "
            f"{self.stationary_fraction:.0%} das series testadas sao estacionarias e "
            f"{1.0 - self.stationary_fraction:.0%} nao sao (teste ADF, alfa=0.05). "
            f"{self.linear_fraction:.0%} das series testadas tem relacao com o lag 1 de si "
            f"mesma e das demais variaveis aproximadamente linear, e "
            f"{1.0 - self.linear_fraction:.0%} nao tem (comparando erro "
            "de previsao fora da amostra entre um modelo linear e um modelo com termos "
            "quadraticos). "
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


_WINSORIZE_PERCENTILES = (1.0, 99.0)


def _winsorize(values: np.ndarray) -> np.ndarray:
    """Recorta valores fora do percentil 1-99 para o limite mais proximo.

    Series reais podem ter erros de sensor/registro (ex.: um unico valor de
    pressao atmosferica de 7679 num dataset onde o normal e ~1010, presente em
    DailyDelhiClimateTrain.csv). Um outlier assim, elevado ao cubo na expansao
    polinomial, domina a regressao inteira e produz "ganhos preditivos"
    numericamente absurdos (ja observados neste projeto: valores na casa dos
    bilhões) que nao tem nada a ver com a forma funcional real da relacao.
    """
    if values.ndim == 1:
        low, high = np.percentile(values, _WINSORIZE_PERCENTILES)
        return np.clip(values, low, high)
    low = np.percentile(values, _WINSORIZE_PERCENTILES[0], axis=0)
    high = np.percentile(values, _WINSORIZE_PERCENTILES[1], axis=0)
    return np.clip(values, low, high)


def _expanding_splits(n: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Janelas expansivas (treino cresce, validacao fica adiante) preservando a ordem temporal.

    Mesmo espirito da validacao usada em ``causal_discovery.ensemble_selection.
    add_predictive_validation_score`` (treino ate uma fracao crescente da amostra,
    validacao no trecho seguinte) -- reimplementado aqui para nao acoplar
    ``causal_algorithms_atlas`` a ``causal_discovery`` fora do modulo de ponte
    dedicado (``ensemble_advisor.py``).
    """
    validation_size = max(5, n // 8)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fraction in np.linspace(0.5, 0.8, max(1, n_splits)):
        train_end = max(10, int(n * fraction))
        validation_end = min(n, train_end + validation_size)
        if validation_end > train_end:
            splits.append((np.arange(train_end), np.arange(train_end, validation_end)))
    return splits


def _ridge_mse(
    train_features: np.ndarray,
    train_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
    *,
    alpha: float,
) -> float | None:
    """Mesma tecnica de ``causal_discovery.ensemble_selection._ridge_validation_mse``.

    Ridge (nao OLS puro) porque a expansao polinomial de grau 3 sobre todas as
    variaveis defasadas cria muitos parametros correlacionados; com amostras
    pequenas, OLS puro sobreajusta e o modelo "nao linear" pode ficar pior fora
    da amostra mesmo quando a nao linearidade e real -- confirmado empiricamente
    neste projeto (OLS puro classificava incorretamente uma autodinamica tanh
    genuina como linear em n=300 por causa disso).
    """
    feature_mean = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    train_scaled = (train_features - feature_mean) / feature_scale
    validation_scaled = (validation_features - feature_mean) / feature_scale

    target_mean = float(train_target.mean())
    try:
        system = train_scaled.T @ train_scaled + alpha * np.eye(train_scaled.shape[1])
        coefficients = np.linalg.solve(system, train_scaled.T @ (train_target - target_mean))
    except Exception:
        return None
    prediction = target_mean + validation_scaled @ coefficients
    return float(np.mean((validation_target - prediction) ** 2))


def _out_of_sample_mse(
    target: np.ndarray,
    design: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    ridge_alpha: float = 1.0,
) -> float | None:
    errors: list[float] = []
    for train_index, validation_index in splits:
        error = _ridge_mse(
            design[train_index],
            target[train_index],
            design[validation_index],
            target[validation_index],
            alpha=ridge_alpha,
        )
        if error is not None:
            errors.append(error)
    return float(np.mean(errors)) if errors else None


def _nonlinearity_effect_size_for_target(
    target_column: str, numeric_data: pd.DataFrame
) -> float | None:
    """Ganho relativo de erro fora da amostra ao permitir termos quadraticos.

    Em vez de testar "existe alguma nao linearidade detectavel" (um teste de
    significancia como RESET rejeita isso para qualquer desvio, por menor que
    seja, assim que ha dados suficientes -- confirmado empiricamente neste
    projeto: RESET rejeitava linearidade em quase todo o dataset real
    DailyDelhiClimateTrain.csv, deixando so 1 dos 8 metodos do framework como
    candidato), este teste mede se a nao linearidade e grande o bastante para
    reduzir o erro de previsao fora da amostra de forma pratica.

    Retorna ``(mse_linear - mse_quadratico) / mse_linear``: positivo e grande
    significa que o modelo quadratico prevê bem melhor fora da amostra (a
    relacao e praticamente nao linear); perto de zero ou negativo significa que
    permitir curvatura nao ajuda a prever (a relacao e praticamente linear,
    mesmo que um teste de significancia pura a rejeitasse).
    """
    lagged = numeric_data.shift(1).add_suffix("_lag1")
    frame = pd.concat([numeric_data[[target_column]], lagged], axis=1).dropna()
    if len(frame) < _MIN_OBSERVATIONS_FOR_EFFECT_SIZE:
        return None

    target = _winsorize(frame[target_column].to_numpy(dtype=float))
    predictors = _winsorize(frame.drop(columns=[target_column]).to_numpy(dtype=float))
    varying_columns = np.std(predictors, axis=0) > 1e-12
    predictors = predictors[:, varying_columns]
    if predictors.shape[1] == 0 or np.std(target) < 1e-12:
        return None

    # Padroniza os preditores ANTES de elevar ao quadrado/cubo, nao depois.
    # Variaveis em escalas bem diferentes (ex.: pressao atmosferica ~1000 vs
    # velocidade do vento ~5) produzem termos cubicos com magnitude descontrolada
    # se elevados ao cubo em escala bruta -- confirmado empiricamente neste
    # projeto: sem essa padronizacao previa, o dataset real
    # DailyDelhiClimateTrain.csv produzia "ganhos" de ate -362 (nonsense numerico),
    # nao um sinal de nao linearidade.
    predictor_mean = predictors.mean(axis=0)
    predictor_scale = predictors.std(axis=0)
    predictor_scale[predictor_scale < 1e-12] = 1.0
    standardized_predictors = (predictors - predictor_mean) / predictor_scale

    # Sem intercepto explicito: _ridge_mse centraliza features e alvo internamente.
    linear_design = standardized_predictors
    # Graus 2 e 3: uma nao linearidade impar e simetrica como tanh(x) nao tem termo
    # quadratico relevante na expansao de Taylor (mesma licao do RESET power=2 vs
    # power=3 aplicada aqui) -- so o grau 2 deixaria passar despercebida.
    nonlinear_design = np.hstack(
        [standardized_predictors, standardized_predictors**2, standardized_predictors**3]
    )

    splits = _expanding_splits(len(frame), _N_VALIDATION_SPLITS)
    if not splits:
        return None

    linear_mse = _out_of_sample_mse(target, linear_design, splits)
    nonlinear_mse = _out_of_sample_mse(target, nonlinear_design, splits)
    if linear_mse is None or nonlinear_mse is None or linear_mse <= 0.0:
        return None

    return float((linear_mse - nonlinear_mse) / linear_mse)


def profile_dataset(data: pd.DataFrame) -> DatasetProfile:
    """Extrai um perfil objetivo (estacionariedade, linearidade) de um dataset.

    Nao infere nada que nao seja diretamente testavel a partir dos dados: uma serie
    curta demais para o teste ADF ou para a comparacao preditiva fica com o campo
    correspondente em None, em vez de assumir um valor default.
    """
    numeric_data = data.apply(pd.to_numeric, errors="coerce")

    variables: list[VariableProfile] = []
    for column in data.columns:
        series = numeric_data[column]
        adf_p_value = _adf_p_value(series)
        stationary = adf_p_value < _STATIONARITY_ALPHA if adf_p_value is not None else None
        effect_size = _nonlinearity_effect_size_for_target(column, numeric_data)
        linear = (
            effect_size < _NONLINEARITY_EFFECT_SIZE_THRESHOLD
            if effect_size is not None
            else None
        )
        variables.append(
            VariableProfile(
                name=str(column),
                stationary=stationary,
                adf_p_value=adf_p_value,
                linear=linear,
                nonlinearity_effect_size=effect_size,
            )
        )

    return DatasetProfile(
        n_variables=len(data.columns),
        n_timepoints=len(data),
        variables=tuple(variables),
    )
