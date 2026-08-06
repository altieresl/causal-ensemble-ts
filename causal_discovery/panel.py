from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import tigramite.data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


PAIR_SCORE_COLUMNS = ["source", "target", "score", "p_value"]


def standardize_trajectories(
    trajectories: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Padroniza cada variavel dentro de cada trajetoria independente.

    A transformacao impede que diferencas de nivel ou escala entre trajetorias
    dominem as associacoes temporais estimadas. Ela nao concatena as fronteiras.
    """
    values = np.asarray(trajectories, dtype=float)
    if values.ndim != 3:
        raise ValueError(
            "trajectories deve ter shape (trajetorias, instantes, variaveis)."
        )
    if values.shape[0] < 1 or values.shape[1] < 2 or values.shape[2] < 2:
        raise ValueError(
            "Sao necessarias ao menos 1 trajetoria, 2 instantes e 2 variaveis."
        )
    if not np.isfinite(values).all():
        raise ValueError("trajectories contem valores ausentes ou infinitos.")

    means = values.mean(axis=1, keepdims=True)
    scales = values.std(axis=1, keepdims=True)
    safe_scales = np.where(scales > epsilon, scales, 1.0)
    return (values - means) / safe_scales


def run_pcmci_multiple_trajectories(
    trajectories: np.ndarray,
    column_names: Sequence[str],
    max_lag: int,
    *,
    pc_alpha: float = 0.05,
    alpha_level: float = 0.05,
    fdr_method: str = "none",
    standardize: bool = True,
) -> pd.DataFrame:
    """Estima escores de esqueleto com PCMCI em trajetorias independentes.

    Retorna todos os pares nao direcionados, inclusive os nao significativos. O
    ``score`` e a maior magnitude de dependencia entre as duas direcoes e lags;
    ``p_value`` e o menor p-valor correspondente. Essa tabela completa permite
    avaliar ranking (AUROC/AUPRC) sem escolher antecipadamente um limiar binario.
    """
    if max_lag < 1:
        raise ValueError("max_lag deve ser maior ou igual a 1.")

    values = np.asarray(trajectories, dtype=float)
    names = tuple(str(name) for name in column_names)
    if values.ndim != 3 or values.shape[2] != len(names):
        raise ValueError(
            "A ultima dimensao de trajectories deve coincidir com column_names."
        )
    if len(set(names)) != len(names):
        raise ValueError("column_names deve conter nomes unicos.")
    if values.shape[1] <= max_lag + 2:
        raise ValueError("Cada trajetoria e curta demais para o max_lag solicitado.")
    if standardize:
        values = standardize_trajectories(values)
    elif not np.isfinite(values).all():
        raise ValueError("trajectories contem valores ausentes ou infinitos.")

    trajectory_map = {
        trajectory_index: trajectory
        for trajectory_index, trajectory in enumerate(values)
    }
    dataframe = pp.DataFrame(
        trajectory_map,
        analysis_mode="multiple",
        var_names=list(names),
    )
    results = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(),
        verbosity=0,
    ).run_pcmci(
        tau_min=1,
        tau_max=max_lag,
        pc_alpha=pc_alpha,
        alpha_level=alpha_level,
        fdr_method=fdr_method,
    )

    val_matrix = np.asarray(results["val_matrix"], dtype=float)
    p_matrix = np.asarray(results["p_matrix"], dtype=float)
    records: list[dict[str, object]] = []
    for first_index, source in enumerate(names):
        for second_index in range(first_index + 1, len(names)):
            target = names[second_index]
            effects = np.concatenate(
                (
                    val_matrix[first_index, second_index, 1 : max_lag + 1],
                    val_matrix[second_index, first_index, 1 : max_lag + 1],
                )
            )
            p_values = np.concatenate(
                (
                    p_matrix[first_index, second_index, 1 : max_lag + 1],
                    p_matrix[second_index, first_index, 1 : max_lag + 1],
                )
            )
            records.append(
                {
                    "source": source,
                    "target": target,
                    "score": float(np.max(np.abs(effects))),
                    "p_value": float(np.min(p_values)),
                }
            )

    return pd.DataFrame(records, columns=PAIR_SCORE_COLUMNS)
