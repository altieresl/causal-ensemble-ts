"""Gera dois datasets sinteticos mais complexos que toy_a/toy_b para testar o
recomendador via chat (causal_algorithms_atlas.chat_recommender), cada um
pressionando um eixo diferente do perfil (dataset_profile.DatasetProfile):

- toy_c_nonstationary: maioria das series NAO estacionaria (raiz unitaria), mas
  todas as relacoes sao lineares. Conjunto correto (recommend_framework_methods)
  exclui os 4 metodos que exigem estacionariedade (ClassicalGranger, LPCMCI,
  NeuralGrangercMLP, PCMCI) e mantem os 4 que nao exigem (DYNOTEARS, FCI, GES,
  VARLiNGAM).
- toy_d_mixed_nonlinear: estacionario, mas com uma minoria relevante (nao
  unanime, ao contrario de toy_b) de series com relacao nao linear -- perfil
  ainda predominantemente linear, mas com percentuais bem menos limpos que
  toy_a (100%), para testar se o chat se deixa enganar pela minoria nao linear
  mencionada no perfil.

Mesmo formato de saida de synthetic_causal_datasets.ipynb: CSV de dados e CSV de
ground truth com colunas Edge/Direct/Coefficient/Lag/Type.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent


def generate_toy_c_nonstationary(T: int = 1500, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    x0 = np.zeros(T)  # random walk (raiz unitaria) -- nao estacionaria
    x2 = np.zeros(T)  # AR(1) estavel -- estacionaria
    x1 = np.zeros(T)  # depende de X0 (herda a nao estacionariedade)
    x3 = np.zeros(T)  # depende de X1 (herda a nao estacionariedade)
    x4 = np.zeros(T)  # depende de X2 (estacionaria)
    y = np.zeros(T)  # depende de X1 e X3 (nao estacionaria)

    for t in range(1, T):
        x0[t] = x0[t - 1] + rng.normal(0, 0.3)
        x2[t] = 0.3 * x2[t - 1] + rng.normal(0, 0.5)
        x1[t] = 0.6 * x0[t - 1] + rng.normal(0, 0.4)
        x3[t] = 0.5 * x1[t - 1] + rng.normal(0, 0.4)
        x4[t] = 0.4 * x2[t - 1] + rng.normal(0, 0.4)
        y[t] = 0.7 * x1[t - 1] + 0.3 * x3[t - 1] + rng.normal(0, 0.5)

    data = pd.DataFrame({"X0": x0, "X1": x1, "X2": x2, "X3": x3, "X4": x4, "Y": y})

    ground_truth = pd.DataFrame(
        [
            {"Edge": "X0 → X1", "Direct": True, "Coefficient": 0.6, "Lag": 1.0, "Type": "linear"},
            {"Edge": "X1 → X3", "Direct": True, "Coefficient": 0.5, "Lag": 1.0, "Type": "linear"},
            {"Edge": "X1 → Y", "Direct": True, "Coefficient": 0.7, "Lag": 1.0, "Type": "linear"},
            {"Edge": "X3 → Y", "Direct": True, "Coefficient": 0.3, "Lag": 1.0, "Type": "linear"},
            {"Edge": "X2 → X4", "Direct": True, "Coefficient": 0.4, "Lag": 1.0, "Type": "linear"},
            {"Edge": "X0 → Y", "Direct": False, "Coefficient": None, "Lag": None, "Type": "indirect via X1"},
            {"Edge": "X1 → Y (via X3)", "Direct": False, "Coefficient": None, "Lag": None, "Type": "indirect via X3"},
        ]
    )
    return data, ground_truth


def generate_toy_d_mixed_nonlinear(T: int = 1200, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    x0 = rng.normal(0, 1.0, T)  # ruido, sem pais -- linear/estacionario por definicao
    x2 = np.zeros(T)  # AR(1) linear -- estacionaria e linear
    x4 = np.zeros(T)  # AR(1) linear -- estacionaria e linear
    x1 = np.zeros(T)  # nao linear (funcao quadratica suave de X4)
    x3 = np.zeros(T)  # nao linear (produto X1*X2, interacao)
    y = np.zeros(T)  # linear em X1 e X3 (a nao linearidade ja esta upstream)

    for t in range(1, T):
        x2[t] = 0.5 * x2[t - 1] + rng.normal(0, 0.5)
        x4[t] = 0.5 * x4[t - 1] + rng.normal(0, 0.5)
        x1[t] = 0.5 * np.tanh(1.2 * x4[t - 1]) + rng.normal(0, 0.4)
        x3[t] = 0.35 * x1[t - 1] * x2[t - 1] + rng.normal(0, 0.4)
        y[t] = 0.6 * x1[t - 1] + 0.4 * x3[t - 1] + rng.normal(0, 0.5)

    data = pd.DataFrame({"X0": x0, "X1": x1, "X2": x2, "X3": x3, "X4": x4, "Y": y})

    ground_truth = pd.DataFrame(
        [
            {"Edge": "X4 → X1", "Direct": True, "Coefficient": 0.5, "Lag": 1.0, "Type": "nonlinear (tanh)"},
            {"Edge": "X1 → X3", "Direct": True, "Coefficient": 0.35, "Lag": 1.0, "Type": "nonlinear (interaction)"},
            {"Edge": "X2 → X3", "Direct": True, "Coefficient": 0.35, "Lag": 1.0, "Type": "nonlinear (interaction)"},
            {"Edge": "X1 → Y", "Direct": True, "Coefficient": 0.6, "Lag": 1.0, "Type": "linear"},
            {"Edge": "X3 → Y", "Direct": True, "Coefficient": 0.4, "Lag": 1.0, "Type": "linear"},
            {"Edge": "X0 → Y", "Direct": False, "Coefficient": 0.0, "Lag": None, "Type": "none (noise)"},
        ]
    )
    return data, ground_truth


if __name__ == "__main__":
    data_c, gt_c = generate_toy_c_nonstationary()
    data_c.to_csv(OUT_DIR / "toy_c_nonstationary.csv", index=False)
    gt_c.to_csv(OUT_DIR / "toy_c_nonstationary_gt.csv", index=False)

    data_d, gt_d = generate_toy_d_mixed_nonlinear()
    data_d.to_csv(OUT_DIR / "toy_d_mixed_nonlinear.csv", index=False)
    gt_d.to_csv(OUT_DIR / "toy_d_mixed_nonlinear_gt.csv", index=False)

    print("OK: toy_c_nonstationary.csv/gt e toy_d_mixed_nonlinear.csv/gt gerados em", OUT_DIR)
