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
- toy_e_boundary_mixed: pressiona os dois eixos ao mesmo tempo, e a
  estacionariedade fica perto da fronteira de 50% (do lado "nao majoritario",
  nao exatamente em cima) em vez de claramente de um lado como no toy_c (33%).
  A nao linearidade tambem aparece em variaveis diferentes das que carregam a
  nao estacionariedade -- os dois problemas nao andam juntos nas mesmas series.
  Testa se o chat aplica a comparacao ">=50%" corretamente quando a maioria
  nao e obvia, em vez de reagir so a "tem serie nao estacionaria/nao linear
  mencionada no perfil".

Mesmo formato de saida de synthetic_causal_datasets.ipynb: CSV de dados e CSV de
ground truth com colunas Edge/Direct/Coefficient/Lag/Type.

- toy_f_non_gaussian: dataset de CALIBRACAO (nao usado nos testes de selecao de
  algoritmo dos outros 3) para o eixo non_gaussian_errors de
  causal_algorithms_atlas/dataset_profile.py. X0/X2 recebem inovacoes uniformes
  (nao gaussiana, simetrica), X3 recebe inovacoes exponenciais centralizadas
  (nao gaussiana, assimetrica), X1/Y recebem ruido gaussiano normal (controle).
  Usado para calibrar os limiares de skewness/curtose em excesso do residuo de
  um VAR(1) que substituiram o teste de significancia (Shapiro-Wilk) original,
  descartado por rejeitar normalidade em amostras grandes mesmo com residuos
  gaussianos de verdade (ver dataset_profile._NON_GAUSSIAN_SKEW_THRESHOLD).
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


def generate_toy_e_boundary_mixed(T: int = 1300, seed: int = 99) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    x0 = np.zeros(T)  # random walk (raiz unitaria) -- nao estacionaria, raiz
    x1 = np.zeros(T)  # AR(1) estavel -- estacionaria, raiz
    x2 = np.zeros(T)  # linear de X0 -- nao estacionaria (herda)
    x3 = np.zeros(T)  # linear de X0 -- nao estacionaria (herda, 2o filho de X0)
    x4 = np.zeros(T)  # linear de X1 -- estacionaria
    x5 = np.zeros(T)  # tanh de X1 -- estacionaria, mas nao linear
    y = np.zeros(T)  # X2 (linear) + tanh(X4) (nao linear) -- nao estacionaria (herda de X2)

    for t in range(1, T):
        x0[t] = x0[t - 1] + rng.normal(0, 0.3)
        x1[t] = 0.4 * x1[t - 1] + rng.normal(0, 0.5)
        x2[t] = 0.6 * x0[t - 1] + rng.normal(0, 0.4)
        x3[t] = 0.5 * x0[t - 1] + rng.normal(0, 0.4)
        x4[t] = 0.5 * x1[t - 1] + rng.normal(0, 0.4)
        x5[t] = 1.1 * np.tanh(2.2 * x1[t - 1]) + rng.normal(0, 0.35)
        y[t] = 0.5 * x2[t - 1] + 0.9 * np.tanh(2.0 * x4[t - 1]) + rng.normal(0, 0.45)

    data = pd.DataFrame({"X0": x0, "X1": x1, "X2": x2, "X3": x3, "X4": x4, "X5": x5, "Y": y})

    ground_truth = pd.DataFrame(
        [
            {"Edge": "X0 → X2", "Direct": True, "Coefficient": 0.6, "Lag": 1.0, "Type": "linear"},
            {"Edge": "X0 → X3", "Direct": True, "Coefficient": 0.5, "Lag": 1.0, "Type": "linear"},
            {"Edge": "X1 → X4", "Direct": True, "Coefficient": 0.5, "Lag": 1.0, "Type": "linear"},
            {"Edge": "X1 → X5", "Direct": True, "Coefficient": 1.1, "Lag": 1.0, "Type": "nonlinear (tanh)"},
            {"Edge": "X2 → Y", "Direct": True, "Coefficient": 0.5, "Lag": 1.0, "Type": "linear"},
            {"Edge": "X4 → Y", "Direct": True, "Coefficient": 0.9, "Lag": 1.0, "Type": "nonlinear (tanh)"},
            {"Edge": "X0 → Y", "Direct": False, "Coefficient": None, "Lag": None, "Type": "indirect via X2"},
            {"Edge": "X1 → Y", "Direct": False, "Coefficient": None, "Lag": None, "Type": "indirect via X4"},
        ]
    )
    return data, ground_truth


def generate_toy_f_non_gaussian(T: int = 1200, seed: int = 123) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    x0 = np.zeros(T)  # AR(1), inovacoes uniformes -- nao gaussiana, simetrica
    x1 = np.zeros(T)  # AR(1), inovacoes gaussianas -- controle
    x2 = np.zeros(T)  # depende de X0, inovacoes uniformes -- nao gaussiana
    x3 = np.zeros(T)  # AR(1), inovacoes exponenciais centralizadas -- nao gaussiana, assimetrica
    y = np.zeros(T)  # depende de X0 (nao gaussiano) e X1 (gaussiano), inovacao propria gaussiana

    for t in range(1, T):
        x0[t] = 0.5 * x0[t - 1] + rng.uniform(-0.6, 0.6)
        x1[t] = 0.4 * x1[t - 1] + rng.normal(0, 0.5)
        x2[t] = 0.5 * x0[t - 1] + 0.3 * x2[t - 1] + rng.uniform(-0.5, 0.5)
        x3[t] = 0.4 * x3[t - 1] + (rng.exponential(0.4) - 0.4)
        y[t] = 0.6 * x0[t - 1] + 0.4 * x1[t - 1] + rng.normal(0, 0.5)

    data = pd.DataFrame({"X0": x0, "X1": x1, "X2": x2, "X3": x3, "Y": y})

    ground_truth = pd.DataFrame(
        [
            {"Edge": "X0 → X2", "Direct": True, "Coefficient": 0.5, "Lag": 1.0, "Type": "linear"},
            {"Edge": "X0 → Y", "Direct": True, "Coefficient": 0.6, "Lag": 1.0, "Type": "linear"},
            {"Edge": "X1 → Y", "Direct": True, "Coefficient": 0.4, "Lag": 1.0, "Type": "linear"},
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

    data_e, gt_e = generate_toy_e_boundary_mixed()
    data_e.to_csv(OUT_DIR / "toy_e_boundary_mixed.csv", index=False)
    gt_e.to_csv(OUT_DIR / "toy_e_boundary_mixed_gt.csv", index=False)

    data_f, gt_f = generate_toy_f_non_gaussian()
    data_f.to_csv(OUT_DIR / "toy_f_non_gaussian.csv", index=False)
    gt_f.to_csv(OUT_DIR / "toy_f_non_gaussian_gt.csv", index=False)

    print(
        "OK: toy_c_nonstationary, toy_d_mixed_nonlinear, toy_e_boundary_mixed e "
        f"toy_f_non_gaussian (csv/gt) gerados em {OUT_DIR}"
    )
