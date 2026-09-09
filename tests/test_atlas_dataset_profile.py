from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from causal_algorithms_atlas.dataset_profile import profile_dataset


def _linear_stationary_data(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.normal(0, 1.0)
        y[t] = 0.4 * y[t - 1] + 0.3 * x[t - 1] + rng.normal(0, 1.0)
    return pd.DataFrame({"x": x, "y": y})


def _nonlinear_stationary_data(n: int = 400, seed: int = 2) -> pd.DataFrame:
    """z e um AR(1) linear estavel; x e y sao funcoes tanh(.) do lag de z.

    Duas series tanh AUTORREFERENCIADAS (x[t] ~ tanh(x[t-1])) com coeficiente/escala
    fortes tem derivada > 1 no ponto fixo em zero -- o mapa fica instavel/quase
    caotico, e nenhum modelo suave (linear ou polinomial) preve bem fora da amostra
    nesse regime, mascarando a comparacao. Usar uma variavel dirigida por um AR(1)
    estavel evita essa armadilha e mede o que interessa: o teste detecta a curvatura
    tanh na relacao causal, nao a previsibilidade de um sistema caotico.
    """
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        z[t] = 0.5 * z[t - 1] + rng.normal(0, 1.0)
        x[t] = 1.2 * np.tanh(z[t - 1]) + rng.normal(0, 0.15)
        y[t] = 1.0 * np.tanh(z[t - 1] * 0.8) + rng.normal(0, 0.15)
    return pd.DataFrame({"z": z, "x": x, "y": y})


def _random_walk_data(n: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1.0, size=n)
    walk = np.cumsum(steps)
    return pd.DataFrame({"w": walk})


def _cross_variable_nonlinear_data(n: int = 400, seed: int = 5) -> pd.DataFrame:
    """x tem autodinamica linear; y depende nao linearmente do LAG de x, nao de si mesma.

    Um teste de linearidade que so olha a autorregressao de cada variavel isolada
    (y[t] ~ y[t-1]) erra este caso: a nao linearidade esta na relacao x -> y, nao na
    dinamica propria de y.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.normal(0, 0.3)
        y[t] = 0.3 * y[t - 1] + 1.5 * np.tanh(x[t - 1] * 3) + rng.normal(0, 0.15)
    return pd.DataFrame({"x": x, "y": y})


def _mildly_nonlinear_data(n: int = 3000, seed: int = 9) -> pd.DataFrame:
    """Termo quadratico pequeno demais para importar na pratica, mas grande sample.

    Calibrado para que um teste de significancia estatistica pura (RESET p-valor)
    rejeite linearidade (p=0.017 neste fixture) mesmo o termo quadratico sendo
    pequeno demais para mudar a escolha de algoritmo na pratica -- e exatamente o
    falso positivo que o criterio de tamanho de efeito (ganho preditivo fora da
    amostra) deve evitar.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.normal(0, 1.0)
        y[t] = 0.5 * y[t - 1] + 0.5 * x[t - 1] + 0.05 * x[t - 1] ** 2 + rng.normal(0, 1.0)
    return pd.DataFrame({"x": x, "y": y})


class ProfileDatasetTests(unittest.TestCase):
    def test_reports_shape(self):
        data = _linear_stationary_data(n=200)
        profile = profile_dataset(data)
        self.assertEqual(profile.n_variables, 2)
        self.assertEqual(profile.n_timepoints, 200)
        self.assertEqual(len(profile.variables), 2)

    def test_flags_stationary_ar_process_as_stationary(self):
        data = _linear_stationary_data(n=300)
        profile = profile_dataset(data)
        self.assertTrue(profile.mostly_stationary)

    def test_flags_random_walk_as_non_stationary(self):
        data = _random_walk_data(n=300)
        profile = profile_dataset(data)
        self.assertFalse(profile.mostly_stationary)

    def test_flags_linear_ar_process_as_linear(self):
        data = _linear_stationary_data(n=300)
        profile = profile_dataset(data)
        self.assertTrue(profile.mostly_linear)

    def test_flags_strongly_nonlinear_process_as_non_linear(self):
        data = _nonlinear_stationary_data(n=300)
        profile = profile_dataset(data)
        self.assertFalse(profile.mostly_linear)

    def test_detects_nonlinearity_that_only_shows_in_cross_variable_relation(self):
        data = _cross_variable_nonlinear_data()
        profile = profile_dataset(data)
        by_name = {v.name: v for v in profile.variables}
        self.assertFalse(by_name["y"].linear)

    def test_statistically_detectable_but_practically_negligible_nonlinearity_is_linear(self):
        # Com n=3000 um teste de p-valor puro rejeitaria linearidade aqui; o criterio
        # de ganho preditivo fora da amostra nao deve, porque o termo quadratico
        # praticamente nao reduz o erro de previsao.
        data = _mildly_nonlinear_data()
        profile = profile_dataset(data)
        by_name = {v.name: v for v in profile.variables}
        self.assertTrue(by_name["y"].linear)

    def test_too_short_series_is_marked_untestable_not_guessed(self):
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        profile = profile_dataset(data)
        variable = profile.variables[0]
        self.assertIsNone(variable.stationary)
        self.assertIsNone(variable.linear)

    def test_query_text_mentions_latent_confounder_limitation(self):
        data = _linear_stationary_data(n=200)
        profile = profile_dataset(data)
        text = profile.to_query_text()
        self.assertIn("confundidor", text.lower())
        self.assertIn(str(profile.n_variables), text)


if __name__ == "__main__":
    unittest.main()
