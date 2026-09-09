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


def _nonlinear_stationary_data(n: int = 300, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * np.tanh(x[t - 1] * 2.5) + rng.normal(0, 0.2)
        y[t] = 0.5 * np.tanh(y[t - 1] * 3) + rng.normal(0, 0.3)
    return pd.DataFrame({"x": x, "y": y})


def _random_walk_data(n: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1.0, size=n)
    walk = np.cumsum(steps)
    return pd.DataFrame({"w": walk})


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
