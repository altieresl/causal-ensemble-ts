import unittest
from unittest.mock import patch

import numpy as np

from causal_discovery.panel import (
    run_pcmci_multiple_trajectories,
    standardize_trajectories,
)


class PanelTests(unittest.TestCase):
    def test_standardization_is_independent_per_trajectory_and_variable(self):
        values = np.arange(48, dtype=float).reshape(2, 8, 3)

        standardized = standardize_trajectories(values)

        np.testing.assert_allclose(standardized.mean(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(standardized.std(axis=1), 1.0, atol=1e-12)

    def test_pcmci_multiple_returns_one_score_for_every_pair(self):
        values = np.random.default_rng(4).normal(size=(3, 20, 3))
        fake_results = {
            "val_matrix": np.arange(27, dtype=float).reshape(3, 3, 3),
            "p_matrix": np.full((3, 3, 3), 0.2),
        }

        with patch("causal_discovery.panel.PCMCI") as pcmci_class:
            pcmci_class.return_value.run_pcmci.return_value = fake_results
            result = run_pcmci_multiple_trajectories(
                values,
                ["a", "b", "c"],
                max_lag=2,
            )

        self.assertEqual(len(result), 3)
        self.assertEqual(
            set(map(tuple, result[["source", "target"]].to_numpy())),
            {("a", "b"), ("a", "c"), ("b", "c")},
        )
        call_dataframe = pcmci_class.call_args.kwargs["dataframe"]
        self.assertEqual(call_dataframe.analysis_mode, "multiple")


if __name__ == "__main__":
    unittest.main()
