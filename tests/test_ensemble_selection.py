import unittest

import pandas as pd

from causal_discovery.ensemble_selection import (
    compute_method_consistency,
    evaluate_method_combination,
    select_robust_ensemble_combination,
)


def method_a(_data, **_kwargs):
    return pd.DataFrame(
        [{"source": "a", "target": "b", "lag": 1, "score": 1.0, "p_value": 0.01, "method": "A"}]
    )


def method_b_empty(_data, **_kwargs):
    return pd.DataFrame(columns=["source", "target", "lag", "score", "p_value", "method"])


def method_c_duplicate(_data, **_kwargs):
    return pd.DataFrame(
        [
            {"source": "a", "target": "b", "lag": 1, "score": 1.0, "p_value": 0.02, "method": "C"},
            {"source": "a", "target": "b", "lag": 1, "score": 2.0, "p_value": 0.01, "method": "C"},
        ]
    )


class EnsembleSelectionTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [2, 3, 4, 5, 6, 7]})

    def test_evaluate_combination_counts_empty_method_in_support(self):
        result = evaluate_method_combination(
            self.data,
            {"A": method_a, "B": method_b_empty},
            min_votes=1,
            n_bootstrap=2,
            block_size=2,
        )

        summary = result["probabilistic_summary"]

        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary.loc[0, "support_ratio"], 0.5)
        self.assertAlmostEqual(summary.loc[0, "weighted_support_ratio"], 0.5)

    def test_select_robust_combination_returns_ranking(self):
        selection = select_robust_ensemble_combination(
            self.data,
            {"A": method_a, "B": method_b_empty},
            min_methods=1,
            max_methods=2,
            min_votes=1,
            n_bootstrap=2,
            block_size=2,
        )

        self.assertIn("ranking", selection)
        self.assertFalse(selection["ranking"].empty)
        self.assertIn(selection["best_combination"][0], {"A", "B"})

    def test_method_consistency_uses_edge_sets(self):
        consistency = compute_method_consistency(
            {
                "A": method_a(self.data),
                "C": method_c_duplicate(self.data),
                "B": method_b_empty(self.data),
            }
        )

        self.assertAlmostEqual(consistency.loc["A", "C"], 1.0)
        self.assertAlmostEqual(consistency.loc["A", "B"], 0.0)


if __name__ == "__main__":
    unittest.main()
