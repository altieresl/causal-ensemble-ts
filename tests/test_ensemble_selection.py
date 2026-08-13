import unittest

import pandas as pd

from causal_discovery.ensemble_selection import (
    compute_method_consistency,
    estimate_adaptive_method_weights,
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
        self.assertGreater(summary.loc[0, "weighted_support_ratio"], 0.5)
        self.assertGreater(
            result["effective_method_weights"]["A"],
            result["effective_method_weights"]["B"],
        )

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

    def test_adaptive_weights_penalize_unstable_dense_method(self):
        stable = pd.DataFrame(
            [{"source": "a", "target": "b", "lag": 1, "score": 1.0, "p_value": 0.01, "method": "Stable"}]
        )
        dense = pd.DataFrame(
            [
                {"source": source, "target": target, "lag": 1, "score": 1.0, "p_value": 0.01, "method": "Dense"}
                for source, target in [("a", "b"), ("a", "c"), ("b", "c"), ("c", "a")]
            ]
        )
        bootstrap_outputs = [
            {
                "Stable": stable,
                "Dense": dense.iloc[[index]].reset_index(drop=True),
            }
            for index in range(len(dense))
        ]

        diagnostics = estimate_adaptive_method_weights(
            {"Stable": stable, "Dense": dense},
            bootstrap_outputs,
        ).set_index("method")

        self.assertGreater(
            diagnostics.loc["Stable", "bootstrap_stability"],
            diagnostics.loc["Dense", "bootstrap_stability"],
        )
        self.assertGreater(
            diagnostics.loc["Stable", "adaptive_weight"],
            diagnostics.loc["Dense", "adaptive_weight"],
        )

    def test_evaluation_blends_base_evidence_with_bootstrap_consensus(self):
        common = pd.DataFrame(
            [{"source": "a", "target": "b", "lag": 1, "score": 1.0, "p_value": 0.01, "method": "A"}]
        )
        extra = pd.DataFrame(
            [
                {"source": "a", "target": "b", "lag": 1, "score": 1.0, "p_value": 0.01, "method": "B"},
                {"source": "a", "target": "c", "lag": 1, "score": 1.0, "p_value": 0.01, "method": "B"},
            ]
        )
        empty = pd.DataFrame(columns=["source", "target", "lag", "score", "p_value", "method"])
        bootstrap_outputs = [
            {"A": common, "B": extra.iloc[[0]].reset_index(drop=True)},
            {"A": common, "B": extra.iloc[[0]].reset_index(drop=True)},
            {"A": common, "B": empty},
        ]

        result = evaluate_method_combination(
            self.data.assign(c=[3, 4, 5, 6, 7, 8]),
            {"A": method_a, "B": method_b_empty},
            precomputed_outputs={"A": common, "B": extra},
            precomputed_bootstrap_outputs=bootstrap_outputs,
            min_votes=1,
            n_bootstrap=3,
            stability_weight=0.8,
        )
        summary = result["probabilistic_summary"].set_index(["source", "target", "lag"])

        self.assertIn("bootstrap_probability", summary.columns)
        self.assertIn("base_edge_probability", summary.columns)
        self.assertGreater(
            summary.loc[("a", "b", 1), "edge_probability"],
            summary.loc[("a", "c", 1), "edge_probability"],
        )
        self.assertIn("method_weight_diagnostics", result)

    def test_local_expert_preserves_strong_unique_edge_without_changing_probability(self):
        stable_a = pd.DataFrame(
            [{"source": "a", "target": "b", "lag": 1, "score": 2.0, "p_value": 0.01, "method": "A"}]
        )
        intermittent_b = pd.DataFrame(
            [{"source": "a", "target": "c", "lag": 1, "score": 2.0, "p_value": 0.01, "method": "B"}]
        )
        empty = pd.DataFrame(columns=["source", "target", "lag", "score", "p_value", "method"])
        bootstrap_outputs = [
            {"A": stable_a, "B": intermittent_b},
            {"A": stable_a, "B": empty},
            {"A": stable_a, "B": empty},
        ]
        common_kwargs = {
            "precomputed_outputs": {"A": stable_a, "B": intermittent_b},
            "precomputed_bootstrap_outputs": bootstrap_outputs,
            "min_votes": 1,
            "n_bootstrap": 3,
            "adaptive_method_weights": False,
        }

        consensus_only = evaluate_method_combination(
            self.data.assign(c=[3, 4, 5, 6, 7, 8]),
            {"A": method_a, "B": method_b_empty},
            local_expert_weight=0.0,
            **common_kwargs,
        )["probabilistic_summary"].set_index(["source", "target", "lag"])
        expert_only = evaluate_method_combination(
            self.data.assign(c=[3, 4, 5, 6, 7, 8]),
            {"A": method_a, "B": method_b_empty},
            local_expert_weight=1.0,
            **common_kwargs,
        )["probabilistic_summary"].set_index(["source", "target", "lag"])

        stable_key = ("a", "b", 1)
        self.assertEqual(expert_only.loc[stable_key, "dominant_method"], "A")
        self.assertEqual(expert_only.loc[stable_key, "dominant_edge_stability"], 1.0)
        self.assertGreater(
            expert_only.loc[stable_key, "ensemble_score"],
            consensus_only.loc[stable_key, "ensemble_score"],
        )
        self.assertAlmostEqual(
            expert_only.loc[stable_key, "edge_probability"],
            consensus_only.loc[stable_key, "edge_probability"],
        )


if __name__ == "__main__":
    unittest.main()
