import unittest

import numpy as np
import pandas as pd

from causal_discovery.ensemble_selection import (
    add_predictive_validation_score,
    add_ranked_structure_selection,
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
            predictive_validation_weight=0.75,
        )

        self.assertIn("ranking", selection)
        self.assertFalse(selection["ranking"].empty)
        self.assertIn(selection["best_combination"][0], {"A", "B"})
        self.assertIn(
            "pre_validation_ensemble_score",
            selection["best_evaluation"]["probabilistic_summary"],
        )

    def test_select_robust_combination_reuses_precomputed_runs(self):
        output_a = pd.DataFrame(
            [{
                "source": "a", "target": "b", "lag": 1,
                "score": 1.0, "p_value": 0.01, "method": "A",
            }]
        )
        output_b = pd.DataFrame(
            [{
                "source": "a", "target": "b", "lag": 1,
                "score": 0.8, "p_value": 0.02, "method": "B",
            }]
        )

        def must_not_run(data, **kwargs):
            raise AssertionError("o método não deveria ser reexecutado")

        cached_outputs = {"A": output_a, "B": output_b}
        selection = select_robust_ensemble_combination(
            pd.DataFrame({"a": range(20), "b": range(20)}),
            {"A": must_not_run, "B": must_not_run},
            precomputed_outputs=cached_outputs,
            precomputed_bootstrap_outputs=[cached_outputs],
            min_methods=2,
            max_methods=2,
            min_votes=1,
            n_bootstrap=1,
        )

        self.assertIs(selection["precomputed_outputs"]["A"], output_a)
        self.assertEqual(len(selection["precomputed_bootstrap_outputs"]), 1)

    def test_select_robust_combination_extends_bootstrap_cache(self):
        template = pd.DataFrame(
            [{
                "source": "a", "target": "b", "lag": 1,
                "score": 1.0, "p_value": 0.01, "method": "cached",
            }]
        )
        calls = {"count": 0}

        def method(data, **kwargs):
            calls["count"] += 1
            return template.copy()

        cached_outputs = {"A": template.copy(), "B": template.copy()}
        selection = select_robust_ensemble_combination(
            pd.DataFrame({"a": range(20), "b": range(20)}),
            {"A": method, "B": method},
            precomputed_outputs=cached_outputs,
            precomputed_bootstrap_outputs=[cached_outputs],
            min_methods=2,
            max_methods=2,
            min_votes=1,
            n_bootstrap=3,
            random_state=42,
        )

        self.assertEqual(len(selection["precomputed_bootstrap_outputs"]), 3)
        self.assertEqual(calls["count"], 4)

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

    def test_adaptive_weights_discount_redundant_methods(self):
        repeated = pd.DataFrame(
            [{"source": "a", "target": "b", "lag": 1,
              "score": 1.0, "p_value": 0.01, "method": "Repeated"}]
        )
        unique = pd.DataFrame(
            [{"source": "b", "target": "c", "lag": 1,
              "score": 1.0, "p_value": 0.01, "method": "Unique"}]
        )

        diagnostics = estimate_adaptive_method_weights(
            {"RepeatedA": repeated, "RepeatedB": repeated, "Unique": unique},
            diversity_bonus=0.0,
            density_penalty=0.0,
            redundancy_penalty=1.0,
        ).set_index("method")

        self.assertGreater(
            diagnostics.loc["RepeatedA", "redundancy"],
            diagnostics.loc["Unique", "redundancy"],
        )
        self.assertGreater(
            diagnostics.loc["Unique", "adaptive_weight"],
            diagnostics.loc["RepeatedA", "adaptive_weight"],
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

    def test_predictive_validation_penalizes_unsupported_edge_and_preserves_probability(self):
        rng = np.random.default_rng(42)
        sample_count = 120
        source = rng.normal(size=sample_count)
        unrelated = rng.normal(size=sample_count)
        target = np.zeros(sample_count)
        target[1:] = 2.0 * source[:-1] + rng.normal(scale=0.05, size=sample_count - 1)
        data = pd.DataFrame({"source": source, "unrelated": unrelated, "target": target})
        summary = pd.DataFrame(
            [
                {
                    "source": "source", "target": "target", "lag": 1,
                    "ensemble_score": 0.8, "edge_probability": 0.7,
                },
                {
                    "source": "unrelated", "target": "target", "lag": 1,
                    "ensemble_score": 0.8, "edge_probability": 0.7,
                },
            ]
        )

        validated = add_predictive_validation_score(
            summary,
            data,
            validation_weight=0.75,
            max_lag=2,
        ).set_index("source")

        self.assertGreater(
            validated.loc["source", "predictive_gain"],
            validated.loc["unrelated", "predictive_gain"],
        )
        self.assertGreater(
            validated.loc["source", "ensemble_score"],
            validated.loc["unrelated", "ensemble_score"],
        )
        self.assertAlmostEqual(validated.loc["source", "edge_probability"], 0.7)
        self.assertAlmostEqual(validated.loc["unrelated", "edge_probability"], 0.7)

    def test_predictive_validation_can_condition_on_other_candidate_parents(self):
        rng = np.random.default_rng(7)
        sample_count = 240
        driver = rng.normal(size=sample_count)
        proxy = driver + rng.normal(scale=0.5, size=sample_count)
        target = np.zeros(sample_count)
        target[1:] = 2.0 * driver[:-1] + rng.normal(
            scale=0.2, size=sample_count - 1
        )
        data = pd.DataFrame({"driver": driver, "proxy": proxy, "target": target})
        summary = pd.DataFrame(
            [
                {
                    "source": "driver", "target": "target", "lag": 1,
                    "ensemble_score": 0.9, "edge_probability": 0.7,
                },
                {
                    "source": "proxy", "target": "target", "lag": 1,
                    "ensemble_score": 0.7, "edge_probability": 0.7,
                },
            ]
        )

        validated = add_predictive_validation_score(
            summary,
            data,
            validation_weight=0.75,
            max_lag=2,
            n_splits=5,
            conditional_parents=1,
        ).set_index("source")

        self.assertGreater(
            validated.loc["driver", "predictive_gain"],
            validated.loc["proxy", "predictive_gain"],
        )

    def test_predictive_validation_uncertainty_penalty_is_conservative(self):
        rng = np.random.default_rng(21)
        sample_count = 160
        source = rng.normal(size=sample_count)
        target = np.zeros(sample_count)
        target[1:] = source[:-1] + rng.normal(scale=0.8, size=sample_count - 1)
        data = pd.DataFrame({"source": source, "target": target})
        summary = pd.DataFrame(
            [{
                "source": "source", "target": "target", "lag": 1,
                "ensemble_score": 0.8, "edge_probability": 0.7,
            }]
        )

        unpenalized = add_predictive_validation_score(
            summary, data, max_lag=2, n_splits=5, uncertainty_penalty=0.0
        )
        conservative = add_predictive_validation_score(
            summary, data, max_lag=2, n_splits=5, uncertainty_penalty=1.0
        )

        self.assertLessEqual(
            conservative.loc[0, "predictive_gain"],
            unpenalized.loc[0, "predictive_gain"],
        )
        self.assertIn("predictive_gain_standard_error", conservative)
        self.assertGreaterEqual(
            conservative.loc[0, "predictive_positive_split_ratio"], 0.0
        )

    def test_ranked_structure_selection_caps_pairs_and_preserves_probability(self):
        summary = pd.DataFrame(
            [
                {"source": "a", "target": "b", "lag": 1,
                 "ensemble_score": 0.9, "edge_probability": 0.3},
                {"source": "b", "target": "a", "lag": 2,
                 "ensemble_score": 0.8, "edge_probability": 0.7},
                {"source": "a", "target": "c", "lag": 1,
                 "ensemble_score": 0.7, "edge_probability": 0.6},
                {"source": "a", "target": "d", "lag": 1,
                 "ensemble_score": 0.6, "edge_probability": 0.6},
                {"source": "b", "target": "c", "lag": 1,
                 "ensemble_score": 0.5, "edge_probability": 0.9},
            ]
        )

        selected = add_ranked_structure_selection(
            summary, nodes=["a", "b", "c", "d"], max_pair_density=0.5
        )

        selected_pairs = {
            tuple(sorted((row.source, row.target)))
            for row in selected.loc[selected["ensemble_selected"]].itertuples()
        }
        self.assertEqual(selected_pairs, {("a", "b"), ("a", "c"), ("a", "d")})
        self.assertTrue(selected.loc[0, "ensemble_selected"])
        self.assertTrue(selected.loc[1, "ensemble_selected"])
        pd.testing.assert_series_equal(
            selected["edge_probability"], summary["edge_probability"]
        )

    def test_ranked_structure_selection_validates_density(self):
        with self.assertRaises(ValueError):
            add_ranked_structure_selection(
                pd.DataFrame([{
                    "source": "a", "target": "b", "ensemble_score": 0.5,
                }]),
                max_pair_density=1.1,
            )

    def test_ranked_structure_selection_rescues_only_jointly_supported_pair(self):
        summary = pd.DataFrame(
            [
                {"source": "a", "target": "b", "ensemble_score": 0.9,
                 "bootstrap_probability": 0.2, "predictive_rank": 0.2,
                 "support_ratio": 0.5},
                {"source": "a", "target": "c", "ensemble_score": 0.8,
                 "bootstrap_probability": 0.2, "predictive_rank": 0.2,
                 "support_ratio": 0.5},
                {"source": "a", "target": "d", "ensemble_score": 0.4,
                 "bootstrap_probability": 0.8, "predictive_rank": 0.7,
                 "support_ratio": 0.75},
                {"source": "b", "target": "c", "ensemble_score": 0.3,
                 "bootstrap_probability": 0.8, "predictive_rank": 0.4,
                 "support_ratio": 0.75},
            ]
        )

        selected = add_ranked_structure_selection(
            summary,
            nodes=["a", "b", "c", "d"],
            max_pair_density=0.25,
            rescue_bootstrap_min=0.6,
            rescue_predictive_rank_min=0.5,
            rescue_support_min=0.75,
        )
        selected_pairs = {
            tuple(sorted((row.source, row.target)))
            for row in selected.loc[selected["ensemble_selected"]].itertuples()
        }

        self.assertEqual(selected_pairs, {("a", "b"), ("a", "c"), ("a", "d")})


if __name__ == "__main__":
    unittest.main()
