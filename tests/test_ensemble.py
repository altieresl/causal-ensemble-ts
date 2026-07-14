import math
import unittest

import pandas as pd

from causal_discovery.ensemble import (
    run_method_suite,
    summarize_ensemble,
    summarize_probabilistic_ensemble,
)


class EnsembleSummaryTests(unittest.TestCase):
    def test_duplicate_edges_from_same_method_count_as_one_vote(self):
        result = pd.DataFrame(
            [
                {"source": "x", "target": "y", "lag": 1, "score": 1.0, "p_value": 0.02, "method": "M1"},
                {"source": "x", "target": "y", "lag": 1, "score": 3.0, "p_value": 0.01, "method": "M1"},
            ]
        )

        summary = summarize_ensemble([result], min_votes=1)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.loc[0, "votes"], 1)
        self.assertEqual(summary.loc[0, "method"], ["M1"])
        self.assertAlmostEqual(summary.loc[0, "mean_score"], 2.0)

    def test_empty_methods_are_counted_in_support_denominator(self):
        result = pd.DataFrame(
            [{"source": "x", "target": "y", "lag": 1, "score": 1.0, "p_value": 0.01, "method": "M1"}]
        )
        empty = pd.DataFrame(columns=["source", "target", "lag", "score", "p_value", "method"])

        summary = summarize_probabilistic_ensemble(
            [result, empty],
            min_votes=1,
            method_names=["M1", "M2"],
        )

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.loc[0, "votes"], 1)
        self.assertAlmostEqual(summary.loc[0, "support_ratio"], 0.5)
        self.assertAlmostEqual(summary.loc[0, "weighted_support_ratio"], 0.5)

    def test_missing_p_value_falls_back_to_score_probability(self):
        result = pd.DataFrame(
            [{"source": "x", "target": "y", "lag": 1, "score": 2.0, "method": "M1"}]
        )

        summary = summarize_probabilistic_ensemble(
            [result],
            min_votes=1,
            method_names=["M1"],
        )

        self.assertEqual(len(summary), 1)
        self.assertTrue(math.isnan(summary.loc[0, "combined_p_value"]))
        self.assertGreater(summary.loc[0, "edge_probability"], 0.0)
        self.assertLessEqual(summary.loc[0, "edge_probability"], 1.0)

    def test_min_votes_returns_empty_frame_with_expected_columns(self):
        result = pd.DataFrame(
            [{"source": "x", "target": "y", "lag": 1, "score": 1.0, "p_value": 0.01, "method": "M1"}]
        )

        summary = summarize_probabilistic_ensemble(
            [result],
            min_votes=2,
            method_names=["M1", "M2"],
        )

        self.assertTrue(summary.empty)
        self.assertIn("edge_probability", summary.columns)
        self.assertIn("support_ratio", summary.columns)

    def test_score_fallback_does_not_cancel_opposite_effect_signs(self):
        negative = pd.DataFrame(
            [{"source": "x", "target": "y", "lag": 1, "score": -1.0, "method": "M1"}]
        )
        positive = pd.DataFrame(
            [{"source": "x", "target": "y", "lag": 1, "score": 1.0, "method": "M2"}]
        )

        summary = summarize_probabilistic_ensemble(
            [negative, positive],
            min_votes=2,
            posterior_weight=1.0,
            method_names=["M1", "M2"],
        )

        self.assertGreater(summary.loc[0, "posterior_probability"], 0.7)

    def test_method_registry_name_controls_output_label_and_weight(self):
        def internally_named(_data):
            return pd.DataFrame(
                [
                    {
                        "source": "x",
                        "target": "y",
                        "lag": 1,
                        "score": 1.0,
                        "p_value": None,
                        "method": "InternalName",
                    }
                ]
            )

        outputs = run_method_suite(pd.DataFrame({"x": [1]}), {"Alias": internally_named})
        summary = summarize_probabilistic_ensemble(
            list(outputs.values()),
            min_votes=1,
            method_names=["Alias"],
            method_weights={"Alias": 10.0},
        )

        self.assertEqual(summary.loc[0, "method"], ["Alias"])
        self.assertAlmostEqual(summary.loc[0, "weighted_support_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
