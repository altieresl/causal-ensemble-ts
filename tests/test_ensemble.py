import math
import unittest

import pandas as pd

from causal_discovery import MethodOutputValidationError, validate_method_output
from causal_discovery.ensemble import (
    run_method_suite,
    summarize_ensemble,
    summarize_probabilistic_ensemble,
)


class EnsembleSummaryTests(unittest.TestCase):
    def test_self_links_are_excluded_from_ensemble_summaries(self):
        result = pd.DataFrame(
            [
                {"source": "x", "target": "x", "lag": 1, "score": 2.0, "p_value": 0.01, "method": "M1"},
                {"source": "x", "target": "y", "lag": 1, "score": 1.0, "p_value": 0.02, "method": "M1"},
            ]
        )

        vote_summary = summarize_ensemble([result], min_votes=1)
        probabilistic_summary = summarize_probabilistic_ensemble(
            [result],
            min_votes=1,
            method_names=["M1"],
        )

        self.assertEqual(list(zip(vote_summary["source"], vote_summary["target"])), [("x", "y")])
        self.assertEqual(
            list(zip(probabilistic_summary["source"], probabilistic_summary["target"])),
            [("x", "y")],
        )

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

    def test_sign_consensus_ignores_structural_presence_scores(self):
        result = pd.DataFrame(
            [
                {"source": "x", "target": "y", "lag": 1, "score": -0.2, "method": "PCMCI"},
                {"source": "x", "target": "y", "lag": 1, "score": -0.5, "method": "VARLiNGAM"},
                {"source": "x", "target": "y", "lag": 1, "score": 1.0, "method": "GES"},
                {"source": "x", "target": "y", "lag": 1, "score": 1.0, "method": "FCI"},
            ]
        )

        summary = summarize_probabilistic_ensemble(
            [result],
            min_votes=1,
            method_names=["PCMCI", "VARLiNGAM", "GES", "FCI"],
        )

        self.assertEqual(summary.loc[0, "positive_votes"], 0)
        self.assertEqual(summary.loc[0, "negative_votes"], 2)
        self.assertEqual(summary.loc[0, "sign_consensus"], "negative")
        self.assertEqual(summary.loc[0, "sign_agreement"], 1.0)
        self.assertEqual(summary.loc[0, "signed_methods"], ["PCMCI", "VARLiNGAM"])

    def test_sign_consensus_reports_mixed_signed_evidence(self):
        result = pd.DataFrame(
            [
                {"source": "x", "target": "y", "lag": 1, "score": -0.2, "method": "PCMCI"},
                {"source": "x", "target": "y", "lag": 1, "score": 0.5, "method": "DYNOTEARS"},
                {"source": "x", "target": "y", "lag": 1, "score": 0.3, "method": "VARLiNGAM"},
            ]
        )

        summary = summarize_ensemble([result], min_votes=1)

        self.assertEqual(summary.loc[0, "positive_votes"], 2)
        self.assertEqual(summary.loc[0, "negative_votes"], 1)
        self.assertEqual(summary.loc[0, "sign_consensus"], "mixed")
        self.assertAlmostEqual(summary.loc[0, "sign_agreement"], 2 / 3)

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
        self.assertIn("sign_consensus", summary.columns)

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

        outputs = run_method_suite(
            pd.DataFrame({"x": [1], "y": [2]}),
            {"Alias": internally_named},
        )
        summary = summarize_probabilistic_ensemble(
            list(outputs.values()),
            min_votes=1,
            method_names=["Alias"],
            method_weights={"Alias": 10.0},
        )

        self.assertEqual(summary.loc[0, "method"], ["Alias"])
        self.assertAlmostEqual(summary.loc[0, "weighted_support_ratio"], 1.0)

    def test_method_suite_rejects_non_dataframe_output(self):
        def invalid_method(_data):
            return []

        with self.assertRaisesRegex(MethodOutputValidationError, "esperado pandas.DataFrame"):
            run_method_suite(pd.DataFrame({"x": [1]}), {"Invalid": invalid_method})

    def test_method_suite_rejects_missing_canonical_columns(self):
        def invalid_method(_data):
            return pd.DataFrame([{"source": "x", "target": "y", "lag": 1}])

        with self.assertRaisesRegex(MethodOutputValidationError, "colunas canonicas ausentes"):
            run_method_suite(
                pd.DataFrame({"x": [1], "y": [2]}),
                {"Invalid": invalid_method},
            )

    def test_method_suite_rejects_invalid_edge_values(self):
        def invalid_method(_data):
            return pd.DataFrame(
                [{
                    "source": "x",
                    "target": "unknown",
                    "lag": 1.5,
                    "score": float("inf"),
                    "p_value": 2.0,
                    "method": "Invalid",
                }]
            )

        with self.assertRaisesRegex(MethodOutputValidationError, "source/target desconhecidos"):
            run_method_suite(pd.DataFrame({"x": [1]}), {"Invalid": invalid_method})

    def test_output_validator_rejects_invalid_lag_score_and_p_value(self):
        valid_row = {
            "source": "x",
            "target": "y",
            "lag": 1,
            "score": 0.5,
            "p_value": 0.05,
            "method": "Test",
        }
        invalid_cases = [
            ({"lag": -1}, "lag"),
            ({"lag": 1.5}, "lag"),
            ({"score": float("nan")}, "score"),
            ({"score": float("inf")}, "score"),
            ({"p_value": -0.1}, "p_value"),
            ({"p_value": 1.1}, "p_value"),
        ]

        for changes, expected_message in invalid_cases:
            with self.subTest(changes=changes):
                row = {**valid_row, **changes}
                with self.assertRaisesRegex(
                    MethodOutputValidationError,
                    expected_message,
                ):
                    validate_method_output(
                        pd.DataFrame([row]),
                        method_name="Test",
                        data_columns=["x", "y"],
                    )

    def test_method_suite_accepts_missing_p_values(self):
        def valid_method(_data):
            return pd.DataFrame(
                [{
                    "source": "x",
                    "target": "y",
                    "lag": 1.0,
                    "score": "0.5",
                    "p_value": None,
                    "method": "Internal",
                }]
            )

        output = run_method_suite(
            pd.DataFrame({"x": [1], "y": [2]}),
            {"Valid": valid_method},
        )["Valid"]

        self.assertEqual(output.loc[0, "lag"], 1)
        self.assertEqual(output.loc[0, "score"], 0.5)
        self.assertTrue(pd.isna(output.loc[0, "p_value"]))
        self.assertEqual(output.loc[0, "method"], "Valid")


if __name__ == "__main__":
    unittest.main()
