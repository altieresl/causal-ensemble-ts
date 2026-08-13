import unittest

import pandas as pd

from causal_discovery.benchmark import (
    build_complete_undirected_pair_scores,
    compute_paired_superiority_statistics,
    compute_ranked_undirected_skeleton_metrics,
    compute_structural_metrics,
    compute_undirected_skeleton_metrics,
    generate_synthetic_timeseries,
    inject_noise_regime_change,
)


class BenchmarkTests(unittest.TestCase):
    def test_synthetic_generator_is_reproducible_without_global_rng(self):
        first, first_truth = generate_synthetic_timeseries(random_state=12)
        second, second_truth = generate_synthetic_timeseries(random_state=12)

        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(first_truth, second_truth)

    def test_reversed_edge_counts_as_one_shd_operation(self):
        truth = pd.DataFrame([{"source": "x", "target": "y", "lag": 1}])
        prediction = pd.DataFrame(
            [{"source": "y", "target": "x", "lag": 1, "edge_probability": 0.9}]
        )

        metrics = compute_structural_metrics(prediction, truth)

        self.assertEqual(metrics["reversed_edges"], 1)
        self.assertEqual(metrics["structural_hamming_distance"], 1)

    def test_undirected_skeleton_ignores_direction_lag_and_duplicates(self):
        truth = pd.DataFrame(
            [
                {"source": "x", "target": "y", "lag": pd.NA},
                {"source": "y", "target": "x", "lag": pd.NA},
                {"source": "y", "target": "z", "lag": pd.NA},
            ]
        )
        prediction = pd.DataFrame(
            [
                {"source": "y", "target": "x", "lag": 1, "edge_probability": 0.8},
                {"source": "x", "target": "y", "lag": 2, "edge_probability": 0.7},
                {"source": "x", "target": "z", "lag": 1, "edge_probability": 0.6},
            ]
        )

        metrics = compute_undirected_skeleton_metrics(
            prediction,
            truth,
            prob_threshold=0.5,
            nodes=["x", "y", "z"],
        )

        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["candidate_pairs"], 3)
        self.assertAlmostEqual(metrics["ground_truth_prevalence"], 2 / 3)
        self.assertEqual(metrics["structural_hamming_distance"], 2)

    def test_undirected_skeleton_applies_probability_threshold(self):
        truth = pd.DataFrame([{"source": "x", "target": "y", "lag": pd.NA}])
        prediction = pd.DataFrame(
            [
                {"source": "x", "target": "y", "lag": 1, "edge_probability": 0.49},
                {"source": "y", "target": "x", "lag": 2, "edge_probability": 0.51},
            ]
        )

        metrics = compute_undirected_skeleton_metrics(prediction, truth)

        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 0)

    def test_undirected_skeleton_respects_evaluated_relations(self):
        truth = pd.DataFrame(
            [
                {"source": "x", "target": "y", "lag": pd.NA},
                {"source": "y", "target": "z", "lag": pd.NA},
            ]
        )
        prediction = pd.DataFrame(
            [{"source": "y", "target": "x", "lag": 2, "edge_probability": 0.9}]
        )

        metrics = compute_undirected_skeleton_metrics(
            prediction,
            truth,
            evaluated_relations=[("x", "y"), ("y", "x")],
        )

        self.assertEqual(metrics["candidate_pairs"], 1)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 0)

    def test_ranked_skeleton_metrics_uses_all_positive_and_negative_pairs(self):
        truth = pd.DataFrame(
            [{"source": "x", "target": "y", "lag": pd.NA}]
        )
        scores = pd.DataFrame(
            [
                {"source": "x", "target": "y", "score": 0.9},
                {"source": "x", "target": "z", "score": 0.2},
                {"source": "y", "target": "z", "score": 0.1},
            ]
        )

        metrics = compute_ranked_undirected_skeleton_metrics(scores, truth)

        self.assertEqual(metrics["candidate_pairs"], 3)
        self.assertEqual(metrics["positive_pairs"], 1)
        self.assertEqual(metrics["negative_pairs"], 2)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["average_precision"], 1.0)
        self.assertAlmostEqual(metrics["random_average_precision"], 1 / 3)

    def test_ranked_skeleton_metrics_rejects_duplicate_pairs(self):
        truth = pd.DataFrame(
            [{"source": "x", "target": "y", "lag": pd.NA}]
        )
        scores = pd.DataFrame(
            [
                {"source": "x", "target": "y", "score": 0.9},
                {"source": "y", "target": "x", "score": 0.8},
                {"source": "x", "target": "z", "score": 0.1},
            ]
        )

        with self.assertRaises(ValueError):
            compute_ranked_undirected_skeleton_metrics(scores, truth)

    def test_complete_pair_scores_aggregates_and_fills_missing_pairs(self):
        predictions = pd.DataFrame(
            [
                {"source": "x", "target": "y", "p_value": 0.20},
                {"source": "y", "target": "x", "p_value": 0.05},
            ]
        )

        scores = build_complete_undirected_pair_scores(
            predictions,
            ["x", "y", "z"],
            evidence="one_minus_p_value",
        )

        self.assertEqual(len(scores), 3)
        lookup = {
            (row.source, row.target): row.score
            for row in scores.itertuples(index=False)
        }
        self.assertAlmostEqual(lookup[("x", "y")], 0.95)
        self.assertEqual(lookup[("x", "z")], 0.0)
        self.assertEqual(lookup[("y", "z")], 0.0)

    def test_complete_pair_scores_accepts_ensemble_ranking_score(self):
        predictions = pd.DataFrame(
            [
                {"source": "x", "target": "y", "ensemble_score": 0.85},
                {"source": "x", "target": "z", "ensemble_score": 0.30},
            ]
        )

        scores = build_complete_undirected_pair_scores(
            predictions,
            ["x", "y", "z"],
            evidence="ensemble_score",
        )
        lookup = {
            (row.source, row.target): row.score
            for row in scores.itertuples(index=False)
        }

        self.assertAlmostEqual(lookup[("x", "y")], 0.85)
        self.assertAlmostEqual(lookup[("x", "z")], 0.30)
        self.assertEqual(lookup[("y", "z")], 0.0)

    def test_paired_superiority_orients_lower_metric_as_improvement(self):
        results = pd.DataFrame(
            [
                {"trajectory_index": index, "strategy": "ENSEMBLE", "shd": value}
                for index, value in enumerate([1.0, 2.0, 1.0, 2.0])
            ]
            + [
                {"trajectory_index": index, "strategy": "PCMCI", "shd": value}
                for index, value in enumerate([3.0, 3.0, 2.0, 4.0])
            ]
        )

        summary = compute_paired_superiority_statistics(
            results,
            candidate="ENSEMBLE",
            baseline="PCMCI",
            metric="shd",
            higher_is_better=False,
            n_bootstrap=500,
            random_state=7,
        )

        self.assertEqual(summary["paired_trajectories"], 4)
        self.assertGreater(summary["mean_improvement"], 0.0)
        self.assertGreater(summary["confidence_interval_low"], 0.0)
        self.assertEqual(summary["win_rate"], 1.0)

    def test_noise_injection_is_reproducible_and_validates_index(self):
        data, _ = generate_synthetic_timeseries(n_samples=30, random_state=5)
        first = inject_noise_regime_change(data, index_change=10, random_state=8)
        second = inject_noise_regime_change(data, index_change=10, random_state=8)

        pd.testing.assert_frame_equal(first, second)
        with self.assertRaises(ValueError):
            inject_noise_regime_change(data, index_change=30)

    def test_noise_injection_uses_position_with_datetime_index(self):
        data = pd.DataFrame(
            {"x": [1.0, 2.0, 3.0, 4.0]},
            index=pd.date_range("2024-01-01", periods=4),
        )

        noisy = inject_noise_regime_change(data, index_change=2, random_state=8)

        pd.testing.assert_series_equal(noisy.iloc[:2, 0], data.iloc[:2, 0])
        self.assertEqual(noisy.index.tolist(), data.index.tolist())
        with self.assertRaises(ValueError):
            inject_noise_regime_change(data, index_change=2, noise_multiplier=-1.0)


if __name__ == "__main__":
    unittest.main()
