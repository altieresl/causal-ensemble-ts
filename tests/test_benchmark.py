import unittest

import pandas as pd

from causal_discovery.benchmark import (
    compute_structural_metrics,
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

    def test_noise_injection_is_reproducible_and_validates_index(self):
        data, _ = generate_synthetic_timeseries(n_samples=30, random_state=5)
        first = inject_noise_regime_change(data, index_change=10, random_state=8)
        second = inject_noise_regime_change(data, index_change=10, random_state=8)

        pd.testing.assert_frame_equal(first, second)
        with self.assertRaises(ValueError):
            inject_noise_regime_change(data, index_change=30)


if __name__ == "__main__":
    unittest.main()
