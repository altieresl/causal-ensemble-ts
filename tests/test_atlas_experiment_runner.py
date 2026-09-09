from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from causal_algorithms_atlas.experiment_runner import run_experiment


def _tiny_linear_dataset(n: int = 60, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.normal(0, 0.5)
        y[t] = 0.4 * y[t - 1] + 0.6 * x[t - 1] + rng.normal(0, 0.5)
    data = pd.DataFrame({"x": x, "y": y})
    ground_truth = pd.DataFrame([{"source": "x", "target": "y", "lag": 1}])
    return data, ground_truth


class RunExperimentTests(unittest.TestCase):
    def test_runs_end_to_end_and_returns_no_leakage_markers(self):
        data, ground_truth = _tiny_linear_dataset()
        result = run_experiment(
            data,
            ground_truth=ground_truth,
            dataset_name="toy_unit_test",
            candidate_method_names={"ClassicalGranger"},
            n_bootstrap=2,
            max_methods=1,
            min_methods=1,
            random_state=1,
        )
        self.assertEqual(result["dataset_name"], "toy_unit_test")
        self.assertIn("profile", result)
        self.assertIn("recommendations", result)
        self.assertIn("best_combination", result)
        self.assertIn("best_combination_methods", result)
        self.assertIn("best_single_method", result)
        self.assertIn("best_combination_metrics_post_hoc", result)
        self.assertIn("best_single_metrics_post_hoc", result)
        self.assertTrue(result["ground_truth_used_only_for_post_hoc_evaluation"])

    def test_writes_history_entry_to_jsonl_file(self):
        data, ground_truth = _tiny_linear_dataset()
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "history.jsonl"
            run_experiment(
                data,
                ground_truth=ground_truth,
                dataset_name="toy_unit_test",
                candidate_method_names={"ClassicalGranger"},
                n_bootstrap=2,
                max_methods=1,
                min_methods=1,
                random_state=1,
                history_path=history_path,
            )
            lines = history_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["dataset_name"], "toy_unit_test")

    def test_without_ground_truth_only_reports_unsupervised_metrics(self):
        data, _ = _tiny_linear_dataset()
        result = run_experiment(
            data,
            ground_truth=None,
            dataset_name="toy_no_gt",
            candidate_method_names={"ClassicalGranger"},
            n_bootstrap=2,
            max_methods=1,
            min_methods=1,
            random_state=1,
        )
        self.assertIsNone(result["best_combination_metrics_post_hoc"])
        self.assertIsNone(result["best_single_metrics_post_hoc"])


if __name__ == "__main__":
    unittest.main()
