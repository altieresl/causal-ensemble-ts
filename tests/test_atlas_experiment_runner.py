from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from causal_algorithms_atlas.experiment_runner import (
    InsufficientCandidatesError,
    run_experiment,
)


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


def _tiny_nonstationary_linear_dataset(
    n: int = 150, seed: int = 11
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """x e um passeio aleatorio (nao estacionario); y depende linearmente do lag de x.

    Usado para exercitar o filtro de premissas: PCMCI exige estacionariedade (violada
    aqui), DYNOTEARS nao (permanece compativel) -- par ideal pra testar que o pool
    ampliado (``use_assumption_soft_filter=True``) mantem PCMCI como candidato
    flagueado, enquanto o filtro rigido (``False``) o excluiria.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] + rng.normal(0, 0.3)
        y[t] = 0.4 * y[t - 1] + 0.5 * x[t - 1] + rng.normal(0, 0.3)
    data = pd.DataFrame({"x": x, "y": y})
    ground_truth = pd.DataFrame([{"source": "x", "target": "y", "lag": 1}])
    return data, ground_truth


_TWO_CANDIDATES = {"ClassicalGranger", "PCMCI"}
_STATIONARITY_MIXED_CANDIDATES = {"PCMCI", "DYNOTEARS"}


class RunExperimentTests(unittest.TestCase):
    def test_runs_end_to_end_and_returns_no_leakage_markers(self):
        data, ground_truth = _tiny_linear_dataset()
        result = run_experiment(
            data,
            ground_truth=ground_truth,
            dataset_name="toy_unit_test",
            candidate_method_names=_TWO_CANDIDATES,
            n_bootstrap=2,
            max_methods=2,
            random_state=1,
        )
        self.assertEqual(result["dataset_name"], "toy_unit_test")
        self.assertIn("profile", result)
        self.assertIn("recommendations", result)
        self.assertIn("best_combination", result)
        self.assertIn("best_combination_methods", result)
        self.assertIn("best_single_method", result)
        self.assertIn("single_method_performance_scores", result)
        self.assertIn("best_combination_metrics_post_hoc", result)
        self.assertIn("best_single_metrics_post_hoc", result)
        self.assertTrue(result["ground_truth_used_only_for_post_hoc_evaluation"])
        self.assertEqual(
            set(result["single_method_performance_scores"]), _TWO_CANDIDATES
        )

    def test_single_method_baseline_uses_min_votes_one_not_two(self):
        # Sob min_votes=2 (o padrao do ensemble), um metodo sozinho nunca teria
        # nenhuma aresta -- por isso o baseline precisa da sua propria reavaliacao
        # com min_votes=1, e deve ser capaz de reportar arestas (num_edges > 0
        # em pelo menos um dos dois candidatos, dado dados lineares claros).
        data, _ = _tiny_linear_dataset(n=200)
        result = run_experiment(
            data,
            ground_truth=None,
            dataset_name="toy_unit_test",
            candidate_method_names=_TWO_CANDIDATES,
            n_bootstrap=2,
            max_methods=2,
            random_state=1,
        )
        scores = result["single_method_performance_scores"]
        self.assertTrue(any(score > 0.0 for score in scores.values()))

    def test_writes_history_entry_to_jsonl_file(self):
        data, ground_truth = _tiny_linear_dataset()
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "history.jsonl"
            run_experiment(
                data,
                ground_truth=ground_truth,
                dataset_name="toy_unit_test",
                candidate_method_names=_TWO_CANDIDATES,
                n_bootstrap=2,
                max_methods=2,
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
            candidate_method_names=_TWO_CANDIDATES,
            n_bootstrap=2,
            max_methods=2,
            random_state=1,
        )
        self.assertIsNone(result["best_combination_metrics_post_hoc"])
        self.assertIsNone(result["best_single_metrics_post_hoc"])

    def test_raises_clear_error_when_fewer_than_two_candidates_survive_filter(self):
        data, _ = _tiny_linear_dataset()
        with self.assertRaises(InsufficientCandidatesError):
            run_experiment(
                data,
                ground_truth=None,
                dataset_name="toy_single_candidate",
                candidate_method_names={"NeuralGrangercMLP"},
                n_bootstrap=2,
                random_state=1,
            )

    def test_insufficient_candidates_is_still_logged_to_history(self):
        data, _ = _tiny_linear_dataset()
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "history.jsonl"
            with self.assertRaises(InsufficientCandidatesError):
                run_experiment(
                    data,
                    ground_truth=None,
                    dataset_name="toy_single_candidate",
                    candidate_method_names={"NeuralGrangercMLP"},
                    n_bootstrap=2,
                    random_state=1,
                    history_path=history_path,
                )
            lines = history_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["outcome"], "insufficient_candidates")
            self.assertEqual(record["dataset_name"], "toy_single_candidate")


class AssumptionSoftFilterTests(unittest.TestCase):
    def test_soft_filter_keeps_assumption_violating_method_as_flagged_candidate(self):
        data, ground_truth = _tiny_nonstationary_linear_dataset()
        result = run_experiment(
            data,
            ground_truth=ground_truth,
            dataset_name="toy_soft_filter",
            candidate_method_names=_STATIONARITY_MIXED_CANDIDATES,
            use_assumption_soft_filter=True,
            n_bootstrap=2,
            max_methods=2,
            random_state=1,
        )
        self.assertEqual(set(result["candidate_methods"]), _STATIONARITY_MIXED_CANDIDATES)
        self.assertIn("PCMCI", result["assumption_flags"])
        self.assertNotIn("DYNOTEARS", result["assumption_flags"])

    def test_rigid_filter_excludes_assumption_violating_method(self):
        data, _ = _tiny_nonstationary_linear_dataset()
        with self.assertRaises(InsufficientCandidatesError):
            run_experiment(
                data,
                ground_truth=None,
                dataset_name="toy_rigid_filter",
                candidate_method_names=_STATIONARITY_MIXED_CANDIDATES,
                use_assumption_soft_filter=False,
                n_bootstrap=2,
                random_state=1,
            )


if __name__ == "__main__":
    unittest.main()
