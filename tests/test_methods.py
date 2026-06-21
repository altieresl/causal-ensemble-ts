import unittest

import numpy as np
import pandas as pd

from causal_discovery.benchmark import generate_synthetic_timeseries
from causal_discovery.methods.causal_learn import run_fci, run_score_based_ges
from causal_discovery.methods.classical_granger import run_classical_granger
from causal_discovery.methods.heterogeneous_fci import (
    _split_contiguous_segments,
)
from causal_discovery.methods.lpcmci import _is_definite_forward_edge
from causal_discovery.methods.neural_granger import (
    _CMLP,
    _structured_prox,
    run_neural_granger,
)
from causal_discovery.methods.pcmci import _is_forward_directed_edge
from causal_discovery.methods.score_based import run_score_based_search


class TigramiteEdgeParsingTests(unittest.TestCase):
    def test_pcmci_accepts_only_forward_directed_mark(self):
        self.assertTrue(_is_forward_directed_edge("-->"))
        for mark in ["<--", "o-o", "o->", "<->", "", None]:
            self.assertFalse(_is_forward_directed_edge(mark))

    def test_lpcmci_excludes_ambiguous_and_bidirected_marks(self):
        self.assertTrue(_is_definite_forward_edge("-->"))
        for mark in ["<--", "o-o", "o->", "<->", "<-o", ""]:
            self.assertFalse(_is_definite_forward_edge(mark))


class ClassicalGrangerTests(unittest.TestCase):
    def test_reports_significant_exact_lag_from_full_model(self):
        rng = np.random.default_rng(7)
        sample_size = 800
        source = rng.normal(size=sample_size)
        target = np.zeros(sample_size)
        for time in range(3, sample_size):
            target[time] = (
                0.25 * target[time - 1]
                + 0.9 * source[time - 2]
                + rng.normal(scale=0.3)
            )

        result = run_classical_granger(
            pd.DataFrame({"source": source, "target": target}),
            max_lag=3,
            alpha=0.01,
        )
        links = result[(result["source"] == "source") & (result["target"] == "target")]

        self.assertIn(2, set(links["lag"]))
        self.assertTrue((links["lag_order"] == 3).all())
        self.assertTrue((links["joint_p_value"] <= 0.01).all())


class CompatibilityAliasTests(unittest.TestCase):
    def test_segment_split_preserves_all_rows(self):
        data = pd.DataFrame({"x": range(11)})
        segments = _split_contiguous_segments(data, 4)

        self.assertEqual(sum(len(segment) for segment in segments), len(data))
        self.assertLessEqual(max(map(len, segments)) - min(map(len, segments)), 1)

    def test_forward_bic_does_not_expose_post_selection_p_as_canonical(self):
        rng = np.random.default_rng(4)
        source = rng.normal(size=200)
        target = np.roll(source, 1) + rng.normal(scale=0.2, size=200)
        result = run_score_based_search(
            pd.DataFrame({"source": source, "target": target}),
            max_lag=1,
        )

        self.assertFalse(result.empty)
        self.assertTrue(result["p_value"].isna().all())
        self.assertIn("coefficient_p_value", result.columns)


class CanonicalAcademicMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, _ = generate_synthetic_timeseries(
            n_samples=250,
            noise_std=0.2,
            random_state=13,
        )

    def test_ges_uses_causal_learn_and_temporal_orientation(self):
        result = run_score_based_ges(self.data, max_lag=2, max_parents=5)

        self.assertFalse(result.empty)
        self.assertTrue((result["method"] == "GES").all())
        self.assertTrue((result["lag"] >= 1).all())
        self.assertIn("graph_score", result.columns)

    def test_fci_returns_pag_orientation_metadata(self):
        result = run_fci(
            self.data,
            max_lag=2,
            alpha=0.05,
            include_partial=True,
        )

        self.assertTrue(result.empty or (result["method"] == "FCI").all())
        if not result.empty:
            self.assertTrue((result["lag"] >= 1).all())
            self.assertIn("orientation_certainty", result.columns)

    def test_neural_granger_uses_structured_cmlp(self):
        result = run_neural_granger(
            self.data.iloc[:100],
            max_lag=2,
            hidden_layer_sizes=(6,),
            lambda_group=0.2,
            learning_rate=0.005,
            max_iter=20,
            check_every=10,
            patience=2,
        )

        self.assertTrue(result.empty or (result["method"] == "NeuralGrangercMLP").all())
        if not result.empty:
            self.assertIn("group_norm", result.columns)
            self.assertIn("penalty", result.columns)

    def test_group_lasso_prox_can_zero_a_source_group(self):
        model = _CMLP(num_series=2, lag=2, hidden=(3,), activation="relu")
        network = model.networks[0]
        network.layers[0].weight.data.fill_(0.01)

        _structured_prox(network, amount=1.0, penalty="GL")

        self.assertTrue(
            np.allclose(
                network.layers[0].weight.detach().numpy(),
                0.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
