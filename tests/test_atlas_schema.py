from __future__ import annotations

import unittest

from causal_algorithms_atlas.schema import (
    Assumption,
    AlgorithmCard,
    AlgorithmFamily,
    DataRequirements,
    KNOWN_ASSUMPTIONS,
    OutputType,
    SampleType,
    SchemaError,
    TemporalHandling,
    VerificationStatus,
)


def _make_card(**overrides):
    defaults = dict(
        id="toy_method",
        name="Toy Method",
        aliases=(),
        family=AlgorithmFamily.CONSTRAINT_BASED,
        temporal_handling=TemporalHandling.NATIVE,
        output_type=OutputType.DAG,
        assumptions=(Assumption(id="stationarity", required=True, statement="x"),),
        handles_latent_confounders=False,
        handles_nonlinearity=False,
        handles_contemporaneous_effects=False,
        data_requirements=DataRequirements(
            min_variables=2, min_timepoints=None, sample_type=SampleType.SINGLE_SERIES
        ),
        implemented_in_framework=False,
        framework_method_name=None,
        references=("toyref",),
        verification=VerificationStatus.DRAFT,
        verified_by=None,
        last_reviewed="2026-09-07",
        sections={"Ideia central": "texto"},
        source_path="causal_algorithms_atlas/algorithms/toy_method.md",
    )
    defaults.update(overrides)
    return AlgorithmCard(**defaults)


class KnownAssumptionsTests(unittest.TestCase):
    def test_contains_core_assumptions(self):
        expected = {
            "causal_sufficiency",
            "stationarity",
            "linearity",
            "acyclicity_instantaneous",
            "non_gaussian_errors",
            "faithfulness",
            "markov_condition",
            "no_selection_bias",
            "deterministic_dynamics",
        }
        self.assertEqual(KNOWN_ASSUMPTIONS, frozenset(expected))


class AlgorithmCardTests(unittest.TestCase):
    def test_builds_with_valid_fields(self):
        card = _make_card()
        self.assertEqual(card.id, "toy_method")
        self.assertEqual(card.family, AlgorithmFamily.CONSTRAINT_BASED)

    def test_rejects_empty_id(self):
        with self.assertRaises(SchemaError):
            _make_card(id="")

    def test_rejects_verified_without_verified_by(self):
        with self.assertRaises(SchemaError):
            _make_card(verification=VerificationStatus.VERIFIED, verified_by=None)

    def test_accepts_verified_with_verified_by(self):
        card = _make_card(
            verification=VerificationStatus.VERIFIED, verified_by="paper-cross-check"
        )
        self.assertEqual(card.verification, VerificationStatus.VERIFIED)

    def test_rejects_unknown_assumption_id(self):
        with self.assertRaises(SchemaError):
            _make_card(
                assumptions=(Assumption(id="not_a_real_assumption", required=True, statement="x"),)
            )


if __name__ == "__main__":
    unittest.main()
