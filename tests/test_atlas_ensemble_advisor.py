from __future__ import annotations

import unittest

from causal_algorithms_atlas.dataset_profile import DatasetProfile, VariableProfile
from causal_algorithms_atlas.ensemble_advisor import (
    recommend_framework_methods,
    select_candidate_methods,
)


def _profile(*, mostly_linear: bool, mostly_stationary: bool) -> DatasetProfile:
    linear_value = True if mostly_linear else False
    stationary_value = True if mostly_stationary else False
    variables = (
        VariableProfile(
            name="x",
            stationary=stationary_value,
            adf_p_value=0.01 if stationary_value else 0.9,
            linear=linear_value,
            reset_p_value=0.9 if linear_value else 0.01,
        ),
    )
    return DatasetProfile(n_variables=1, n_timepoints=300, variables=variables)


class RecommendFrameworkMethodsTests(unittest.TestCase):
    def test_linear_stationary_profile_includes_linear_methods(self):
        profile = _profile(mostly_linear=True, mostly_stationary=True)
        recommendations = recommend_framework_methods(profile)
        by_name = {r.framework_method_name: r for r in recommendations}
        self.assertTrue(by_name["ClassicalGranger"].included)
        self.assertTrue(by_name["VARLiNGAM"].included)
        self.assertTrue(by_name["PCMCI"].included)

    def test_nonlinear_profile_excludes_linear_only_methods(self):
        profile = _profile(mostly_linear=False, mostly_stationary=True)
        recommendations = recommend_framework_methods(profile)
        by_name = {r.framework_method_name: r for r in recommendations}
        self.assertFalse(by_name["ClassicalGranger"].included)
        self.assertFalse(by_name["VARLiNGAM"].included)
        self.assertFalse(by_name["PCMCI"].included)
        self.assertTrue(by_name["NeuralGrangercMLP"].included)

    def test_non_stationary_profile_excludes_stationarity_dependent_methods(self):
        profile = _profile(mostly_linear=True, mostly_stationary=False)
        recommendations = recommend_framework_methods(profile)
        by_name = {r.framework_method_name: r for r in recommendations}
        self.assertFalse(by_name["PCMCI"].included)
        self.assertFalse(by_name["LPCMCI"].included)

    def test_every_recommendation_carries_a_reason(self):
        profile = _profile(mostly_linear=True, mostly_stationary=True)
        recommendations = recommend_framework_methods(profile)
        for rec in recommendations:
            self.assertTrue(rec.reasons)

    def test_covers_all_eight_framework_methods(self):
        profile = _profile(mostly_linear=True, mostly_stationary=True)
        recommendations = recommend_framework_methods(profile)
        names = {r.framework_method_name for r in recommendations}
        self.assertEqual(
            names,
            {
                "PCMCI",
                "LPCMCI",
                "ClassicalGranger",
                "NeuralGrangercMLP",
                "VARLiNGAM",
                "DYNOTEARS",
                "GES",
                "FCI",
            },
        )


class SelectCandidateMethodsTests(unittest.TestCase):
    def test_returns_only_included_methods_as_callables(self):
        profile = _profile(mostly_linear=False, mostly_stationary=True)
        recommendations = recommend_framework_methods(profile)
        methods = select_candidate_methods(recommendations)
        self.assertIn("NeuralGrangercMLP", methods)
        self.assertNotIn("ClassicalGranger", methods)
        for callable_method in methods.values():
            self.assertTrue(callable(callable_method))


if __name__ == "__main__":
    unittest.main()
