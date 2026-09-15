from __future__ import annotations

import unittest
from unittest.mock import patch

from causal_algorithms_atlas.dataset_profile import DatasetProfile, VariableProfile
from causal_algorithms_atlas.ensemble_advisor import (
    explain_recommendation,
    recommend_framework_methods,
    select_candidate_methods,
    select_candidate_methods_with_assumption_flags,
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
            nonlinearity_effect_size=0.01 if linear_value else 0.9,
            non_gaussian=True,
            residual_skewness=0.0,
            residual_excess_kurtosis=2.0,
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

    def test_unknown_causal_sufficiency_does_not_exclude_by_default(self):
        # declared_causal_sufficiency=None (default): comportamento identico ao de
        # antes desse parametro existir -- GES/PCMCI exigem suficiencia causal, mas
        # como ela nao e verificavel a partir dos dados, nao declarar nada NAO deve
        # virar uma exclusao (isso seria tratar "desconhecido" como "violado").
        profile = _profile(mostly_linear=True, mostly_stationary=True)
        recommendations = recommend_framework_methods(profile)
        by_name = {r.framework_method_name: r for r in recommendations}
        self.assertTrue(by_name["GES"].included)
        self.assertTrue(by_name["PCMCI"].included)
        self.assertTrue(
            any("nao e verificavel" in reason for reason in by_name["GES"].reasons)
        )

    def test_declared_causal_sufficiency_true_includes_with_reason(self):
        profile = _profile(mostly_linear=True, mostly_stationary=True)
        recommendations = recommend_framework_methods(
            profile, declared_causal_sufficiency=True
        )
        by_name = {r.framework_method_name: r for r in recommendations}
        self.assertTrue(by_name["GES"].included)
        self.assertTrue(
            any("declarada como satisfeita" in reason for reason in by_name["GES"].reasons)
        )

    def test_declared_causal_sufficiency_false_excludes_dependent_methods(self):
        profile = _profile(mostly_linear=True, mostly_stationary=True)
        recommendations = recommend_framework_methods(
            profile, declared_causal_sufficiency=False
        )
        by_name = {r.framework_method_name: r for r in recommendations}
        self.assertFalse(by_name["GES"].included)
        self.assertFalse(by_name["PCMCI"].included)
        self.assertTrue(
            any("declarada como violada" in reason for reason in by_name["GES"].reasons)
        )
        # FCI/LPCMCI nao exigem suficiencia causal -- nao devem ser afetados por essa
        # declaracao, seja qual for o valor.
        self.assertTrue(by_name["FCI"].included)
        self.assertTrue(by_name["LPCMCI"].included)

    def test_handles_latent_confounders_methods_carry_advisory_note(self):
        profile = _profile(mostly_linear=True, mostly_stationary=True)
        recommendations = recommend_framework_methods(profile)
        by_name = {r.framework_method_name: r for r in recommendations}
        self.assertTrue(
            any(
                "confundidores latentes nao verificados" in reason
                for reason in by_name["FCI"].reasons
            )
        )
        self.assertTrue(
            any(
                "confundidores latentes nao verificados" in reason
                for reason in by_name["LPCMCI"].reasons
            )
        )
        # DYNOTEARS nao declara suficiencia causal nem lida com confundidores latentes
        # -- nao deve ganhar nenhuma nota nova sobre o assunto.
        self.assertFalse(
            any(
                "confundidor" in reason.lower()
                for reason in by_name["DYNOTEARS"].reasons
            )
        )

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


class SelectCandidateMethodsWithAssumptionFlagsTests(unittest.TestCase):
    def test_keeps_assumption_violating_methods_as_flagged_candidates(self):
        profile = _profile(mostly_linear=False, mostly_stationary=True)
        recommendations = recommend_framework_methods(profile)
        flagged = select_candidate_methods_with_assumption_flags(recommendations)
        # ClassicalGranger exige linearidade e o perfil e nao-linear -- o filtro
        # rigido o excluiria (ver teste acima), mas o pool ampliado deve mante-lo,
        # com a violacao registrada em vez de escondida.
        self.assertIn("ClassicalGranger", flagged)
        fn, reasons = flagged["ClassicalGranger"]
        self.assertTrue(callable(fn))
        self.assertTrue(reasons)

    def test_compliant_methods_have_no_reasons(self):
        profile = _profile(mostly_linear=True, mostly_stationary=True)
        recommendations = recommend_framework_methods(profile)
        flagged = select_candidate_methods_with_assumption_flags(recommendations)
        _fn, reasons = flagged["ClassicalGranger"]
        self.assertEqual(reasons, ())

    def test_returns_every_registered_method_regardless_of_compliance(self):
        profile = _profile(mostly_linear=False, mostly_stationary=False)
        recommendations = recommend_framework_methods(profile)
        rigid = select_candidate_methods(recommendations)
        flagged = select_candidate_methods_with_assumption_flags(recommendations)
        self.assertGreaterEqual(len(flagged), len(rigid))
        self.assertEqual(set(flagged), {rec.framework_method_name for rec in recommendations})


class ExplainRecommendationTests(unittest.TestCase):
    def test_prompt_sent_to_ollama_cites_included_and_excluded_reasons(self):
        profile = _profile(mostly_linear=False, mostly_stationary=True)
        recommendations = recommend_framework_methods(profile)

        captured_prompts: list[str] = []

        def fake_call_ollama(prompt: str, **kwargs):
            captured_prompts.append(prompt)
            return "resposta simulada"

        with patch(
            "causal_algorithms_atlas.rag_chat.call_ollama", side_effect=fake_call_ollama
        ):
            answer = explain_recommendation(profile, recommendations)

        self.assertEqual(answer, "resposta simulada")
        self.assertEqual(len(captured_prompts), 1)
        prompt = captured_prompts[0]
        self.assertIn("NeuralGrangercMLP", prompt)
        self.assertIn("ClassicalGranger", prompt)
        self.assertIn("nao lineares previu bem melhor", prompt)


if __name__ == "__main__":
    unittest.main()
