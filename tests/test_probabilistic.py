import unittest

from causal_discovery.probabilistic import (
    combine_p_values_fisher,
    score_to_probability,
    wilson_support_interval,
)


class ProbabilisticTests(unittest.TestCase):
    def test_fisher_combination_ignores_invalid_p_values(self):
        combined = combine_p_values_fisher([0.05, None, -1.0, 2.0])

        self.assertGreater(combined, 0.0)
        self.assertLessEqual(combined, 1.0)

    def test_score_to_probability_is_bounded(self):
        self.assertGreater(score_to_probability(10.0, scale=1.0), 0.5)
        self.assertGreater(score_to_probability(float("nan")), 0.0)
        self.assertLess(score_to_probability(float("nan")), 1.0)

    def test_wilson_interval_clips_votes_and_confidence_level(self):
        low, high = wilson_support_interval(10, 2, confidence_level=2.0)

        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertLessEqual(low, high)


if __name__ == "__main__":
    unittest.main()
