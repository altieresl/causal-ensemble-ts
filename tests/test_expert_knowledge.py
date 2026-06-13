import unittest

import pandas as pd

from causal_discovery.expert_knowledge import (
    apply_expert_knowledge_to_summary,
    normalize_expert_knowledge,
)


class ExpertKnowledgeTests(unittest.TestCase):
    def test_inverse_rule_is_preserved_and_marks_negative_effect(self):
        rules = [
            {
                "source": "x",
                "target": "y",
                "lag": 1,
                "relation": "inverse",
                "constraint": "soft",
                "confidence": 0.8,
            }
        ]
        summary = pd.DataFrame(
            [{"source": "x", "target": "y", "lag": 1, "edge_probability": 0.4, "uncertainty": 0.6}]
        )

        normalized = normalize_expert_knowledge(rules)
        adjusted = apply_expert_knowledge_to_summary(summary, rules)

        self.assertEqual(normalized.loc[0, "relation"], "inverse")
        self.assertEqual(adjusted.loc[0, "expert_adjustment"], "soft_inverse")
        self.assertEqual(adjusted.loc[0, "expert_effect"], "negative")
        self.assertGreater(adjusted.loc[0, "edge_probability"], 0.4)

    def test_hard_none_rule_removes_matching_edge(self):
        rules = [
            {
                "source": "x",
                "target": "y",
                "lag": 1,
                "relation": "none",
                "constraint": "hard",
                "confidence": 1.0,
            }
        ]
        summary = pd.DataFrame(
            [
                {"source": "x", "target": "y", "lag": 1, "edge_probability": 0.8, "uncertainty": 0.2},
                {"source": "a", "target": "b", "lag": 1, "edge_probability": 0.8, "uncertainty": 0.2},
            ]
        )

        adjusted = apply_expert_knowledge_to_summary(summary, rules)

        self.assertEqual(len(adjusted), 1)
        self.assertEqual(adjusted.loc[0, "source"], "a")
        self.assertEqual(adjusted.loc[0, "target"], "b")

    def test_soft_none_rule_reduces_probability_without_removing_edge(self):
        rules = [
            {
                "source": "x",
                "target": "y",
                "lag": 1,
                "relation": "none",
                "constraint": "soft",
                "confidence": 0.5,
            }
        ]
        summary = pd.DataFrame(
            [{"source": "x", "target": "y", "lag": 1, "edge_probability": 0.8, "uncertainty": 0.2}]
        )

        adjusted = apply_expert_knowledge_to_summary(summary, rules)

        self.assertEqual(len(adjusted), 1)
        self.assertLess(adjusted.loc[0, "edge_probability"], 0.8)
        self.assertEqual(adjusted.loc[0, "expert_effect"], "forbidden")


if __name__ == "__main__":
    unittest.main()
