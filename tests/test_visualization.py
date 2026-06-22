import unittest
from unittest.mock import patch

import pandas as pd

from causal_discovery.visualization import create_advanced_expert_dashboard


class AdvancedDashboardTests(unittest.TestCase):
    @patch("IPython.display.display")
    def test_dashboard_preserves_initial_state(self, _display):
        rules = [
            {
                "source": "x",
                "target": "y",
                "lag": 1,
                "relation": "strong",
                "constraint": "soft",
                "confidence": 0.8,
            }
        ]

        dashboard = create_advanced_expert_dashboard(
            processed_data=pd.DataFrame({"x": [1, 2], "y": [2, 3]}),
            candidate_methods={"A": lambda data: pd.DataFrame()},
            candidate_method_kwargs={"A": {}},
            method_weights={"A": 1.0},
            all_nodes=["x", "y"],
            pipeline_callback=lambda **kwargs: (pd.DataFrame(), pd.DataFrame()),
            initial_expert_knowledge=rules,
            initial_quick_mode=False,
            initial_n_bootstrap=9,
            initial_parallel_jobs=3,
        )

        self.assertFalse(dashboard.quick_mode_control.value)
        self.assertEqual(dashboard.bootstrap_control.value, 9)
        self.assertEqual(dashboard.parallel_jobs_control.value, 3)
        self.assertEqual(dashboard.current_rules, rules)
        self.assertIsNone(dashboard.pipeline_result)

    @patch("IPython.display.display")
    def test_dashboard_accepts_rule_with_same_source_and_target(self, _display):
        dashboard = create_advanced_expert_dashboard(
            processed_data=pd.DataFrame({"x": [1, 2]}),
            candidate_methods={"A": lambda data: pd.DataFrame()},
            candidate_method_kwargs={"A": {}},
            method_weights={"A": 1.0},
            all_nodes=["x"],
            pipeline_callback=lambda **kwargs: (pd.DataFrame(), pd.DataFrame()),
        )

        dashboard.expert_source_control.value = "x"
        dashboard.expert_target_control.value = "x"
        dashboard.expert_lag_control.value = 1
        dashboard.add_expert_rule_button.click()

        self.assertEqual(len(dashboard.current_rules), 1)
        self.assertEqual(dashboard.current_rules[0]["source"], "x")
        self.assertEqual(dashboard.current_rules[0]["target"], "x")
        self.assertEqual(dashboard.current_rules[0]["lag"], 1)


if __name__ == "__main__":
    unittest.main()
