import unittest
from unittest.mock import patch

import pandas as pd

from causal_discovery.visualization import (
    create_advanced_expert_dashboard,
    plot_probabilistic_causal_graph,
    plot_temporal_dag,
)


class AdvancedDashboardTests(unittest.TestCase):
    def test_probabilistic_graph_draws_visible_self_loop(self):
        summary = pd.DataFrame(
            [
                {
                    "source": "x",
                    "target": "x",
                    "lag": 1,
                    "edge_probability": 0.8,
                    "confidence": 0.7,
                }
            ]
        )

        fig = plot_probabilistic_causal_graph(summary)
        coordinates = set(zip(fig.data[0].x, fig.data[0].y))

        self.assertGreater(len(coordinates), 2)

    @patch("causal_discovery.visualization._require_plotly")
    def test_plot_temporal_dag_returns_figure_for_empty_summary(self, mock_require_plotly):
        class FakeFigure:
            def update_layout(self, **_kwargs):
                return None

        class FakeGo:
            Figure = FakeFigure

        mock_require_plotly.return_value = (None, FakeGo)
        fig = plot_temporal_dag(pd.DataFrame())
        self.assertIsNotNone(fig)

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
        self.assertEqual(
            set(dashboard.relation_selection_control.value),
            {("x", "y"), ("y", "x")},
        )

    @patch("causal_discovery.visualization.create_interactive_ensemble_dashboard")
    @patch("IPython.display.display")
    def test_dashboard_passes_selected_directional_relations(
        self,
        _display,
        _interactive_dashboard,
    ):
        received = {}

        def pipeline_callback(**kwargs):
            received.update(kwargs)
            return pd.DataFrame(), pd.DataFrame()

        dashboard = create_advanced_expert_dashboard(
            processed_data=pd.DataFrame({"x": [1, 2], "y": [2, 3]}),
            candidate_methods={"A": lambda data: pd.DataFrame()},
            candidate_method_kwargs={"A": {}},
            method_weights={"A": 1.0},
            all_nodes=["x", "y"],
            pipeline_callback=pipeline_callback,
        )
        dashboard.relation_selection_control.value = (("x", "y"),)

        run_button = dashboard.children[2].children[1]
        run_button.click()

        self.assertEqual(received["selected_relations"], [("x", "y")])
        self.assertEqual(dashboard.selected_relations, [("x", "y")])

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
