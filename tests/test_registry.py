import unittest

import pandas as pd

from causal_discovery import (
    causal_method,
    discover_causal_methods,
    get_registered_method_kwargs,
    get_registered_method_weights,
    get_registered_methods,
)


class CausalMethodRegistryTests(unittest.TestCase):
    def test_discovers_all_pipeline_methods(self):
        expected = {
            "PCMCI",
            "LPCMCI",
            "ClassicalGranger",
            "NeuralGrangercMLP",
            "VARLiNGAM",
            "DYNOTEARS",
            "GES",
            "FCI",
        }

        self.assertEqual(set(discover_causal_methods()), expected)
        self.assertEqual(set(get_registered_methods()), expected)

    def test_registers_new_method_without_editing_a_central_mapping(self):
        @causal_method()
        def run_example_plugin(data, max_lag):
            return pd.DataFrame(
                columns=["source", "target", "lag", "score", "p_value", "method"]
            )

        spec = run_example_plugin.__causal_method_spec__

        self.assertEqual(spec.name, "ExamplePlugin")
        self.assertIs(get_registered_methods()["ExamplePlugin"], run_example_plugin)

    def test_rejects_method_without_max_lag_parameter(self):
        with self.assertRaisesRegex(TypeError, "max_lag"):
            causal_method()(lambda data: pd.DataFrame())

    def test_builds_runtime_configuration_from_metadata(self):
        kwargs = get_registered_method_kwargs(max_lag=4)
        weights = get_registered_method_weights()

        self.assertTrue(all(config["max_lag"] == 4 for config in kwargs.values()))
        self.assertEqual(kwargs["NeuralGrangercMLP"]["max_iter"], 400)
        self.assertEqual(kwargs["DYNOTEARS"]["max_iter"], 50)
        self.assertTrue(all(weight == 1.0 for weight in weights.values()))


if __name__ == "__main__":
    unittest.main()
