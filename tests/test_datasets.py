import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from causal_discovery.datasets import (
    create_synthetic_dataset,
    load_time_series_dataset,
)


class DatasetTests(unittest.TestCase):
    def test_synthetic_lags_do_not_wrap_future_values_to_the_start(self):
        seed = 7
        sample_size = 3
        rng = np.random.default_rng(seed)
        rng.normal(size=sample_size)  # Ruido de A.
        noise_b = rng.normal(size=sample_size)
        noise_c = rng.normal(size=sample_size)

        data = create_synthetic_dataset(n_samples=sample_size, seed=seed)

        self.assertAlmostEqual(data.iloc[0]["B"], noise_b[0])
        self.assertAlmostEqual(data.iloc[0]["C"], noise_c[0])
        with self.assertRaises(ValueError):
            create_synthetic_dataset(n_samples=0, seed=seed)

    def test_generic_csv_loader_selects_numeric_columns_dynamically(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "series.csv"
            pd.DataFrame(
                {
                    "date": ["2024-01-01", "2024-01-02"],
                    "temperature": [20.0, 21.0],
                    "humidity": [70.0, 68.0],
                    "label": ["a", "b"],
                }
            ).to_csv(path, index=False)

            loaded = load_time_series_dataset(path, date_column="date")

            self.assertEqual(
                loaded.selected_columns,
                ("temperature", "humidity"),
            )
            self.assertEqual(loaded.data.shape, (2, 2))
            self.assertEqual(loaded.source_format, "csv")

    def test_causaltime_loader_uses_observed_nodes_and_filters_self_links(self):
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            generated = np.arange(2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6)
            graph = np.array(
                [
                    [1, 1, 0],
                    [0, 1, 1],
                    [0, 0, 1],
                ],
                dtype=float,
            )
            np.save(directory / "gen_data.npy", generated)
            np.save(directory / "graph.npy", graph)

            loaded = load_time_series_dataset(
                directory / "gen_data.npy",
                data_format="causaltime",
                trajectory_index=1,
                selected_columns=["traffic_00", "traffic_01"],
                column_prefix="traffic",
            )

            self.assertEqual(loaded.available_columns, ("traffic_00", "traffic_01", "traffic_02"))
            self.assertEqual(loaded.data.shape, (4, 2))
            self.assertEqual(loaded.trajectory_count, 2)
            self.assertEqual(len(loaded.ground_truth), 1)
            self.assertEqual(
                tuple(loaded.ground_truth.iloc[0][["source", "target"]]),
                ("traffic_00", "traffic_01"),
            )
            self.assertTrue(loaded.ground_truth["lag"].isna().all())
            pd.testing.assert_frame_equal(
                loaded.data,
                loaded.trajectory_frame(1),
            )


if __name__ == "__main__":
    unittest.main()
