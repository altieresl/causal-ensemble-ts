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

    def test_csv_loader_with_ground_truth_parses_edges_and_filters_indirect_rows(self):
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            data_path = directory / "toy.csv"
            gt_path = directory / "toy_gt.csv"

            pd.DataFrame(
                {
                    "Y": [0.0, 0.1],
                    "X1": [0.0, 0.2],
                    "X3": [0.0, 0.3],
                    "X4": [0.0, 0.4],
                    "X0": [0.0, 0.5],
                }
            ).to_csv(data_path, index=False)

            pd.DataFrame(
                [
                    {"Edge": "X1 → Y", "Direct": True, "Coefficient": 0.7, "Lag": 1.0, "Type": "linear"},
                    {"Edge": "X3 → Y", "Direct": True, "Coefficient": 0.3, "Lag": 1.0, "Type": "linear"},
                    {"Edge": "X4 → X1", "Direct": True, "Coefficient": 0.6, "Lag": 1.0, "Type": "linear"},
                    {"Edge": "X4 → Y", "Direct": False, "Coefficient": None, "Lag": None, "Type": "indirect via X1"},
                    {"Edge": "X0 → Y", "Direct": False, "Coefficient": 0.0, "Lag": None, "Type": "none (noise)"},
                ]
            ).to_csv(gt_path, index=False)

            loaded = load_time_series_dataset(
                data_path,
                data_format="csv",
                ground_truth_path=gt_path,
                selected_columns=["Y", "X1", "X3", "X4"],
            )

            self.assertEqual(loaded.selected_columns, ("Y", "X1", "X3", "X4"))
            self.assertEqual(len(loaded.ground_truth), 3)
            self.assertEqual(
                set(zip(loaded.ground_truth["source"], loaded.ground_truth["target"])),
                {("X1", "Y"), ("X3", "Y"), ("X4", "X1")},
            )
            self.assertTrue((loaded.ground_truth["lag"] == 1).all())
            self.assertEqual(loaded.ground_truth["lag"].dtype.name, "Int64")

            excluding_x4 = load_time_series_dataset(
                data_path,
                data_format="csv",
                ground_truth_path=gt_path,
                selected_columns=["Y", "X1", "X3"],
            )
            self.assertEqual(
                set(zip(excluding_x4.ground_truth["source"], excluding_x4.ground_truth["target"])),
                {("X1", "Y"), ("X3", "Y")},
            )

        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            dummy_data_path = directory / "gen_data.npy"
            np.save(dummy_data_path, np.zeros((1, 1, 1)))
            with self.assertRaises(ValueError):
                load_time_series_dataset(
                    dummy_data_path,
                    data_format="causaltime",
                    ground_truth_path=directory / "toy_gt.csv",
                )

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
            np.testing.assert_array_equal(
                loaded.selected_trajectories(),
                generated[:, :, [0, 1]],
            )
            np.testing.assert_array_equal(
                loaded.observed_trajectories(),
                generated[:, :, :3],
            )


if __name__ == "__main__":
    unittest.main()
