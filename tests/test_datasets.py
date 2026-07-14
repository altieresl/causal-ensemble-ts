import unittest

import numpy as np

from causal_discovery.datasets import create_synthetic_dataset


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


if __name__ == "__main__":
    unittest.main()
