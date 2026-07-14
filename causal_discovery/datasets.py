from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_daily_delhi_climate(csv_path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path, parse_dates=["date"])
    data = data.set_index("date").sort_index()
    return data


def create_synthetic_dataset(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    if n_samples <= 0:
        raise ValueError("n_samples deve ser maior que zero.")

    rng = np.random.default_rng(seed)
    timeline = np.linspace(0, 100, n_samples)
    lag_1 = np.zeros_like(timeline)
    lag_2 = np.zeros_like(timeline)
    lag_1[1:] = timeline[:-1]
    lag_2[2:] = timeline[:-2]
    return pd.DataFrame(
        {
            "A": 0.5 * timeline + np.sin(timeline * 2 * np.pi / 12) + rng.normal(size=n_samples),
            "B": -0.3 * timeline + 0.4 * lag_1 + rng.normal(size=n_samples),
            "C": 0.6 * lag_2 + rng.normal(size=n_samples),
        },
        index=pd.date_range("2000-01-01", periods=n_samples, freq="ME"),
    )
