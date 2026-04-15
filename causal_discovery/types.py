from __future__ import annotations

from typing import Iterable

import pandas as pd


CANONICAL_COLUMNS = ["source", "target", "lag", "score", "p_value", "method"]


def canonical_links_to_dataframe(records: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(list(records))
    if frame.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    for column in CANONICAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    leading_columns = CANONICAL_COLUMNS + [
        column for column in frame.columns if column not in CANONICAL_COLUMNS
    ]
    return frame[leading_columns]