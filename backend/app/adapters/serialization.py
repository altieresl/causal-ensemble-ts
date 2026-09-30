"""Conversao de objetos do nucleo (pandas/numpy) em JSON seguro."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(value: Any) -> Any:
    """Converte recursivamente para tipos JSON; NaN/inf viram ``None``."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, pd.DataFrame):
        return [to_jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return to_jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, str):
        return value
    if value is pd.NA or value is pd.NaT:
        return None
    return str(value)


def frame_records(frame: pd.DataFrame | None, columns: list[str] | None = None) -> list[dict]:
    """Registros de um DataFrame (opcionalmente so algumas colunas), tolerando vazio."""
    if frame is None or frame.empty:
        return []
    if columns is not None:
        frame = frame[[column for column in columns if column in frame.columns]]
    return to_jsonable(frame.reset_index(drop=True))
