from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = ["source", "target", "lag", "score", "p_value", "method"]


class MethodOutputValidationError(ValueError):
    """Raised when a method violates the canonical edge-table contract."""


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


def validate_method_output(
    output: object,
    *,
    method_name: str,
    data_columns: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Validate and normalize a method result at the ensemble boundary."""

    prefix = f"Saida invalida do metodo {method_name!r}:"
    if not isinstance(output, pd.DataFrame):
        raise MethodOutputValidationError(
            f"{prefix} esperado pandas.DataFrame, recebido {type(output).__name__}."
        )

    duplicated_columns = output.columns[output.columns.duplicated()].tolist()
    if duplicated_columns:
        raise MethodOutputValidationError(
            f"{prefix} nomes de colunas duplicados: {duplicated_columns}."
        )

    missing = [column for column in CANONICAL_COLUMNS if column not in output.columns]
    if missing:
        raise MethodOutputValidationError(
            f"{prefix} colunas canonicas ausentes: {missing}. "
            f"Esperado: {CANONICAL_COLUMNS}."
        )

    frame = output.copy()
    if frame.empty:
        return frame

    for column in ("source", "target", "method"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise MethodOutputValidationError(
                f"{prefix} a coluna {column!r} contem valores nulos ou vazios."
            )

    if data_columns is not None:
        allowed = set(data_columns)
        unknown = sorted(
            set(frame["source"]).union(frame["target"]) - allowed,
            key=str,
        )
        if unknown:
            raise MethodOutputValidationError(
                f"{prefix} source/target desconhecidos: {unknown}. "
                "Os nomes devem corresponder as colunas dos dados de entrada."
            )

    numeric_lag = pd.to_numeric(frame["lag"], errors="coerce")
    invalid_lag = (
        numeric_lag.isna()
        | ~np.isfinite(numeric_lag)
        | numeric_lag.lt(0)
        | numeric_lag.mod(1).ne(0)
    )
    if invalid_lag.any():
        raise MethodOutputValidationError(
            f"{prefix} 'lag' deve conter inteiros nao negativos."
        )
    frame["lag"] = numeric_lag.astype(int)

    numeric_score = pd.to_numeric(frame["score"], errors="coerce")
    if numeric_score.isna().any() or (~np.isfinite(numeric_score)).any():
        raise MethodOutputValidationError(
            f"{prefix} 'score' deve conter numeros finitos e nao nulos."
        )
    frame["score"] = numeric_score.astype(float)

    numeric_p_value = pd.to_numeric(frame["p_value"], errors="coerce")
    supplied_p_values = frame["p_value"].notna()
    invalid_p_value = supplied_p_values & (
        numeric_p_value.isna()
        | ~np.isfinite(numeric_p_value)
        | numeric_p_value.lt(0.0)
        | numeric_p_value.gt(1.0)
    )
    if invalid_p_value.any():
        raise MethodOutputValidationError(
            f"{prefix} 'p_value' deve estar entre 0 e 1 ou ser ausente."
        )
    frame["p_value"] = numeric_p_value.astype(float)
    return frame
