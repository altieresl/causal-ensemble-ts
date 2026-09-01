from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


UNKNOWN_LAG_EDGE_COLUMNS = ["source", "target", "lag"]


@dataclass(frozen=True)
class TimeSeriesDataset:
    """Dados temporais carregados com metadados independentes do formato de origem."""

    data: pd.DataFrame
    available_columns: tuple[str, ...]
    selected_columns: tuple[str, ...]
    source_format: str
    ground_truth: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=UNKNOWN_LAG_EDGE_COLUMNS)
    )
    trajectories: np.ndarray | None = field(default=None, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def trajectory_count(self) -> int:
        if self.trajectories is None:
            return 1
        return int(self.trajectories.shape[0])

    def trajectory_frame(self, index: int) -> pd.DataFrame:
        """Retorna uma trajetória observada sem misturar fronteiras temporais."""
        if self.trajectories is None:
            if index != 0:
                raise IndexError("Datasets tabulares possuem somente a trajetória 0.")
            return self.data.copy()

        if not 0 <= index < self.trajectory_count:
            raise IndexError(
                f"trajectory_index deve estar entre 0 e {self.trajectory_count - 1}."
            )

        selected_indices = [
            self.available_columns.index(column)
            for column in self.selected_columns
        ]
        values = self.trajectories[index][:, selected_indices]
        return pd.DataFrame(
            values,
            columns=list(self.selected_columns),
            index=pd.RangeIndex(values.shape[0], name="time"),
        )

    def selected_trajectories(self) -> np.ndarray:
        """Retorna todas as trajetorias restritas as colunas selecionadas."""
        if self.trajectories is None:
            return self.data.to_numpy(dtype=float, copy=True)[None, :, :]

        selected_indices = [
            self.available_columns.index(column)
            for column in self.selected_columns
        ]
        return np.asarray(self.trajectories[:, :, selected_indices]).copy()

    def observed_trajectories(self) -> np.ndarray:
        """Retorna todos os nos observados para uso como contexto de ajuste."""
        if self.trajectories is None:
            return self.data.to_numpy(dtype=float, copy=True)[None, :, :]
        return np.asarray(self.trajectories).copy()


def _normalize_selected_columns(
    available_columns: Sequence[str],
    selected_columns: Sequence[str] | None,
) -> tuple[str, ...]:
    available = tuple(str(column) for column in available_columns)
    if not available:
        raise ValueError("O dataset não possui colunas numéricas disponíveis.")

    if selected_columns is None:
        return available

    selected = tuple(dict.fromkeys(str(column) for column in selected_columns))
    if not selected:
        raise ValueError("selected_columns não pode ser uma sequência vazia.")

    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(
            "Colunas selecionadas não encontradas no dataset: "
            f"{unknown}. Disponíveis: {list(available)}"
        )
    return selected


def _parse_toy_ground_truth_edge(edge: object) -> tuple[str, str]:
    """Separa uma aresta no formato ``'ORIGEM → DESTINO'`` em (origem, destino)."""
    parts = [part.strip() for part in re.split(r"\s*→\s*", str(edge).strip())]
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Não foi possível interpretar a aresta do ground truth: {edge!r}. "
            "Formato esperado: 'ORIGEM → DESTINO'."
        )
    return parts[0], parts[1]


def _load_toy_ground_truth_csv(
    ground_truth_path: Path,
    selected_columns: Sequence[str],
) -> pd.DataFrame:
    """Carrega o ground truth no formato produzido por ``synthetic_causal_datasets.ipynb``.

    Espera as colunas ``Edge`` (ex.: ``'X1 → Y'``) e ``Direct`` (booleano), com uma
    coluna ``Lag`` opcional. Somente arestas com ``Direct == True`` entram no ground
    truth; as demais linhas apenas documentam relações indiretas ou inexistentes e
    não devem ser tratadas como arestas verdadeiras.
    """
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth não encontrado em {ground_truth_path}.")

    frame = pd.read_csv(ground_truth_path)
    required_columns = {"Edge", "Direct"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            "O CSV de ground truth não possui as colunas obrigatórias: "
            f"{sorted(missing_columns)}"
        )

    direct_mask = frame["Direct"].astype(bool)
    has_lag_column = "Lag" in frame.columns
    allowed = {str(column) for column in selected_columns}

    records = []
    for _, row in frame.loc[direct_mask].iterrows():
        source, target = _parse_toy_ground_truth_edge(row["Edge"])
        if source not in allowed or target not in allowed:
            continue
        lag_value = row["Lag"] if has_lag_column else pd.NA
        records.append(
            {
                "source": source,
                "target": target,
                "lag": pd.NA if pd.isna(lag_value) else int(lag_value),
            }
        )

    ground_truth = pd.DataFrame(records, columns=UNKNOWN_LAG_EDGE_COLUMNS)
    ground_truth["lag"] = pd.array(ground_truth["lag"], dtype="Int64")
    return ground_truth


def _load_csv_dataset(
    data_path: Path,
    *,
    selected_columns: Sequence[str] | None,
    date_column: str | None,
    ground_truth_path: Path | None = None,
) -> TimeSeriesDataset:
    parse_dates = [date_column] if date_column else None
    frame = pd.read_csv(data_path, parse_dates=parse_dates)
    if date_column:
        if date_column not in frame.columns:
            raise ValueError(f"Coluna temporal não encontrada: {date_column}")
        frame = frame.set_index(date_column).sort_index()

    available = tuple(frame.select_dtypes(include=[np.number]).columns.astype(str))
    selected = _normalize_selected_columns(available, selected_columns)
    data = frame.loc[:, list(selected)].copy()

    ground_truth = (
        _load_toy_ground_truth_csv(ground_truth_path, selected)
        if ground_truth_path is not None
        else pd.DataFrame(columns=UNKNOWN_LAG_EDGE_COLUMNS)
    )

    return TimeSeriesDataset(
        data=data,
        available_columns=available,
        selected_columns=selected,
        source_format="csv",
        ground_truth=ground_truth,
        metadata={
            "data_path": str(data_path),
            "date_column": date_column,
            "trajectory_index": 0,
            "ground_truth_path": (
                str(ground_truth_path) if ground_truth_path is not None else None
            ),
            "ground_truth_has_lag": ground_truth_path is not None,
        },
    )


def _matrix_to_ground_truth(
    graph: np.ndarray,
    column_names: Sequence[str],
    selected_columns: Sequence[str],
) -> pd.DataFrame:
    selected = set(selected_columns)
    source_indices, target_indices = np.where(np.asarray(graph) != 0)
    records = [
        {
            "source": column_names[source_index],
            "target": column_names[target_index],
            "lag": pd.NA,
        }
        for source_index, target_index in zip(source_indices, target_indices)
        if source_index != target_index
        and column_names[source_index] in selected
        and column_names[target_index] in selected
    ]
    ground_truth = pd.DataFrame(records, columns=UNKNOWN_LAG_EDGE_COLUMNS)
    ground_truth["lag"] = pd.array(ground_truth["lag"], dtype="Int64")
    return ground_truth


def _load_causaltime_dataset(
    data_path: Path,
    *,
    graph_path: Path | None,
    selected_columns: Sequence[str] | None,
    trajectory_index: int,
    column_names: Sequence[str] | None,
    column_prefix: str,
) -> TimeSeriesDataset:
    generated = np.load(data_path, allow_pickle=False)
    if generated.ndim != 3:
        raise ValueError(
            "O CausalTime deve ter shape "
            "(trajetórias, instantes, 2 * número_de_nós)."
        )

    resolved_graph_path = graph_path or data_path.with_name("graph.npy")
    if not resolved_graph_path.exists():
        raise FileNotFoundError(
            f"Ground truth não encontrado em {resolved_graph_path}."
        )
    graph = np.load(resolved_graph_path, allow_pickle=False)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("graph.npy deve ser uma matriz quadrada.")

    node_count = int(graph.shape[0])
    if generated.shape[2] < node_count:
        raise ValueError(
            "gen_data.npy possui menos canais que o número de nós do grafo."
        )

    if column_names is None:
        width = max(2, len(str(max(node_count - 1, 0))))
        available = tuple(
            f"{column_prefix}_{index:0{width}d}"
            for index in range(node_count)
        )
    else:
        available = tuple(str(column) for column in column_names)
        if len(available) != node_count:
            raise ValueError(
                "column_names deve possuir exatamente "
                f"{node_count} nomes."
            )

    selected = _normalize_selected_columns(available, selected_columns)
    observed_trajectories = np.asarray(generated[:, :, :node_count])
    if not 0 <= trajectory_index < observed_trajectories.shape[0]:
        raise IndexError(
            "trajectory_index deve estar entre 0 e "
            f"{observed_trajectories.shape[0] - 1}."
        )

    selected_indices = [available.index(column) for column in selected]
    values = observed_trajectories[trajectory_index][:, selected_indices]
    data = pd.DataFrame(
        values,
        columns=list(selected),
        index=pd.RangeIndex(values.shape[0], name="time"),
    )
    ground_truth = _matrix_to_ground_truth(graph, available, selected)
    return TimeSeriesDataset(
        data=data,
        available_columns=available,
        selected_columns=selected,
        source_format="causaltime",
        ground_truth=ground_truth,
        trajectories=observed_trajectories,
        metadata={
            "data_path": str(data_path),
            "graph_path": str(resolved_graph_path),
            "trajectory_index": int(trajectory_index),
            "trajectory_count": int(observed_trajectories.shape[0]),
            "trajectory_length": int(observed_trajectories.shape[1]),
            "observed_node_count": node_count,
            "auxiliary_channel_count": int(generated.shape[2] - node_count),
            "ground_truth_has_lag": False,
            "ground_truth_is_symmetric": bool(np.array_equal(graph, graph.T)),
        },
    )


def load_time_series_dataset(
    data_path: str | Path,
    *,
    data_format: str = "auto",
    selected_columns: Sequence[str] | None = None,
    date_column: str | None = None,
    ground_truth_path: str | Path | None = None,
    graph_path: str | Path | None = None,
    trajectory_index: int = 0,
    column_names: Sequence[str] | None = None,
    column_prefix: str = "variable",
) -> TimeSeriesDataset:
    """Carrega CSV ou CausalTime NPY por uma API única.

    ``selected_columns=None`` seleciona dinamicamente todas as variáveis numéricas
    do CSV ou todos os nós observados do CausalTime. ``ground_truth_path`` é
    exclusivo de ``data_format='csv'`` e aponta para um CSV com colunas ``Edge``
    (formato ``'ORIGEM → DESTINO'``) e ``Direct`` — como o gerado por
    ``synthetic_causal_datasets.ipynb`` — usado para anexar arestas causais
    conhecidas a um dataset tabular. O CausalTime usa ``graph_path`` para o mesmo
    propósito.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    normalized_format = str(data_format).strip().lower()
    if normalized_format == "auto":
        normalized_format = "csv" if path.suffix.lower() == ".csv" else "causaltime"

    if normalized_format == "csv":
        return _load_csv_dataset(
            path,
            selected_columns=selected_columns,
            date_column=date_column,
            ground_truth_path=(
                Path(ground_truth_path) if ground_truth_path is not None else None
            ),
        )
    if normalized_format in {"causaltime", "npy"}:
        if ground_truth_path is not None:
            raise ValueError(
                "ground_truth_path só é aplicável a data_format='csv'; "
                "o CausalTime usa graph_path."
            )
        return _load_causaltime_dataset(
            path,
            graph_path=Path(graph_path) if graph_path is not None else None,
            selected_columns=selected_columns,
            trajectory_index=int(trajectory_index),
            column_names=column_names,
            column_prefix=str(column_prefix),
        )
    raise ValueError("data_format deve ser 'auto', 'csv', 'causaltime' ou 'npy'.")


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
