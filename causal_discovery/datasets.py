from __future__ import annotations

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


def _load_csv_dataset(
    data_path: Path,
    *,
    selected_columns: Sequence[str] | None,
    date_column: str | None,
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
    return TimeSeriesDataset(
        data=data,
        available_columns=available,
        selected_columns=selected,
        source_format="csv",
        metadata={
            "data_path": str(data_path),
            "date_column": date_column,
            "trajectory_index": 0,
            "ground_truth_has_lag": False,
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
    graph_path: str | Path | None = None,
    trajectory_index: int = 0,
    column_names: Sequence[str] | None = None,
    column_prefix: str = "variable",
) -> TimeSeriesDataset:
    """Carrega CSV ou CausalTime NPY por uma API única.

    ``selected_columns=None`` seleciona dinamicamente todas as variáveis numéricas
    do CSV ou todos os nós observados do CausalTime.
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
        )
    if normalized_format in {"causaltime", "npy"}:
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
