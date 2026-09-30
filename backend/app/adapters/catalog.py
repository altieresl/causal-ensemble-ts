"""Catalogo de datasets: embutidos (configuracao vinda do notebook) e uploads."""

from __future__ import annotations

import io
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..domain import NotFoundError, is_safe_id


@dataclass(frozen=True)
class DatasetEntry:
    id: str
    name: str
    description: str
    origin: str  # "builtin" | "upload"
    loader_kwargs: dict[str, Any] = field(default_factory=dict)  # load_time_series_dataset
    default_max_lag: int = 2
    decomposition_period: int | None = None


_TOY_DIR = "datasets/synthetic_causal"
_TRAFFIC_COLUMNS = [f"traffic_{n:02d}" for n in (2, 4, 6, 8, 13, 19)]
_TOY_DESCRIPTION = "Dataset sintetico com grafo causal conhecido (usado so na validacao pos-hoc)."

_TOYS = [
    ("toy_a", "toy_a_linear", "Sintetico A - linear"),
    ("toy_b", "toy_b_nonlinear", "Sintetico B - nao linear (tanh)"),
    ("toy_c", "toy_c_nonstationary", "Sintetico C - nao estacionario"),
    ("toy_d", "toy_d_mixed_nonlinear", "Sintetico D - misto nao linear"),
    ("toy_e", "toy_e_boundary_mixed", "Sintetico E - fronteira mista"),
    ("toy_f", "toy_f_non_gaussian", "Sintetico F - nao gaussiano"),
]


def builtin_entries() -> list[DatasetEntry]:
    entries = [
        DatasetEntry(
            id="delhi_csv",
            name="Clima diario de Delhi",
            description="Serie diaria real (temperatura, umidade, vento, pressao). Sem ground truth.",
            origin="builtin",
            loader_kwargs={
                "data_path": "DailyDelhiClimateTrain.csv",
                "data_format": "csv",
                "date_column": "date",
                "selected_columns": ["meantemp", "humidity", "wind_speed", "meanpressure"],
            },
            default_max_lag=5,
            decomposition_period=30,
        ),
        DatasetEntry(
            id="causaltime_traffic",
            name="CausalTime - Traffic",
            description="Benchmark CausalTime (trajetoria 0, subgrafo de 6 nos) com grafo verdadeiro.",
            origin="builtin",
            loader_kwargs={
                "data_path": "datasets/causaltime/traffic/gen_data.npy",
                "data_format": "causaltime",
                "graph_path": "datasets/causaltime/traffic/graph.npy",
                "trajectory_index": 0,
                "column_prefix": "traffic",
                "selected_columns": _TRAFFIC_COLUMNS,
            },
            default_max_lag=2,
        ),
    ]
    for dataset_id, stem, title in _TOYS:
        entries.append(
            DatasetEntry(
                id=dataset_id,
                name=title,
                description=_TOY_DESCRIPTION,
                origin="builtin",
                loader_kwargs={
                    "data_path": f"{_TOY_DIR}/{stem}.csv",
                    "data_format": "csv",
                    "ground_truth_path": f"{_TOY_DIR}/{stem}_gt.csv",
                },
            )
        )
    return entries


class DatasetCatalog:
    """Resolve IDs para entradas. Nenhum caminho vem do cliente (anti path traversal)."""

    def __init__(self, repo_root: Path, uploads_dir: Path) -> None:
        self._repo_root = repo_root
        self._uploads_dir = uploads_dir
        self._builtin = {entry.id: entry for entry in builtin_entries()}

    def list(self) -> list[DatasetEntry]:
        return [*self._builtin.values(), *self._list_uploads()]

    def get(self, dataset_id: str) -> DatasetEntry:
        if dataset_id in self._builtin:
            return self._builtin[dataset_id]
        upload = self._read_upload_entry(dataset_id)
        if upload is None:
            raise NotFoundError(f"Dataset '{dataset_id}' nao existe.")
        return upload

    def resolve_kwargs(self, entry: DatasetEntry) -> dict[str, Any]:
        """kwargs de ``load_time_series_dataset`` com caminhos absolutos."""
        kwargs = dict(entry.loader_kwargs)
        if entry.origin == "builtin":
            for key in ("data_path", "graph_path", "ground_truth_path"):
                if kwargs.get(key) is not None:
                    kwargs[key] = self._repo_root / kwargs[key]
        return kwargs

    def add_upload(self, name: str, content: bytes, date_column: str | None) -> DatasetEntry:
        self._uploads_dir.mkdir(parents=True, exist_ok=True)
        dataset_id = f"upload_{uuid.uuid4().hex[:10]}"
        (self._uploads_dir / f"{dataset_id}.csv").write_bytes(content)
        meta = {"name": name, "date_column": date_column}
        (self._uploads_dir / f"{dataset_id}.json").write_text(json.dumps(meta), encoding="utf-8")
        return self.get(dataset_id)

    def remove_upload(self, dataset_id: str) -> None:
        if dataset_id in self._builtin:
            raise NotFoundError("Datasets embutidos nao podem ser removidos.")
        self.get(dataset_id)
        for suffix in (".csv", ".json"):
            (self._uploads_dir / f"{dataset_id}{suffix}").unlink(missing_ok=True)

    def _list_uploads(self) -> list[DatasetEntry]:
        if not self._uploads_dir.is_dir():
            return []
        entries = [self._read_upload_entry(p.stem) for p in sorted(self._uploads_dir.glob("upload_*.json"))]
        return [entry for entry in entries if entry is not None]

    def _read_upload_entry(self, dataset_id: str) -> DatasetEntry | None:
        if not is_safe_id(dataset_id) or not dataset_id.startswith("upload_"):
            return None
        meta_path = self._uploads_dir / f"{dataset_id}.json"
        csv_path = self._uploads_dir / f"{dataset_id}.csv"
        if not meta_path.is_file() or not csv_path.is_file():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return DatasetEntry(
            id=dataset_id,
            name=str(meta.get("name") or dataset_id),
            description="CSV enviado pelo usuario. Sem ground truth.",
            origin="upload",
            loader_kwargs={
                "data_path": csv_path,
                "data_format": "csv",
                "date_column": meta.get("date_column"),
            },
        )


def validate_upload_csv(content: bytes, date_column: str | None) -> pd.DataFrame:
    """Le o CSV enviado e garante o minimo para a analise; levanta ValueError."""
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except Exception as error:  # noqa: BLE001 - qualquer falha de parse vira 400
        raise ValueError(f"CSV ilegivel: {error}") from error
    if date_column is not None and date_column not in frame.columns:
        raise ValueError(f"Coluna temporal '{date_column}' nao existe no CSV.")
    numeric = [c for c in frame.select_dtypes(include="number").columns if c != date_column]
    if len(numeric) < 2:
        raise ValueError("O CSV precisa de ao menos 2 colunas numericas.")
    if len(frame) < 30:
        raise ValueError("O CSV precisa de ao menos 30 linhas.")
    return frame
