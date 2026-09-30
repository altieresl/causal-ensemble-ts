"""Casos de uso de datasets: catalogo, upload, preview e perfil/recomendacao de metodos."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..adapters.catalog import DatasetCatalog, DatasetEntry, validate_upload_csv
from ..adapters.serialization import to_jsonable
from ..domain import DomainError


class DatasetService:
    def __init__(self, catalog: DatasetCatalog, max_upload_bytes: int) -> None:
        self._catalog = catalog
        self._max_upload_bytes = max_upload_bytes

    def list(self) -> list[DatasetEntry]:
        return self._catalog.list()

    def get(self, dataset_id: str) -> DatasetEntry:
        return self._catalog.get(dataset_id)

    def load(self, dataset_id: str) -> Any:
        """Carrega via nucleo (``load_time_series_dataset``)."""
        from causal_discovery import load_time_series_dataset

        entry = self._catalog.get(dataset_id)
        kwargs = self._catalog.resolve_kwargs(entry)
        data_path = kwargs.pop("data_path")
        try:
            return load_time_series_dataset(data_path, **kwargs)
        except (ValueError, FileNotFoundError) as error:
            raise DomainError(f"Falha ao carregar o dataset: {error}") from error

    def upload(self, name: str, content: bytes, date_column: str | None) -> DatasetEntry:
        if len(content) > self._max_upload_bytes:
            raise DomainError(f"Arquivo maior que o limite de {self._max_upload_bytes // (1024 * 1024)} MB.")
        try:
            validate_upload_csv(content, date_column)
        except ValueError as error:
            raise DomainError(str(error)) from error
        return self._catalog.add_upload(name, content, date_column)

    def delete_upload(self, dataset_id: str) -> None:
        self._catalog.remove_upload(dataset_id)

    def details(self, dataset_id: str, preview_rows: int = 8) -> dict[str, Any]:
        entry = self._catalog.get(dataset_id)
        bundle = self.load(dataset_id)
        head = bundle.data.head(preview_rows).reset_index()
        return to_jsonable(
            {
                "entry": entry,
                "available_columns": list(bundle.available_columns),
                "selected_columns": list(bundle.selected_columns),
                "n_rows": len(bundle.data),
                "preview": head,
                "has_ground_truth": not bundle.ground_truth.empty,
                "default_max_lag": entry.default_max_lag,
            }
        )

    def profile(
        self,
        dataset_id: str,
        columns: list[str] | None,
        declared_causal_sufficiency: bool | None,
    ) -> dict[str, Any]:
        """Perfil objetivo + recomendacao. Nunca consulta ground truth (ver skill de vazamento)."""
        from causal_algorithms_atlas.dataset_profile import profile_dataset
        from causal_algorithms_atlas.ensemble_advisor import recommend_framework_methods

        bundle = self.load(dataset_id)
        data = bundle.data if not columns else bundle.data.loc[:, columns]
        profile = profile_dataset(data)
        recommendations = recommend_framework_methods(
            profile, declared_causal_sufficiency=declared_causal_sufficiency
        )
        return to_jsonable(
            {
                "n_variables": profile.n_variables,
                "n_timepoints": profile.n_timepoints,
                "stationary_fraction": profile.stationary_fraction,
                "linear_fraction": profile.linear_fraction,
                "non_gaussian_fraction": profile.non_gaussian_fraction,
                "variables": [asdict(variable) for variable in profile.variables],
                "recommendations": [
                    {
                        "method": rec.framework_method_name,
                        "algorithm_id": rec.algorithm_id,
                        "included": rec.included,
                        "reasons": list(rec.reasons),
                    }
                    for rec in recommendations
                ],
            }
        )
