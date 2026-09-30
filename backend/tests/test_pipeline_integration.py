"""Roda o pipeline real (nucleo) em dataset toy pequeno, so com metodos rapidos."""

import json

import pytest

from backend.app.domain import RunParams
from backend.app.services.pipeline import run_pipeline

from .conftest import make_settings


@pytest.mark.slow
def test_real_pipeline_returns_serializable_result_with_post_hoc_validation(tmp_path):
    from backend.app.adapters.catalog import DatasetCatalog
    from backend.app.services.datasets import DatasetService

    settings = make_settings(tmp_path)
    service = DatasetService(DatasetCatalog(settings.repo_root, settings.uploads_dir), 1024)
    bundle = service.load("toy_a")
    bundle = type(bundle)(**{**bundle.__dict__, "data": bundle.data.iloc[:200].reset_index(drop=True)})

    params = RunParams(
        dataset_id="toy_a",
        max_lag=1,
        quick_mode=True,
        n_bootstrap=2,
        parallel_jobs=1,
        methods=["ClassicalGranger", "VARLiNGAM"],
    )
    result = run_pipeline(bundle, params)

    json.dumps(result)  # tudo serializavel (sem NaN/numpy)
    assert result["columns"] == list(bundle.selected_columns)
    assert result["edges"], "esperava ao menos uma aresta candidata"
    assert {"source", "target", "edge_probability"} <= set(result["edges"][0])
    assert result["validation"] is not None
    assert "f1_score" in result["validation"]
