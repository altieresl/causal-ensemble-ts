from backend.app.adapters.run_store import FileRunRepository
from backend.app.domain import Run, RunParams, RunStatus
from backend.app.adapters.serialization import to_jsonable

import numpy as np
import pandas as pd


def test_repository_marks_interrupted_runs_failed(tmp_path):
    repo = FileRunRepository(tmp_path)
    repo.save(Run(id="run_a", params={"dataset_id": "toy_a"}, status=RunStatus.RUNNING))
    repo.save(Run(id="run_b", params={"dataset_id": "toy_a"}, status=RunStatus.SUCCEEDED))

    reopened = FileRunRepository(tmp_path)
    assert reopened.get("run_a").status is RunStatus.FAILED
    assert reopened.get("run_b").status is RunStatus.SUCCEEDED


def test_params_roundtrip_keeps_relation_tuples(tmp_path):
    repo = FileRunRepository(tmp_path)
    params = RunParams(dataset_id="toy_a", selected_relations=[("X1", "Y")])
    repo.save(Run(id="run_c", params=params.to_dict()))
    stored = RunParams.from_dict(repo.get("run_c").params)
    assert stored.selected_relations == [("X1", "Y")]


def test_legacy_run_files_without_kind_load_as_pipeline(tmp_path):
    (tmp_path / "run_old.json").write_text(
        '{"id": "run_old", "params": {"dataset_id": "toy_a"}, "status": "succeeded", '
        '"created_at": "2026-01-01T00:00:00+00:00", "started_at": null, "finished_at": null, '
        '"error": null, "result": {}}',
        encoding="utf-8",
    )
    run = FileRunRepository(tmp_path).get("run_old")
    assert run.kind == "pipeline" and run.progress is None


def test_to_jsonable_handles_nan_inf_numpy_and_frames():
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [np.int64(2), np.int64(3)]})
    payload = to_jsonable({"f": frame, "x": np.float64("inf"), "t": ("a", np.bool_(True)), "n": None})
    assert payload == {"f": [{"a": 1.0, "b": 2}, {"a": None, "b": 3}], "x": None, "t": ["a", True], "n": None}
