import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.jobs import InlineJobRunner

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: executa o pipeline real do nucleo")


FAKE_RESULT = {"edges": [{"source": "X1", "target": "Y", "edge_probability": 0.9}], "validation": None}


def make_settings(tmp_path):
    base = Settings.from_env()
    return Settings(
        repo_root=base.repo_root,
        data_dir=tmp_path / "var",
        cors_origins=("http://localhost:5173",),
        max_upload_bytes=64 * 1024,
        max_workers=1,
        frontend_dist=None,
    )


@pytest.fixture
def calls():
    return []


@pytest.fixture
def client(tmp_path, calls):
    def fake_pipeline(dataset, params):
        calls.append(params)
        if params.max_lag == 19:
            raise RuntimeError("falha simulada")
        return dict(FAKE_RESULT)

    app = create_app(
        make_settings(tmp_path),
        pipeline=fake_pipeline,
        jobs=InlineJobRunner(),
        method_weights={"pcmci": 1.0, "ges": 0.8, "granger": 0.6},
    )
    with TestClient(app) as test_client:
        yield test_client
