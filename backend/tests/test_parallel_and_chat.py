import json
from pathlib import Path
from unittest import mock

import pytest

from backend.app.adapters.catalog import DatasetCatalog
from backend.app.services.atlas import AtlasService
from backend.app.services.datasets import DatasetService
from backend.app.services.replicated import plan_parallelism

from .conftest import make_settings


def test_plan_parallelism_divides_cpu_budget_and_respects_overrides():
    with mock.patch("backend.app.services.replicated.os.cpu_count", return_value=16):
        workers, jobs = plan_parallelism({}, 10)
        assert workers == 3 and jobs == 4 and workers * jobs <= 16
        assert plan_parallelism({"parallel_replicas": 2}, 10) == (2, 4)
        assert plan_parallelism({"parallel_replicas": 6, "parallel_jobs": 2}, 4) == (4, 2)  # nunca mais que as replicas
    with mock.patch("backend.app.services.replicated.os.cpu_count", return_value=2):
        assert plan_parallelism({}, 5) == (1, 1)  # maquina pequena: sem oversubscription


@pytest.fixture
def toy_bundle(tmp_path):
    settings = make_settings(tmp_path)
    service = DatasetService(DatasetCatalog(settings.repo_root, settings.uploads_dir), 1024)
    return settings, service.load("toy_a")


def _fake_model(profile_by_prompt):
    """'Modelo' que obedece o perfil: percentuais reais e nenhuma premissa exigida."""

    def call(_prompt, **_kwargs):
        profile = profile_by_prompt
        return json.dumps(
            {
                "stationary_fraction_pct": round(profile.stationary_fraction * 100),
                "dataset_is_majority_stationary": profile.mostly_stationary,
                "linear_fraction_pct": round(profile.linear_fraction * 100),
                "dataset_is_majority_linear": profile.mostly_linear,
                "non_gaussian_fraction_pct": round(profile.non_gaussian_fraction * 100),
                "dataset_is_majority_non_gaussian": profile.mostly_non_gaussian,
                "algorithm_requires_stationarity": False,
                "algorithm_requires_linearity": False,
                "algorithm_requires_non_gaussian_errors": False,
                "include": True,
                "reason": "premissas do algoritmo satisfeitas pelo perfil",
            }
        )

    return call


def test_chat_result_explains_each_vote_and_compares_with_statistical_filter(toy_bundle):
    from causal_algorithms_atlas import rag_chat
    from causal_algorithms_atlas.dataset_profile import profile_dataset

    settings, bundle = toy_bundle
    profile = profile_dataset(bundle.data)
    atlas = AtlasService(settings.repo_root, Path(settings.data_dir) / "history.jsonl")
    calls = []

    with mock.patch.object(rag_chat, "call_ollama", side_effect=_fake_model(profile)) as fake:
        result = atlas.run_chat_recommendation(
            bundle, {"dataset_id": "toy_a", "parallel_calls": 4}, lambda d, t, m: calls.append((d, t, m))
        )
    assert fake.call_count >= 8  # uma chamada por algoritmo do framework

    decisions = result["chat"]["decisions"]
    assert decisions and all(
        {"name", "include", "reason", "statistical_included", "statistical_reasons", "agrees",
         "dataset_is_majority_stationary", "algorithm_requires_stationarity"} <= set(d)
        for d in decisions
    )
    assert "prompt" not in decisions[0] and "raw_response" not in decisions[0]
    assert all(d["agrees"] == (d["statistical_included"] == d["include"]) for d in decisions)
    summary = result["profile_summary"]
    assert summary["n_variables"] == profile.n_variables
    assert set(result["agreement"]) == {"only_statistical", "only_chat", "shared"}
    assert calls[-1][2] == "Concluido"
