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


def test_experiment_compares_soft_and_rigid_filters_sequentially(toy_bundle):
    from causal_algorithms_atlas import experiment_runner

    settings, bundle = toy_bundle
    atlas = AtlasService(settings.repo_root, Path(settings.data_dir) / "history.jsonl")
    order = []

    def fake_run(data, *, use_assumption_soft_filter, **_kwargs):
        order.append("soft" if use_assumption_soft_filter else "rigid")
        if not use_assumption_soft_filter:
            raise experiment_runner.InsufficientCandidatesError("so 1 metodo compativel")
        return {
            "candidate_methods": ["A", "B", "C"],
            "ranking": [{}, {}, {}, {}],
            "best_combination_methods": ["A", "B"],
            "best_combination_performance_score": 0.7,
            "best_single_method": "A",
            "best_combination_metrics_post_hoc": {"f1_score": 0.8},
            "best_single_metrics_post_hoc": {"f1_score": 0.6},
        }

    params = {"dataset_id": "toy_a", "max_lag": 1, "n_bootstrap": 2, "compare_filters": True}
    with mock.patch.object(experiment_runner, "run_experiment", side_effect=fake_run):
        result = atlas.run_experiment(bundle, params, lambda *_: None)

    assert order == ["soft", "rigid"]  # em sequencia, a escolhida primeiro
    assert result["outcome"] == "completed"  # resultado principal = filtro escolhido (suave)
    comparison = result["filter_comparison"]
    assert comparison["soft"]["combinations_evaluated"] == 4
    assert comparison["soft"]["f1_combination_post_hoc"] == 0.8
    assert comparison["rigid"]["outcome"] == "insufficient_candidates"
    assert comparison["rigid"]["message"] == "so 1 metodo compativel"
    assert comparison["excluded_by_rigid"] == ["A", "B", "C"]
    assert comparison["soft"]["elapsed_seconds"] >= 0 and comparison["rigid"]["elapsed_seconds"] >= 0

    order.clear()
    with mock.patch.object(experiment_runner, "run_experiment", side_effect=fake_run):
        single = atlas.run_experiment(bundle, {**params, "compare_filters": False}, lambda *_: None)
    assert order == ["soft"] and "filter_comparison" not in single and "elapsed_seconds" in single
