"""Tipos de execucao alem do pipeline, atlas e progresso/cancelamento (com execucao fake)."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.domain import RunKind, RunStatus
from backend.app.main import create_app
from backend.app.services import analysis
from backend.app.services.jobs import InlineJobRunner
from backend.app.services.runs import KindSpec

from .conftest import make_settings

API = "/api/v1"


@pytest.fixture
def seen():
    return []


@pytest.fixture
def app(tmp_path, seen):
    def fake(kind):
        def execute(params, progress):
            seen.append((kind, params))
            progress(1, 2, "meio")
            progress(2, 2, "fim")
            return {"kind": kind}

        return execute

    heavy = {
        kind.value: KindSpec(lambda raw: dict(raw), fake(kind.value))
        for kind in (RunKind.BENCHMARK, RunKind.REPLICATED_VALIDATION, RunKind.ATLAS_EXPERIMENT, RunKind.ATLAS_CHAT)
    }
    # Mantem a validacao real de cada tipo; troca so a execucao pesada.
    application = create_app(
        make_settings(tmp_path),
        pipeline=lambda dataset, params: {"ok": True},
        jobs=InlineJobRunner(),
        method_weights={"PCMCI": 1.0, "GES": 1.0, "VARLiNGAM": 1.0},
    )
    runs = application.state.container.runs
    for name, spec in heavy.items():
        real = runs._kinds[name]
        runs._kinds[name] = KindSpec(real.validate, spec.execute)
    return application


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def test_benchmark_run_records_progress_and_kind(client, seen):
    run = client.post(f"{API}/runs/benchmark", json={"n_samples": 300, "index_change": 100}).json()
    assert run["kind"] == "benchmark"
    stored = client.get(f"{API}/runs/{run['id']}").json()
    assert stored["status"] == "succeeded"
    assert stored["progress"] == {"done": 2, "total": 2, "message": "fim"}
    assert seen[0][1]["index_change"] == 100


def test_benchmark_validation(client):
    assert client.post(f"{API}/runs/benchmark", json={"n_samples": 300, "index_change": 400}).status_code == 400
    assert client.post(f"{API}/runs/benchmark", json={"n_samples": 10}).status_code == 422
    assert client.post(f"{API}/runs/benchmark", json={"methods": ["GES"]}).status_code == 400


def test_replicated_validation_requires_replicas(client):
    body = {"dataset_id": "toy_a", "n_replicates": 3}
    response = client.post(f"{API}/runs/replicated-validation", json=body)
    assert response.status_code == 400
    assert "replicas" in response.json()["detail"]

    ok = client.post(f"{API}/runs/replicated-validation", json={**body, "dataset_id": "causaltime_traffic"})
    assert ok.status_code == 202 and ok.json()["kind"] == "replicated_validation"


def test_atlas_experiment_and_chat_validate_dataset(client):
    assert client.post(f"{API}/runs/atlas-experiment", json={"dataset_id": "nada"}).status_code == 404
    assert client.post(f"{API}/runs/atlas-experiment", json={"dataset_id": "toy_a", "n_bootstrap": 0}).status_code == 422
    experiment = client.post(f"{API}/runs/atlas-experiment", json={"dataset_id": "toy_a", "max_rows": 200})
    assert experiment.status_code == 202 and experiment.json()["kind"] == "atlas_experiment"
    chat = client.post(f"{API}/runs/atlas-chat", json={"dataset_id": "toy_a"})
    assert chat.status_code == 202 and chat.json()["kind"] == "atlas_chat"


def test_pipeline_params_columns_trajectory_and_decomposition(client):
    body = {
        "dataset_id": "causaltime_traffic",
        "columns": ["traffic_00", "traffic_01", "traffic_02"],
        "trajectory_index": 5,
        "decomposition_period": 30,
        "objective": {"type": "causes_of_target", "primary_variable": "traffic_00"},
        "selected_relations": [["traffic_01", "traffic_00"], ["traffic_02", "traffic_00"]],
    }
    run = client.post(f"{API}/runs", json=body)
    assert run.status_code == 202
    params = run.json()["params"]
    assert params["trajectory_index"] == 5 and params["decomposition_period"] == 30
    assert params["panel_max_lag"] == 1 and params["objective"]["type"] == "causes_of_target"

    assert client.post(f"{API}/runs", json={**body, "trajectory_index": 9999}).status_code == 400
    assert client.post(f"{API}/runs", json={"dataset_id": "toy_a", "trajectory_index": 1}).status_code == 400
    assert client.post(f"{API}/runs", json={"dataset_id": "toy_a", "selected_relations": []}).status_code == 400


def test_dataset_details_expose_run_options(client):
    traffic = client.get(f"{API}/datasets/causaltime_traffic").json()
    assert traffic["trajectory_count"] > 1 and traffic["supports_replicates"] is True
    assert len(traffic["available_columns"]) > len(traffic["selected_columns"])
    delhi = client.get(f"{API}/datasets/delhi_csv").json()
    assert delhi["decomposition_period"] == 30 and delhi["supports_replicates"] is False
    assert client.get(f"{API}/datasets/toy_a").json()["trajectory_count"] == 1


def test_cancellation_is_cooperative(app):
    service = app.state.container.runs
    state = {}

    def execute(params, progress):
        progress(0, 3, "primeira")
        service.delete(state["id"])  # cancelamento chega durante a execucao
        progress(1, 3, "segunda")  # deve abortar aqui
        state["reached_end"] = True
        return {}

    service._kinds["benchmark"] = KindSpec(lambda raw: dict(raw), execute)

    class CaptureRunner(InlineJobRunner):
        def submit(self, fn):
            state["id"] = service.list()[0].id
            fn()

    service._jobs = CaptureRunner()
    run = service.create("benchmark", {})
    assert "reached_end" not in state
    assert service.get(run.id).status is RunStatus.CANCELLED
    assert service.get(run.id).result is None


def test_atlas_algorithms_and_rag_retrieval(client):
    algorithms = client.get(f"{API}/atlas/algorithms").json()
    assert algorithms and {"id", "name", "assumptions", "references"} <= set(algorithms[0])
    detail = client.get(f"{API}/atlas/algorithms/{algorithms[0]['id']}").json()
    assert "sections" in detail
    assert client.get(f"{API}/atlas/algorithms/nao-existe").status_code == 404

    asked = client.post(f"{API}/atlas/ask", json={"query": "quais metodos assumem estacionariedade?", "k": 3}).json()
    assert len(asked["retrieved"]) == 3 and asked["answer"] is None and asked["error"] is None
    assert asked["retrieved"][0]["score"] >= asked["retrieved"][-1]["score"]
    assert client.post(f"{API}/atlas/ask", json={"query": "x"}).status_code == 422


def _summary(rows):
    return pd.DataFrame(rows)


def test_compare_strategies_without_ground_truth_reports_counts_and_overlap():
    method_output = pd.DataFrame(
        [{"source": "A", "target": "B", "lag": 1, "score": 0.5, "p_value": 0.01},
         {"source": "A", "target": "A", "lag": 1, "score": 0.9, "p_value": 0.01}]
    )
    selection = {"precomputed_outputs": {"PCMCI": method_output}}
    summary = _summary(
        [{"source": "A", "target": "B", "lag": 1, "edge_probability": 0.9, "ensemble_selected": True, "ensemble_score": 0.9},
         {"source": "B", "target": "C", "lag": 1, "edge_probability": 0.2, "ensemble_selected": False, "ensemble_score": 0.1}]
    )
    result = analysis.compare_strategies(
        selection, summary, ground_truth=pd.DataFrame(columns=["source", "target", "lag"]),
        nodes=["A", "B", "C"], relations=[("A", "B"), ("B", "A"), ("B", "C"), ("C", "B"), ("A", "C"), ("C", "A")],
        threshold=0.5,
    )
    rows = {r["strategy"]: r for r in result["rows"]}
    assert rows["PCMCI"]["returned_edges"] == 1  # autoaresta descartada
    assert rows["ENSEMBLE_AUTO"]["detected_pairs"] == 1 and "f1_score" not in rows["ENSEMBLE_AUTO"]
    assert result["overlap"][0]["shared_pairs"] == 1 and result["overlap"][0]["jaccard"] == 1.0
    assert result["reference"] is None and result["evaluated_pairs"] == 3


def test_compare_strategies_with_ground_truth_scores_each_strategy():
    truth = pd.DataFrame([{"source": "A", "target": "B", "lag": 1}])
    method_output = pd.DataFrame(
        [{"source": "A", "target": "B", "lag": 1, "score": 0.8, "p_value": 0.01},
         {"source": "B", "target": "C", "lag": 1, "score": 0.1, "p_value": 0.4}]
    )
    summary = _summary(
        [{"source": "A", "target": "B", "lag": 1, "edge_probability": 0.9, "ensemble_selected": True, "ensemble_score": 0.9},
         {"source": "B", "target": "C", "lag": 1, "edge_probability": 0.3, "ensemble_selected": False, "ensemble_score": 0.2}]
    )
    relations = [(s, t) for s in "ABC" for t in "ABC" if s != t]
    result = analysis.compare_strategies(
        {"precomputed_outputs": {"PCMCI": method_output}}, summary, ground_truth=truth,
        nodes=["A", "B", "C"], relations=relations, threshold=0.5,
    )
    rows = {r["strategy"]: r for r in result["rows"]}
    assert rows["ENSEMBLE_AUTO"]["precision"] == 1.0 and rows["ENSEMBLE_AUTO"]["recall"] == 1.0
    assert rows["PCMCI"]["precision"] == 0.5 and rows["PCMCI"]["false_positive_pairs"] == [("B", "C")]
    assert result["reference"]["ground_truth_pairs"] == 1
    assert result["rows"][0]["strategy"] == "ENSEMBLE_AUTO"  # ordenado por F1
