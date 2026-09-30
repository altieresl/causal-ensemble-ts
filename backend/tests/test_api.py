API = "/api/v1"


def test_health_and_methods(client):
    assert client.get(f"{API}/health").json() == {"status": "ok"}
    methods = client.get(f"{API}/methods").json()
    assert {m["name"] for m in methods} == {"pcmci", "ges", "granger"}


def test_dataset_catalog_and_details(client):
    ids = {d["id"] for d in client.get(f"{API}/datasets").json()}
    assert {"delhi_csv", "causaltime_traffic", "toy_a"} <= ids

    details = client.get(f"{API}/datasets/toy_a").json()
    assert details["has_ground_truth"] is True
    assert set(details["available_columns"]) >= {"Y", "X1"}
    assert len(details["preview"]) == 8


def test_unknown_dataset_is_problem_json(client):
    response = client.get(f"{API}/datasets/nao_existe")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["status"] == 404


def test_path_traversal_ids_are_rejected(client):
    assert client.get(f"{API}/runs/..%2F..%2Fsegredo").status_code == 404
    assert client.get(f"{API}/datasets/..%2Fx").status_code == 404


def test_profile_does_not_need_ground_truth(client):
    response = client.post(f"{API}/datasets/toy_a/profile", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["n_variables"] == 5
    assert body["recommendations"]
    assert {"method", "included", "reasons"} <= set(body["recommendations"][0])


def test_upload_valid_and_invalid(client):
    rows = "\n".join(f"{i},{i * 2},{i % 7}" for i in range(40))
    good = client.post(f"{API}/datasets", files={"file": ("meu.csv", f"a,b,c\n{rows}", "text/csv")})
    assert good.status_code == 201
    dataset_id = good.json()["id"]
    assert good.json()["origin"] == "upload"
    assert client.get(f"{API}/datasets/{dataset_id}").json()["n_rows"] == 40
    assert client.delete(f"{API}/datasets/{dataset_id}").status_code == 204
    assert client.get(f"{API}/datasets/{dataset_id}").status_code == 404

    short = client.post(f"{API}/datasets", files={"file": ("x.csv", "a,b\n1,2\n3,4", "text/csv")})
    assert short.status_code == 400

    huge = client.post(f"{API}/datasets", files={"file": ("x.csv", "a,b\n" + "1,2\n" * 40000, "text/csv")})
    assert huge.status_code == 400


def test_builtin_cannot_be_deleted(client):
    assert client.delete(f"{API}/datasets/toy_a").status_code == 404


def test_run_lifecycle_success(client, calls):
    response = client.post(f"{API}/runs", json={"dataset_id": "toy_a", "methods": ["pcmci", "ges"]})
    assert response.status_code == 202
    run = response.json()
    assert response.headers["location"] == f"{API}/runs/{run['id']}"

    status = client.get(f"{API}/runs/{run['id']}").json()
    assert status["status"] == "succeeded"
    assert client.get(f"{API}/runs/{run['id']}/result").json()["edges"][0]["source"] == "X1"
    assert calls[0].methods == ["pcmci", "ges"]
    assert [r["id"] for r in client.get(f"{API}/runs").json()] == [run["id"]]

    assert client.delete(f"{API}/runs/{run['id']}").status_code == 204
    assert client.get(f"{API}/runs/{run['id']}").status_code == 404


def test_run_failure_is_reported_and_result_conflicts(client):
    run = client.post(f"{API}/runs", json={"dataset_id": "toy_a", "max_lag": 19}).json()
    status = client.get(f"{API}/runs/{run['id']}").json()
    assert status["status"] == "failed"
    assert "falha simulada" in status["error"]
    assert client.get(f"{API}/runs/{run['id']}/result").status_code == 409


def test_run_validation_errors(client):
    def post(**body):
        return client.post(f"{API}/runs", json={"dataset_id": "toy_a", **body})

    assert post(columns=["Y", "inexistente"]).status_code == 400
    assert post(columns=["Y"]).status_code == 400
    assert post(methods=["pcmci"]).status_code == 400
    assert post(methods=["pcmci", "xyz"]).status_code == 400
    assert post(max_lag=0).status_code == 422
    assert post(expert_knowledge=[{"source": "Y", "target": "X1", "relation": "magica"}]).status_code == 422
    assert post(expert_knowledge=[{"source": "Y", "target": "ZZZ"}]).status_code == 400
    assert post(selected_relations=[["Y", "Y"]]).status_code == 400
    assert client.post(f"{API}/runs", json={"dataset_id": "nada"}).status_code == 404


def test_expert_rules_are_forwarded(client, calls):
    rule = {"source": "X1", "target": "Y", "relation": "strong", "confidence": 0.9, "constraint": "hard"}
    assert client.post(f"{API}/runs", json={"dataset_id": "toy_a", "expert_knowledge": [rule]}).status_code == 202
    assert calls[0].expert_knowledge[0]["relation"] == "strong"
    assert calls[0].expert_knowledge[0]["constraint"] == "hard"
