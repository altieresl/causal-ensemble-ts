"""Assistente da interface: streaming, contexto da tela e ausencia de gabarito no prompt."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.jobs import InlineJobRunner

from .conftest import make_settings

API = "/api/v1"


class FakeOllama(BaseHTTPRequestHandler):
    received: list[dict] = []

    def log_message(self, *_args):  # silencia o servidor de teste
        return

    def do_GET(self):
        body = json.dumps({"models": [{"name": "qwen2.5:7b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.received.append(payload)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for token in ["Olá", ", ", "PCMCI", " assume estacionariedade."]:
            self.wfile.write((json.dumps({"message": {"content": token}, "done": False}) + "\n").encode())
            self.wfile.flush()
        self.wfile.write((json.dumps({"message": {"content": ""}, "done": True}) + "\n").encode())


@pytest.fixture
def ollama():
    FakeOllama.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _client(tmp_path, ollama_url, pipeline=None):
    base = make_settings(tmp_path)
    settings = Settings(**{**base.__dict__, "ollama_url": ollama_url})
    app = create_app(
        settings,
        pipeline=pipeline or (lambda dataset, params: {}),
        jobs=InlineJobRunner(),
        method_weights={"PCMCI": 1.0, "GES": 1.0},
    )
    return TestClient(app)


def _events(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def test_streams_tokens_with_sources_and_dataset_context(tmp_path, ollama):
    with _client(tmp_path, ollama) as client:
        response = client.post(
            f"{API}/assistant/chat",
            json={
                "messages": [{"role": "user", "content": "Quais metodos assumem estacionariedade?"}],
                "context": {"kind": "dataset", "id": "toy_a"},
            },
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = _events(response)
    assert events[0]["type"] == "sources" and events[0]["sources"]
    assert "".join(e["content"] for e in events if e["type"] == "token") == "Olá, PCMCI assume estacionariedade."
    assert events[-1]["type"] == "done"

    system = FakeOllama.received[0]["messages"][0]
    assert system["role"] == "system"
    assert "Sintetico A - linear" in system["content"] and "TRECHOS DO ATLAS" in system["content"]
    assert FakeOllama.received[0]["stream"] is True
    assert FakeOllama.received[0]["messages"][-1]["content"].startswith("Quais metodos")


def test_run_context_never_contains_ground_truth_metrics(tmp_path, ollama):
    result = {
        "best_combination": ["PCMCI", "GES"],
        "method_weights": {"PCMCI": 1.0, "GES": 1.0},
        "edges": [{"source": "X1", "target": "Y", "lag": 1, "edge_probability": 0.97, "ensemble_selected": True}],
        "validation": {"f1_score": 0.123456, "true_positive_pairs": [["X1", "Y"]]},
        "comparison": {"rows": [{"strategy": "PCMCI", "f1_score": 0.654321}]},
        "panel_evidence": {"ranking_metrics": {"roc_auc": 0.777777}},
    }
    with _client(tmp_path, ollama, pipeline=lambda dataset, params: result) as client:
        run = client.post(f"{API}/runs", json={"dataset_id": "toy_a"}).json()
        client.post(
            f"{API}/assistant/chat",
            json={"messages": [{"role": "user", "content": "Explique o resultado"}], "context": {"kind": "run", "id": run["id"]}},
        )
    system = FakeOllama.received[0]["messages"][0]["content"]
    assert "X1 -> Y" in system and "PCMCI + GES" in system
    for leaked in ("0.123456", "0.654321", "0.777777", "true_positive_pairs", "f1_score"):
        assert leaked not in system


def test_keeps_history_and_validates_input(tmp_path, ollama):
    with _client(tmp_path, ollama) as client:
        history = [
            {"role": "user", "content": "O que e PCMCI?"},
            {"role": "assistant", "content": "Um metodo baseado em restricoes."},
            {"role": "user", "content": "E ele lida com confundidores?"},
        ]
        assert client.post(f"{API}/assistant/chat", json={"messages": history}).status_code == 200
        sent = FakeOllama.received[0]["messages"]
        assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]

        ends_with_assistant = history[:2]
        assert client.post(f"{API}/assistant/chat", json={"messages": ends_with_assistant}).status_code == 400
        assert client.post(f"{API}/assistant/chat", json={"messages": []}).status_code == 422
        too_long = [{"role": "user", "content": "x" * 5000}]
        assert client.post(f"{API}/assistant/chat", json={"messages": too_long}).status_code == 422


def test_unknown_context_does_not_break_the_conversation(tmp_path, ollama):
    with _client(tmp_path, ollama) as client:
        response = client.post(
            f"{API}/assistant/chat",
            json={"messages": [{"role": "user", "content": "oi"}], "context": {"kind": "run", "id": "run_naoexiste"}},
        )
    assert _events(response)[-1]["type"] == "done"
    assert "nao foi encontrado" in FakeOllama.received[0]["messages"][0]["content"]


def test_status_and_offline_ollama_returns_error_event(tmp_path, ollama):
    with _client(tmp_path, ollama) as client:
        status = client.get(f"{API}/assistant/status").json()
    assert status["available"] is True and status["models"] == ["qwen2.5:7b"]

    with _client(tmp_path, "http://127.0.0.1:1") as client:
        assert client.get(f"{API}/assistant/status").json()["available"] is False
        events = _events(client.post(f"{API}/assistant/chat", json={"messages": [{"role": "user", "content": "oi"}]}))
    assert events[0]["type"] == "sources"
    assert events[-1]["type"] == "error" and "ollama serve" in events[-1]["message"]
