"""Assistente conversacional da interface: RAG nas fichas do atlas + contexto da tela + Ollama.

Sem estado no servidor: o histórico vem do cliente a cada pergunta. O contexto enviado ao modelo
NUNCA inclui nada derivado do grafo verdadeiro (validação, comparação pós-hoc, evidência de painel,
benchmarks): o assistente pode ser perguntado "quais métodos usar?", e responder com base no gabarito
seria vazamento do gabarito para uma recomendação (ver ``no-ground-truth-leakage``).
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..domain import DomainError, NotFoundError
from .datasets import DatasetService
from .retrieval import retrieve
from .runs import RunService

MAX_MESSAGES = 20
MAX_CHARS = 4000
DEFAULT_MODEL = os.environ.get("CAUSAL_OLLAMA_MODEL", "qwen2.5:7b")
DEFAULT_URL = os.environ.get("CAUSAL_OLLAMA_URL", "http://localhost:11434")
TIMEOUT_SECONDS = 180

SYSTEM_PROMPT = """Voce e o assistente da ferramenta "Causal Discovery TS", que descobre relacoes causais em series temporais combinando varios algoritmos (ensemble).
Responda em portugues do Brasil, de forma clara e objetiva.
Use as informacoes do CONTEXTO DA TELA e dos TRECHOS DO ATLAS abaixo. Se algo nao estiver neles, diga que nao sabe em vez de inventar; nunca invente numeros.
Use linguagem cautelosa: dados observacionais sugerem, mas nao provam, causalidade.
Voce NAO tem acesso ao grafo verdadeiro (gabarito) e nao deve recomendar metodos com base em acertos medidos contra ele."""


def _clip(text: str, limit: int = 1200) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


class AssistantService:
    def __init__(self, datasets: DatasetService, runs: RunService, atlas_cards_loader, base_url: str = DEFAULT_URL) -> None:
        self._datasets = datasets
        self._runs = runs
        self._cards_loader = atlas_cards_loader
        self._base_url = base_url.rstrip("/")
        self._retriever = None
        self._card_cache: dict[str, Any] | None = None
        self._profiles: dict[str, str] = {}
        self._lock = threading.Lock()

    # -- recuperação e contexto ------------------------------------------
    def _cards(self) -> dict[str, Any]:
        with self._lock:
            if self._card_cache is None:
                self._card_cache = self._cards_loader()
            return self._card_cache

    def _retrieve(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        from causal_algorithms_atlas import rag_chat
        from causal_algorithms_atlas.export import cards_to_chunks

        cards = self._cards()
        with self._lock:
            if self._retriever is None:
                self._retriever = rag_chat.build_retriever(cards_to_chunks(cards))
        return [
            {"algorithm_id": c.algorithm_id, "section": c.section, "text": c.text, "score": float(score)}
            for c, score in retrieve(self._retriever, cards, query, k=k)
            if score > 0
        ]

    def _dataset_context(self, dataset_id: str) -> str:
        if dataset_id in self._profiles:
            return self._profiles[dataset_id]
        from causal_algorithms_atlas.dataset_profile import profile_dataset
        from causal_algorithms_atlas.ensemble_advisor import recommend_framework_methods

        entry = self._datasets.get(dataset_id)
        bundle = self._datasets.load(dataset_id)
        profile = profile_dataset(bundle.data)
        recommendations = recommend_framework_methods(profile)
        lines = [
            f"O usuario esta vendo o dataset '{entry.name}' ({entry.description}).",
            f"{len(bundle.data)} observacoes; variaveis: {', '.join(bundle.selected_columns)}.",
            profile.to_query_text().rsplit(" Quais algoritmos", 1)[0],
            "Recomendacao por premissas (fichas do atlas):",
            *[
                f"- {r.framework_method_name}: {'incluir' if r.included else 'premissa violada'} ({_clip(' '.join(r.reasons), 300)})"
                for r in recommendations
            ],
        ]
        text = "\n".join(lines)
        self._profiles[dataset_id] = text
        return text

    def _run_context(self, run_id: str) -> str:
        run = self._runs.get(run_id)
        params = {k: v for k, v in run.params.items() if k not in {"expert_knowledge", "selected_relations", "objective"}}
        lines = [
            f"O usuario esta vendo a execucao {run.id} do tipo '{run.kind}', status '{run.status.value}'.",
            f"Parametros: {json.dumps(params, ensure_ascii=False, default=str)[:600]}",
        ]
        if run.error:
            lines.append(f"Erro da execucao: {_clip(run.error, 400)}")
        result = run.result or {}
        if run.kind == "pipeline" and result:
            selected = [e for e in result.get("edges", []) if e.get("ensemble_selected")]
            selected.sort(key=lambda e: e.get("edge_probability") or 0, reverse=True)
            lines.append(f"Melhor combinacao de metodos (ENSEMBLE_AUTO): {' + '.join(result.get('best_combination', []))}.")
            lines.append(f"Pesos dos metodos: {json.dumps(result.get('method_weights', {}), default=str)[:300]}")
            lines.append(f"{len(selected)} arestas selecionadas de {len(result.get('edges', []))} candidatas. Principais:")
            lines += [
                f"- {e['source']} -> {e['target']} (lag {e.get('lag')}, probabilidade {round(e.get('edge_probability') or 0, 2)})"
                for e in selected[:15]
            ]
            lines.append("(Validacao contra o grafo verdadeiro existe na tela, mas foi omitida aqui de proposito.)")
        elif run.kind == "atlas_chat" and result:
            decisions = result.get("chat", {}).get("decisions", [])
            lines.append("Votos do chat por algoritmo (e se concordam com o filtro estatistico):")
            lines += [
                f"- {d['name']}: {'incluir' if d['include'] else 'excluir'}; concorda={d.get('agrees')}; motivo: {_clip(d.get('reason', ''), 200)}"
                for d in decisions
            ]
        elif run.kind in {"benchmark", "replicated_validation", "atlas_experiment"}:
            lines.append(
                "Os numeros deste resultado sao avaliacoes contra o grafo verdadeiro; por isso nao foram enviados ao assistente."
            )
        return "\n".join(lines)

    def build_context(self, context: dict[str, Any] | None) -> str:
        if not context:
            return "O usuario nao esta vendo nenhum dataset ou execucao especifico."
        kind, ident = context.get("kind"), str(context.get("id") or "")
        try:
            if kind == "dataset":
                return self._dataset_context(ident)
            if kind == "run":
                return self._run_context(ident)
        except NotFoundError:
            return f"O {kind} '{ident}' nao foi encontrado."
        except DomainError as error:
            return f"Nao foi possivel carregar o contexto: {error}"
        return "Contexto desconhecido."

    def status(self) -> dict[str, Any]:
        """Ollama esta no ar? Quais modelos estao baixados? (consulta rapida, para a interface)."""
        try:
            with urlopen(f"{self._base_url}/api/tags", timeout=2) as response:
                models = [m.get("name") for m in json.loads(response.read().decode("utf-8")).get("models", [])]
            return {"available": True, "models": models, "default_model": DEFAULT_MODEL, "base_url": self._base_url}
        except (URLError, HTTPError, OSError, ValueError) as error:
            return {
                "available": False,
                "models": [],
                "default_model": DEFAULT_MODEL,
                "base_url": self._base_url,
                "error": str(error),
            }

    # -- conversa ---------------------------------------------------------
    @staticmethod
    def validate_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if not messages or messages[-1].get("role") != "user":
            raise DomainError("A conversa deve terminar com uma pergunta do usuario.")
        cleaned = [
            {"role": m["role"], "content": str(m["content"])[:MAX_CHARS]}
            for m in messages
            if m.get("role") in {"user", "assistant"} and str(m.get("content", "")).strip()
        ]
        return cleaned[-MAX_MESSAGES:]

    def _prompt_messages(self, messages, context_text: str, sources: list[dict[str, Any]]) -> list[dict[str, str]]:
        excerpts = "\n\n".join(f"[{s['algorithm_id']} | {s['section']}]\n{_clip(s['text'], 900)}" for s in sources)
        system = (
            f"{SYSTEM_PROMPT}\n\nCONTEXTO DA TELA:\n{context_text}\n\n"
            f"TRECHOS DO ATLAS (recuperados para a ultima pergunta):\n{excerpts or '(nenhum trecho relevante)'}"
        )
        return [{"role": "system", "content": system}, *messages]

    def stream(self, messages: list[dict[str, str]], context: dict[str, Any] | None, model: str | None) -> Iterator[str]:
        """Eventos NDJSON: ``sources`` → ``token``* → ``done`` (ou ``error``). Nunca levanta no meio do stream."""
        messages = self.validate_messages(messages)
        sources = self._retrieve(messages[-1]["content"])
        yield json.dumps({"type": "sources", "sources": sources}, ensure_ascii=False) + "\n"
        try:
            context_text = self.build_context(context)
        except Exception as error:  # noqa: BLE001 - contexto e opcional; a conversa segue sem ele
            context_text = f"(contexto indisponivel: {error})"
        payload = json.dumps(
            {"model": model or DEFAULT_MODEL, "messages": self._prompt_messages(messages, context_text, sources), "stream": True}
        ).encode("utf-8")
        request = Request(
            f"{self._base_url}/api/chat", data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise RuntimeError(chunk["error"])
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield json.dumps({"type": "token", "content": token}, ensure_ascii=False) + "\n"
                    if chunk.get("done"):
                        break
        except (URLError, HTTPError, OSError, RuntimeError, ValueError) as error:
            yield json.dumps(
                {
                    "type": "error",
                    "message": (
                        f"Nao foi possivel falar com o Ollama em {self._base_url} (modelo {model or DEFAULT_MODEL!r}). "
                        "Verifique se 'ollama serve' esta rodando e se o modelo foi baixado. "
                        f"Detalhe: {error}"
                    ),
                },
                ensure_ascii=False,
            ) + "\n"
            return
        yield json.dumps({"type": "done"}) + "\n"
