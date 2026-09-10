from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from causal_algorithms_atlas.export import Chunk, cards_to_chunks
from causal_algorithms_atlas.loader import load_algorithm_cards

_ATLAS_ROOT_ALGORITHMS = "causal_algorithms_atlas/algorithms"
_DEFAULT_MODEL = "qwen2.5:7b"
_DEFAULT_BASE_URL = "http://localhost:11434"
_TIMEOUT_SECONDS = 120


class RagChatError(RuntimeError):
    """Raised when the local Ollama server can't be reached or errors out."""


@dataclass
class TfidfRetriever:
    chunks: list[Chunk]
    vectorizer: TfidfVectorizer
    matrix: object

    def top_k(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:k]


def build_retriever(chunks: list[Chunk]) -> TfidfRetriever:
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
    return TfidfRetriever(chunks=chunks, vectorizer=vectorizer, matrix=matrix)


def build_prompt(query: str, retrieved: list[tuple[Chunk, float]]) -> str:
    context_blocks = "\n\n".join(
        f"[{chunk.algorithm_id} | {chunk.section}]\n{chunk.text}" for chunk, _ in retrieved
    )
    return (
        "Voce e um assistente que responde exclusivamente com base no contexto abaixo, "
        "extraido de uma base de conhecimento sobre algoritmos de causal discovery em "
        "series temporais. Se o contexto nao tiver a resposta, diga isso explicitamente "
        "em vez de inventar.\n\n"
        f"Contexto:\n{context_blocks}\n\n"
        f"Pergunta: {query}\n"
        "Resposta:"
    )


def call_ollama(
    prompt: str,
    *,
    model: str = _DEFAULT_MODEL,
    base_url: str = _DEFAULT_BASE_URL,
    format: str | dict | None = None,
) -> str:
    payload_dict: dict[str, object] = {"model": model, "prompt": prompt, "stream": False}
    if format is not None:
        payload_dict["format"] = format
    payload = json.dumps(payload_dict).encode("utf-8")
    request = Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (URLError, HTTPError, OSError) as exc:
        raise RagChatError(
            f"Nao foi possivel falar com o Ollama em {base_url} (modelo {model!r}). "
            f"Verifique se o servico esta rodando ('ollama serve') e se o modelo foi "
            f"baixado ('ollama pull {model}'). Erro original: {exc}"
        ) from exc
    return body.get("response", "")


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("Uso: python -m causal_algorithms_atlas.rag_chat \"sua pergunta\"")
        raise SystemExit(1)
    query = argv[0]

    cards = load_algorithm_cards(_ATLAS_ROOT_ALGORITHMS)
    chunks = cards_to_chunks(cards)
    retriever = build_retriever(chunks)
    retrieved = retriever.top_k(query, k=4)

    print("--- chunks recuperados ---")
    for chunk, score in retrieved:
        print(f"[{score:.3f}] {chunk.algorithm_id} / {chunk.section}")
    print()

    prompt = build_prompt(query, retrieved)
    answer = call_ollama(prompt)
    print("--- resposta ---")
    print(answer)


if __name__ == "__main__":
    main()
