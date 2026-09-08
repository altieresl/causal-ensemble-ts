from __future__ import annotations

import unittest
from unittest.mock import patch

from causal_algorithms_atlas.export import cards_to_chunks
from causal_algorithms_atlas.loader import load_algorithm_cards
from causal_algorithms_atlas.rag_chat import (
    RagChatError,
    build_prompt,
    build_retriever,
    call_ollama,
)
from pathlib import Path

_ATLAS_ROOT = Path(__file__).resolve().parent.parent / "causal_algorithms_atlas"


class TfidfRetrieverTests(unittest.TestCase):
    def setUp(self):
        cards = load_algorithm_cards(_ATLAS_ROOT / "algorithms")
        self.chunks = cards_to_chunks(cards)
        self.retriever = build_retriever(self.chunks)

    def test_top_k_returns_requested_count(self):
        results = self.retriever.top_k("confundidor latente", k=3)
        self.assertEqual(len(results), 3)

    def test_top_k_is_sorted_descending_by_score(self):
        results = self.retriever.top_k("relacoes nao lineares", k=5)
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_query_about_latent_confounders_surfaces_lpcmci_or_fci(self):
        results = self.retriever.top_k("qual metodo tolera confundidor latente nao observado", k=5)
        surfaced_ids = {chunk.algorithm_id for chunk, _ in results}
        self.assertTrue(surfaced_ids & {"lpcmci", "fci"})


class BuildPromptTests(unittest.TestCase):
    def test_prompt_includes_query_and_retrieved_text(self):
        cards = load_algorithm_cards(_ATLAS_ROOT / "algorithms")
        chunks = cards_to_chunks(cards)
        retriever = build_retriever(chunks)
        retrieved = retriever.top_k("PCMCI", k=2)
        prompt = build_prompt("O que e PCMCI?", retrieved)
        self.assertIn("O que e PCMCI?", prompt)
        for chunk, _ in retrieved:
            self.assertIn(chunk.text, prompt)


class CallOllamaTests(unittest.TestCase):
    def test_raises_rag_chat_error_when_ollama_unreachable(self):
        with patch(
            "causal_algorithms_atlas.rag_chat.urlopen",
            side_effect=OSError("connection refused"),
        ):
            with self.assertRaises(RagChatError):
                call_ollama("prompt de teste", base_url="http://localhost:1")


if __name__ == "__main__":
    unittest.main()
