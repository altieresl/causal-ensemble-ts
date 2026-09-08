from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from causal_algorithms_atlas.export import (
    SUMMARY_SECTION_NAME,
    cards_to_chunks,
    write_chunks_jsonl,
)
from causal_algorithms_atlas.loader import load_algorithm_cards

_ATLAS_ROOT = Path(__file__).resolve().parent.parent / "causal_algorithms_atlas"


class CardsToChunksTests(unittest.TestCase):
    def test_one_chunk_per_prose_section_plus_summary(self):
        cards = load_algorithm_cards(_ATLAS_ROOT / "algorithms")
        pcmci = cards["pcmci"]
        chunks = cards_to_chunks({"pcmci": pcmci})

        sections = {chunk.section for chunk in chunks}
        self.assertIn(SUMMARY_SECTION_NAME, sections)
        for name in pcmci.sections:
            self.assertIn(name, sections)
        self.assertEqual(len(chunks), len(pcmci.sections) + 1)

    def test_summary_chunk_mentions_key_attributes(self):
        cards = load_algorithm_cards(_ATLAS_ROOT / "algorithms")
        lpcmci = cards["lpcmci"]
        chunks = cards_to_chunks({"lpcmci": lpcmci})
        summary = next(c for c in chunks if c.section == SUMMARY_SECTION_NAME)

        self.assertIn("confundidor", summary.text.lower())
        self.assertEqual(summary.metadata["family"], "constraint-based")
        self.assertTrue(summary.metadata["handles_latent_confounders"])

    def test_chunk_ids_are_unique_and_namespaced(self):
        cards = load_algorithm_cards(_ATLAS_ROOT / "algorithms")
        chunks = cards_to_chunks(cards)
        ids = [chunk.id for chunk in chunks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(chunk.id.startswith(f"{chunk.algorithm_id}#") for chunk in chunks))


class WriteChunksJsonlTests(unittest.TestCase):
    def test_writes_one_json_object_per_line(self):
        cards = load_algorithm_cards(_ATLAS_ROOT / "algorithms")
        chunks = cards_to_chunks(cards)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "chunks.jsonl"
            write_chunks_jsonl(chunks, out_path)
            lines = out_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), len(chunks))
        first = json.loads(lines[0])
        self.assertIn("id", first)
        self.assertIn("text", first)
        self.assertIn("metadata", first)


if __name__ == "__main__":
    unittest.main()
