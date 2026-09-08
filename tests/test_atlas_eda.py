from __future__ import annotations

import unittest
from pathlib import Path

from causal_algorithms_atlas.eda import (
    assumption_coverage_figure,
    cards_to_dataframe,
    family_counts_figure,
)
from causal_algorithms_atlas.loader import load_algorithm_cards

_ATLAS_ROOT = Path(__file__).resolve().parent.parent / "causal_algorithms_atlas"


class CardsToDataframeTests(unittest.TestCase):
    def test_one_row_per_algorithm(self):
        cards = load_algorithm_cards(_ATLAS_ROOT / "algorithms")
        df = cards_to_dataframe(cards)
        self.assertEqual(len(df), len(cards))
        self.assertIn("family", df.columns)
        self.assertIn("n_assumptions", df.columns)

    def test_pcmci_row_matches_card_fields(self):
        cards = load_algorithm_cards(_ATLAS_ROOT / "algorithms")
        df = cards_to_dataframe(cards)
        row = df[df["id"] == "pcmci"].iloc[0]
        self.assertEqual(row["family"], "constraint-based")
        self.assertFalse(row["handles_latent_confounders"])


class FigureBuildersTests(unittest.TestCase):
    def test_family_counts_figure_has_one_trace(self):
        cards = load_algorithm_cards(_ATLAS_ROOT / "algorithms")
        df = cards_to_dataframe(cards)
        figure = family_counts_figure(df)
        self.assertGreaterEqual(len(figure.data), 1)

    def test_assumption_coverage_figure_builds_without_error(self):
        cards = load_algorithm_cards(_ATLAS_ROOT / "algorithms")
        figure = assumption_coverage_figure(cards)
        self.assertGreaterEqual(len(figure.data), 1)


if __name__ == "__main__":
    unittest.main()
