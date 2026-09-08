from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from causal_discovery import discover_causal_methods
from causal_algorithms_atlas.loader import load_algorithm_cards
from causal_algorithms_atlas.schema import VerificationStatus
from causal_algorithms_atlas.validate import (
    validate_framework_alignment,
    validate_references,
)

_ATLAS_ROOT = Path(__file__).resolve().parent.parent / "causal_algorithms_atlas"


def _load_references() -> dict:
    with open(_ATLAS_ROOT / "references.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class FrameworkAlignmentTests(unittest.TestCase):
    def test_every_registered_method_has_a_verified_card(self):
        cards = load_algorithm_cards(_ATLAS_ROOT / "algorithms")
        registered = frozenset(discover_causal_methods())

        cards_by_method = {
            card.framework_method_name: card
            for card in cards.values()
            if card.framework_method_name
        }
        missing = registered - frozenset(cards_by_method)
        self.assertEqual(missing, frozenset(), f"Metodos sem ficha no atlas: {missing}")

        not_verified = [
            name
            for name, card in cards_by_method.items()
            if name in registered and card.verification is not VerificationStatus.VERIFIED
        ]
        self.assertEqual(not_verified, [], f"Fichas de metodos do framework nao verified: {not_verified}")

    def test_references_are_all_resolvable(self):
        cards = load_algorithm_cards(_ATLAS_ROOT / "algorithms")
        validate_references(cards, _load_references())

    def test_framework_alignment_is_consistent(self):
        cards = load_algorithm_cards(_ATLAS_ROOT / "algorithms")
        validate_framework_alignment(cards, frozenset(discover_causal_methods()))


if __name__ == "__main__":
    unittest.main()
