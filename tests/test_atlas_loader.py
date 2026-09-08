from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from causal_algorithms_atlas.loader import (
    LoaderError,
    REQUIRED_SECTIONS,
    load_algorithm_cards,
    parse_algorithm_card,
)
from causal_algorithms_atlas.schema import AlgorithmFamily, VerificationStatus

_VALID_CARD = textwrap.dedent(
    """\
    ---
    id: toy_method
    name: "Toy Method"
    aliases: []
    family: constraint-based
    temporal_handling: native
    output_type: dag
    assumptions:
      - id: stationarity
        required: true
        statement: "Processo estacionario."
    handles_latent_confounders: false
    handles_nonlinearity: false
    handles_contemporaneous_effects: false
    data_requirements:
      min_variables: 2
      min_timepoints: null
      sample_type: single-series
    implemented_in_framework: false
    framework_method_name: null
    references: [toyref]
    verification: draft
    verified_by: null
    last_reviewed: "2026-09-07"
    ---

    ## Ideia central

    Texto de exemplo.

    ## Premissas

    Texto de exemplo.

    ## Quando usar

    Texto de exemplo.

    ## Quando evitar

    Texto de exemplo.

    ## Relação com outros métodos

    Texto de exemplo.
    """
)


class ParseAlgorithmCardTests(unittest.TestCase):
    def test_parses_valid_card(self):
        card = parse_algorithm_card(_VALID_CARD, source_path="toy_method.md")
        self.assertEqual(card.id, "toy_method")
        self.assertEqual(card.family, AlgorithmFamily.CONSTRAINT_BASED)
        self.assertEqual(card.verification, VerificationStatus.DRAFT)
        for section in REQUIRED_SECTIONS:
            self.assertIn(section, card.sections)
        self.assertEqual(card.sections["Ideia central"], "Texto de exemplo.")

    def test_missing_frontmatter_delimiter_raises(self):
        with self.assertRaises(LoaderError):
            parse_algorithm_card("# sem frontmatter\n", source_path="bad.md")

    def test_missing_required_section_raises(self):
        broken = _VALID_CARD.replace("## Quando evitar", "## Secao Errada")
        with self.assertRaises(LoaderError):
            parse_algorithm_card(broken, source_path="bad.md")


class LoadAlgorithmCardsTests(unittest.TestCase):
    def test_loads_all_cards_keyed_by_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy_method.md"
            path.write_text(_VALID_CARD, encoding="utf-8")
            cards = load_algorithm_cards(tmp)
        self.assertEqual(set(cards), {"toy_method"})

    def test_duplicate_id_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.md").write_text(_VALID_CARD, encoding="utf-8")
            (Path(tmp) / "b.md").write_text(_VALID_CARD, encoding="utf-8")
            with self.assertRaises(LoaderError):
                load_algorithm_cards(tmp)


if __name__ == "__main__":
    unittest.main()
