from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from causal_algorithms_atlas.evidence import (
    EvidenceError,
    load_evidence_records,
    validate_evidence_algorithm_ids,
)
from causal_algorithms_atlas.loader import load_algorithm_cards

_ATLAS_ROOT = Path(__file__).resolve().parent.parent / "causal_algorithms_atlas"

_VALID_RECORD = textwrap.dedent(
    """\
    algorithm_id: toy_method
    dataset: toy_a_linear
    source: "datasets/synthetic_causal/toy_a_linear.csv"
    metrics:
      precision: 0.75
      recall: 1.0
      f1_score: 0.857
    notes: "Exemplo de teste."
    """
)


class LoadEvidenceRecordsTests(unittest.TestCase):
    def test_loads_all_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "toy_method__toy_a_linear.yaml").write_text(
                _VALID_RECORD, encoding="utf-8"
            )
            records = load_evidence_records(tmp)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].algorithm_id, "toy_method")
        self.assertAlmostEqual(records[0].metrics["precision"], 0.75)


class ValidateEvidenceAlgorithmIdsTests(unittest.TestCase):
    def test_passes_when_id_known(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "toy_method__toy_a_linear.yaml").write_text(
                _VALID_RECORD, encoding="utf-8"
            )
            records = load_evidence_records(tmp)
        validate_evidence_algorithm_ids(records, frozenset({"toy_method"}))

    def test_raises_when_id_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "toy_method__toy_a_linear.yaml").write_text(
                _VALID_RECORD, encoding="utf-8"
            )
            records = load_evidence_records(tmp)
        with self.assertRaises(EvidenceError):
            validate_evidence_algorithm_ids(records, frozenset({"other_method"}))


class RealEvidenceContentTests(unittest.TestCase):
    def test_all_evidence_algorithm_ids_exist_in_atlas(self):
        cards = load_algorithm_cards(_ATLAS_ROOT / "algorithms")
        records = load_evidence_records(_ATLAS_ROOT / "evidence")
        validate_evidence_algorithm_ids(records, frozenset(cards))


if __name__ == "__main__":
    unittest.main()
