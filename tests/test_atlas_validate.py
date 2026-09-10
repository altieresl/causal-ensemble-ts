from __future__ import annotations

import unittest

from causal_algorithms_atlas.loader import parse_algorithm_card
from causal_algorithms_atlas.validate import (
    ValidationError,
    validate_framework_alignment,
    validate_references,
)

_CARD_TEXT = """---
id: toy_method
name: "Toy Method"
aliases: []
family: constraint-based
temporal_handling: native
output_type: dag
assumptions: []
handles_latent_confounders: false
handles_nonlinearity: false
handles_contemporaneous_effects: false
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "PCMCI"
references: [toyref]
verification: verified
verified_by: "paper-cross-check"
last_reviewed: "2026-09-07"
---

## Core idea

x

## Assumptions

x

## When to use

x

## When to avoid

x

## Relationship to other methods

x
"""


class ValidateReferencesTests(unittest.TestCase):
    def test_passes_when_reference_exists(self):
        card = parse_algorithm_card(_CARD_TEXT, source_path="t.md")
        validate_references({"toy_method": card}, {"toyref": {}})

    def test_raises_when_reference_missing(self):
        card = parse_algorithm_card(_CARD_TEXT, source_path="t.md")
        with self.assertRaises(ValidationError):
            validate_references({"toy_method": card}, {})


class ValidateFrameworkAlignmentTests(unittest.TestCase):
    def test_passes_when_name_registered(self):
        card = parse_algorithm_card(_CARD_TEXT, source_path="t.md")
        validate_framework_alignment({"toy_method": card}, frozenset({"PCMCI"}))

    def test_raises_when_name_not_registered(self):
        card = parse_algorithm_card(_CARD_TEXT, source_path="t.md")
        with self.assertRaises(ValidationError):
            validate_framework_alignment({"toy_method": card}, frozenset({"OTHER"}))

    def test_raises_when_implemented_but_name_missing(self):
        broken_text = _CARD_TEXT.replace(
            'framework_method_name: "PCMCI"', "framework_method_name: null"
        )
        card = parse_algorithm_card(broken_text, source_path="t.md")
        with self.assertRaises(ValidationError):
            validate_framework_alignment({"toy_method": card}, frozenset({"PCMCI"}))


if __name__ == "__main__":
    unittest.main()
