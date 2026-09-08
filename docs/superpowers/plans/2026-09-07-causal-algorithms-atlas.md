# Causal Algorithms Atlas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, verification-gated knowledge base of causal-discovery-in-time-series
algorithms (`causal_algorithms_atlas/`), plus an EDA pass over the base and a debug RAG chat
backed by a local Llama model, so the base can later feed a real RAG system for ensemble
algorithm selection.

**Architecture:** A new top-level Python package holds both content (Markdown cards with YAML
frontmatter, one per algorithm) and tooling (`schema.py`, `loader.py`, `validate.py`,
`export.py`) that parses, validates, and exports that content. It has zero import dependency
on `causal_discovery/`; the only coupling is a content test that cross-checks the atlas against
`causal_discovery.discover_causal_methods()`. EDA and the RAG chat are later modules in the
same package, consuming the exported chunks.

**Tech Stack:** Python 3.13, PyYAML (new dependency), scikit-learn (already present, used for
TF-IDF retrieval), plotly (already present, used for EDA charts), Ollama local HTTP API (stdlib
`urllib.request`, no new dependency), pytest/unittest (existing test style in `tests/`).

**Spec:** `docs/superpowers/specs/2026-09-07-causal-algorithms-atlas-design.md`

## Global Constraints

- Only algorithms with a verifiable primary reference (DOI/arXiv/peer-reviewed venue) get a
  card; unverifiable fields stay `null`, never guessed.
- Frontmatter field **keys and enum values** are English (controlled vocabulary); prose body is
  Portuguese (pt-BR).
- `verification: verified` requires a non-empty `verified_by`.
- Every method returned by `causal_discovery.discover_causal_methods()` must have a card with
  `verification: verified`.
- No new dependency except `pyyaml`. Retrieval uses scikit-learn's `TfidfVectorizer`; HTTP calls
  to Ollama use stdlib `urllib.request`.
- Follow `AGENTS.md`: `python -m compileall causal_discovery causal_algorithms_atlas` after
  Python changes; run `python -m pytest tests -q` before finishing (this work touches nothing
  shared in `causal_discovery/`, but the suite must still pass since new tests are added to
  `tests/`).
- Commit after each task.

---

## Task 1: Package skeleton and controlled vocabulary (`schema.py`)

**Files:**
- Create: `causal_algorithms_atlas/__init__.py`
- Create: `causal_algorithms_atlas/schema.py`
- Test: `tests/test_atlas_schema.py`

**Interfaces:**
- Produces: `AlgorithmFamily`, `TemporalHandling`, `OutputType`, `SampleType`,
  `VerificationStatus` (all `str, Enum` subclasses); `KNOWN_ASSUMPTIONS: frozenset[str]`;
  `Assumption` dataclass (`id: str`, `required: bool`, `statement: str`);
  `DataRequirements` dataclass (`min_variables: int | None`, `min_timepoints: int | None`,
  `sample_type: SampleType | None`); `AlgorithmCard` dataclass with all frontmatter fields plus
  `sections: dict[str, str]` (prose body sections) and `source_path: str`; `SchemaError(ValueError)`
  raised by `AlgorithmCard.__post_init__` on structural violations (empty `id`/`name`,
  `verification == VerificationStatus.VERIFIED and not verified_by`, unknown `Assumption.id`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_atlas_schema.py
from __future__ import annotations

import unittest

from causal_algorithms_atlas.schema import (
    Assumption,
    AlgorithmCard,
    AlgorithmFamily,
    DataRequirements,
    KNOWN_ASSUMPTIONS,
    OutputType,
    SampleType,
    SchemaError,
    TemporalHandling,
    VerificationStatus,
)


def _make_card(**overrides):
    defaults = dict(
        id="toy_method",
        name="Toy Method",
        aliases=(),
        family=AlgorithmFamily.CONSTRAINT_BASED,
        temporal_handling=TemporalHandling.NATIVE,
        output_type=OutputType.DAG,
        assumptions=(Assumption(id="stationarity", required=True, statement="x"),),
        handles_latent_confounders=False,
        handles_nonlinearity=False,
        handles_contemporaneous_effects=False,
        data_requirements=DataRequirements(
            min_variables=2, min_timepoints=None, sample_type=SampleType.SINGLE_SERIES
        ),
        implemented_in_framework=False,
        framework_method_name=None,
        references=("toyref",),
        verification=VerificationStatus.DRAFT,
        verified_by=None,
        last_reviewed="2026-09-07",
        sections={"Ideia central": "texto"},
        source_path="causal_algorithms_atlas/algorithms/toy_method.md",
    )
    defaults.update(overrides)
    return AlgorithmCard(**defaults)


class KnownAssumptionsTests(unittest.TestCase):
    def test_contains_core_assumptions(self):
        expected = {
            "causal_sufficiency",
            "stationarity",
            "linearity",
            "acyclicity_instantaneous",
            "non_gaussian_errors",
            "faithfulness",
            "markov_condition",
            "no_selection_bias",
        }
        self.assertEqual(KNOWN_ASSUMPTIONS, frozenset(expected))


class AlgorithmCardTests(unittest.TestCase):
    def test_builds_with_valid_fields(self):
        card = _make_card()
        self.assertEqual(card.id, "toy_method")
        self.assertEqual(card.family, AlgorithmFamily.CONSTRAINT_BASED)

    def test_rejects_empty_id(self):
        with self.assertRaises(SchemaError):
            _make_card(id="")

    def test_rejects_verified_without_verified_by(self):
        with self.assertRaises(SchemaError):
            _make_card(verification=VerificationStatus.VERIFIED, verified_by=None)

    def test_accepts_verified_with_verified_by(self):
        card = _make_card(
            verification=VerificationStatus.VERIFIED, verified_by="paper-cross-check"
        )
        self.assertEqual(card.verification, VerificationStatus.VERIFIED)

    def test_rejects_unknown_assumption_id(self):
        with self.assertRaises(SchemaError):
            _make_card(
                assumptions=(Assumption(id="not_a_real_assumption", required=True, statement="x"),)
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_atlas_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'causal_algorithms_atlas'`

- [ ] **Step 3: Implement `causal_algorithms_atlas/__init__.py`**

```python
"""Verification-gated knowledge base of causal-discovery-in-time-series algorithms."""

from __future__ import annotations
```

- [ ] **Step 4: Implement `causal_algorithms_atlas/schema.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SchemaError(ValueError):
    """Raised when a parsed algorithm card violates the atlas schema."""


class AlgorithmFamily(str, Enum):
    CONSTRAINT_BASED = "constraint-based"
    SCORE_BASED = "score-based"
    FUNCTIONAL_CAUSAL_MODEL = "functional-causal-model"
    GRANGER_BASED = "granger-based"
    CONTINUOUS_OPTIMIZATION = "continuous-optimization"
    INFORMATION_THEORETIC = "information-theoretic"
    HYBRID = "hybrid"


class TemporalHandling(str, Enum):
    NATIVE = "native"
    WINDOWED_UNROLLING = "windowed-unrolling"
    SEGMENTATION = "segmentation"
    NONE = "none"


class OutputType(str, Enum):
    DAG = "dag"
    PAG = "pag"
    SIGNED_GRAPH = "signed-graph"
    UNSIGNED_GRAPH = "unsigned-graph"
    PARTIAL_GRAPH = "partial-graph"


class SampleType(str, Enum):
    SINGLE_SERIES = "single-series"
    PANEL = "panel"
    BOTH = "both"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    DRAFT = "draft"


KNOWN_ASSUMPTIONS: frozenset[str] = frozenset(
    {
        "causal_sufficiency",
        "stationarity",
        "linearity",
        "acyclicity_instantaneous",
        "non_gaussian_errors",
        "faithfulness",
        "markov_condition",
        "no_selection_bias",
    }
)


@dataclass(frozen=True)
class Assumption:
    id: str
    required: bool
    statement: str

    def __post_init__(self) -> None:
        if self.id not in KNOWN_ASSUMPTIONS:
            raise SchemaError(
                f"Assumption id {self.id!r} nao esta em KNOWN_ASSUMPTIONS: "
                f"{sorted(KNOWN_ASSUMPTIONS)}."
            )


@dataclass(frozen=True)
class DataRequirements:
    min_variables: int | None
    min_timepoints: int | None
    sample_type: SampleType | None


@dataclass(frozen=True)
class AlgorithmCard:
    id: str
    name: str
    aliases: tuple[str, ...]
    family: AlgorithmFamily
    temporal_handling: TemporalHandling
    output_type: OutputType
    assumptions: tuple[Assumption, ...]
    handles_latent_confounders: bool
    handles_nonlinearity: bool
    handles_contemporaneous_effects: bool
    data_requirements: DataRequirements
    implemented_in_framework: bool
    framework_method_name: str | None
    references: tuple[str, ...]
    verification: VerificationStatus
    verified_by: str | None
    last_reviewed: str
    sections: dict[str, str] = field(default_factory=dict)
    source_path: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise SchemaError("O campo 'id' da ficha nao pode ser vazio.")
        if not self.name.strip():
            raise SchemaError("O campo 'name' da ficha nao pode ser vazio.")
        if self.verification is VerificationStatus.VERIFIED and not (
            self.verified_by and self.verified_by.strip()
        ):
            raise SchemaError(
                f"Ficha {self.id!r}: verification=verified exige 'verified_by' preenchido."
            )
        if not self.references:
            raise SchemaError(f"Ficha {self.id!r}: precisa de ao menos uma referencia.")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_atlas_schema.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Compile check**

Run: `python -m compileall causal_algorithms_atlas`
Expected: no syntax errors reported

- [ ] **Step 7: Commit**

```bash
git add causal_algorithms_atlas/__init__.py causal_algorithms_atlas/schema.py tests/test_atlas_schema.py
git commit -m "feat: adicionar schema tipado do causal_algorithms_atlas"
```

---

## Task 2: Markdown+frontmatter loader (`loader.py`)

**Files:**
- Create: `causal_algorithms_atlas/loader.py`
- Test: `tests/test_atlas_loader.py`

**Interfaces:**
- Consumes: `causal_algorithms_atlas.schema.{AlgorithmCard, Assumption, DataRequirements,
  AlgorithmFamily, TemporalHandling, OutputType, SampleType, VerificationStatus, SchemaError}`
- Produces: `LoaderError(ValueError)`; `parse_algorithm_card(text: str, *, source_path: str) ->
  AlgorithmCard` (parses one Markdown file's content); `load_algorithm_cards(directory:
  str | Path) -> dict[str, AlgorithmCard]` (keyed by `id`, raises `LoaderError` on duplicate
  ids); `REQUIRED_SECTIONS: tuple[str, ...] = ("Ideia central", "Premissas", "Quando usar",
  "Quando evitar", "Relação com outros métodos")`.

Frontmatter is delimited by `---` lines at the top of the file (standard Jekyll-style
frontmatter), parsed with `yaml.safe_load`. Body sections are Markdown `##` headings; the text
between one heading and the next (or EOF) is that section's value, stripped.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_atlas_loader.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_atlas_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'causal_algorithms_atlas.loader'`

- [ ] **Step 3: Implement `causal_algorithms_atlas/loader.py`**

```python
from __future__ import annotations

import re
from pathlib import Path

import yaml

from causal_algorithms_atlas.schema import (
    AlgorithmCard,
    AlgorithmFamily,
    Assumption,
    DataRequirements,
    OutputType,
    SampleType,
    SchemaError,
    TemporalHandling,
    VerificationStatus,
)

REQUIRED_SECTIONS: tuple[str, ...] = (
    "Ideia central",
    "Premissas",
    "Quando usar",
    "Quando evitar",
    "Relação com outros métodos",
)

_FRONTMATTER_PATTERN = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)
_HEADING_PATTERN = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


class LoaderError(ValueError):
    """Raised when an algorithm card file cannot be parsed."""


def _split_sections(body: str) -> dict[str, str]:
    matches = list(_HEADING_PATTERN.finditer(body))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[match.group(1)] = body[start:end].strip()
    return sections


def _build_assumptions(raw: list[dict]) -> tuple[Assumption, ...]:
    return tuple(
        Assumption(id=item["id"], required=bool(item["required"]), statement=item["statement"])
        for item in raw
    )


def _build_data_requirements(raw: dict) -> DataRequirements:
    sample_type = raw.get("sample_type")
    return DataRequirements(
        min_variables=raw.get("min_variables"),
        min_timepoints=raw.get("min_timepoints"),
        sample_type=SampleType(sample_type) if sample_type is not None else None,
    )


def parse_algorithm_card(text: str, *, source_path: str) -> AlgorithmCard:
    match = _FRONTMATTER_PATTERN.match(text)
    if not match:
        raise LoaderError(
            f"{source_path}: frontmatter YAML delimitado por '---' nao encontrado."
        )
    frontmatter_text, body = match.group(1), match.group(2)
    try:
        raw = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as exc:
        raise LoaderError(f"{source_path}: YAML invalido no frontmatter ({exc}).") from exc

    sections = _split_sections(body)
    missing = [name for name in REQUIRED_SECTIONS if name not in sections]
    if missing:
        raise LoaderError(f"{source_path}: secoes obrigatorias ausentes: {missing}.")

    try:
        card = AlgorithmCard(
            id=raw["id"],
            name=raw["name"],
            aliases=tuple(raw.get("aliases") or ()),
            family=AlgorithmFamily(raw["family"]),
            temporal_handling=TemporalHandling(raw["temporal_handling"]),
            output_type=OutputType(raw["output_type"]),
            assumptions=_build_assumptions(raw.get("assumptions") or []),
            handles_latent_confounders=bool(raw["handles_latent_confounders"]),
            handles_nonlinearity=bool(raw["handles_nonlinearity"]),
            handles_contemporaneous_effects=bool(raw["handles_contemporaneous_effects"]),
            data_requirements=_build_data_requirements(raw["data_requirements"]),
            implemented_in_framework=bool(raw["implemented_in_framework"]),
            framework_method_name=raw.get("framework_method_name"),
            references=tuple(raw.get("references") or ()),
            verification=VerificationStatus(raw["verification"]),
            verified_by=raw.get("verified_by"),
            last_reviewed=str(raw["last_reviewed"]),
            sections=sections,
            source_path=source_path,
        )
    except (KeyError, ValueError, SchemaError) as exc:
        raise LoaderError(f"{source_path}: {exc}") from exc
    return card


def load_algorithm_cards(directory: str | Path) -> dict[str, AlgorithmCard]:
    directory = Path(directory)
    cards: dict[str, AlgorithmCard] = {}
    for path in sorted(directory.glob("*.md")):
        card = parse_algorithm_card(path.read_text(encoding="utf-8"), source_path=str(path))
        if card.id in cards:
            raise LoaderError(
                f"Id de algoritmo duplicado {card.id!r}: {cards[card.id].source_path} e {path}."
            )
        cards[card.id] = card
    return cards
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_atlas_loader.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add causal_algorithms_atlas/loader.py tests/test_atlas_loader.py
git commit -m "feat: adicionar loader de fichas markdown+frontmatter do atlas"
```

---

## Task 3: Cross-reference validation (`validate.py`)

**Files:**
- Create: `causal_algorithms_atlas/validate.py`
- Test: `tests/test_atlas_validate.py`

**Interfaces:**
- Consumes: `causal_algorithms_atlas.schema.AlgorithmCard`; `causal_algorithms_atlas.loader.load_algorithm_cards`
- Produces: `ValidationError(ValueError)`; `validate_references(cards: dict[str, AlgorithmCard],
  references: dict[str, dict]) -> None` (raises if any `card.references` entry is not a key in
  `references`); `validate_framework_alignment(cards: dict[str, AlgorithmCard],
  framework_method_names: frozenset[str]) -> None` (raises if any card's
  `framework_method_name` is not `None` and not in `framework_method_names`, or if it is `None`
  while `implemented_in_framework` is `True`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_atlas_validate.py
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

## Ideia central

x

## Premissas

x

## Quando usar

x

## Quando evitar

x

## Relação com outros métodos

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_atlas_validate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'causal_algorithms_atlas.validate'`

- [ ] **Step 3: Implement `causal_algorithms_atlas/validate.py`**

```python
from __future__ import annotations

from causal_algorithms_atlas.schema import AlgorithmCard


class ValidationError(ValueError):
    """Raised when cross-referenced atlas content is inconsistent."""


def validate_references(
    cards: dict[str, AlgorithmCard], references: dict[str, dict]
) -> None:
    for card in cards.values():
        missing = [key for key in card.references if key not in references]
        if missing:
            raise ValidationError(
                f"Ficha {card.id!r} referencia chaves ausentes em references.yaml: {missing}."
            )


def validate_framework_alignment(
    cards: dict[str, AlgorithmCard], framework_method_names: frozenset[str]
) -> None:
    for card in cards.values():
        if card.implemented_in_framework and not card.framework_method_name:
            raise ValidationError(
                f"Ficha {card.id!r}: implemented_in_framework=true exige framework_method_name."
            )
        if (
            card.framework_method_name is not None
            and card.framework_method_name not in framework_method_names
        ):
            raise ValidationError(
                f"Ficha {card.id!r}: framework_method_name={card.framework_method_name!r} "
                f"nao esta registrado no framework ({sorted(framework_method_names)})."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_atlas_validate.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add causal_algorithms_atlas/validate.py tests/test_atlas_validate.py
git commit -m "feat: adicionar validacao cruzada de referencias e alinhamento com o framework"
```

---

## Task 4: `references.yaml` and the 8 framework algorithm cards

**Files:**
- Create: `causal_algorithms_atlas/references.yaml`
- Create: `causal_algorithms_atlas/algorithms/pcmci.md`
- Create: `causal_algorithms_atlas/algorithms/lpcmci.md`
- Create: `causal_algorithms_atlas/algorithms/classical_granger.md`
- Create: `causal_algorithms_atlas/algorithms/neural_granger_cmlp.md`
- Create: `causal_algorithms_atlas/algorithms/var_lingam.md`
- Create: `causal_algorithms_atlas/algorithms/dynotears.md`
- Create: `causal_algorithms_atlas/algorithms/ges.md`
- Create: `causal_algorithms_atlas/algorithms/fci.md`
- Test: `tests/test_atlas_content.py`

**Interfaces:**
- Consumes: `causal_algorithms_atlas.loader.load_algorithm_cards`,
  `causal_algorithms_atlas.validate.{validate_references, validate_framework_alignment}`,
  `causal_discovery.discover_causal_methods`
- Produces: fixed content invariant relied on by later tasks — every id in
  `{pcmci, lpcmci, classical_granger, neural_granger_cmlp, var_lingam, dynotears, ges, fci}`
  exists with `verification: verified`.

Content for these 8 cards is already verified via `.local/METHODS.md` (cross-checked against
the wrapper source in `causal_discovery/methods/`) plus the original papers. `references.yaml`
entries below use the DOIs already present in `.local/METHODS.md`.

- [ ] **Step 1: Create `causal_algorithms_atlas/references.yaml`**

```yaml
runge2019:
  authors: "Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., Sejdinovic, D."
  year: 2019
  title: "Detecting and quantifying causal associations in large nonlinear time series datasets"
  venue: "Science Advances"
  doi: "10.1126/sciadv.aau4996"
  verified: true

gerhardus2020:
  authors: "Gerhardus, A., Runge, J."
  year: 2020
  title: "High-recall causal discovery for autocorrelated time series with latent confounders"
  venue: "NeurIPS"
  url: "https://proceedings.neurips.cc/paper_files/paper/2020/hash/94e70705efae423efda1088614128d0b-Abstract.html"
  verified: true

granger1969:
  authors: "Granger, C. W. J."
  year: 1969
  title: "Investigating Causal Relations by Econometric Models and Cross-spectral Methods"
  venue: "Econometrica"
  url: "https://www.jstor.org/stable/1912791"
  verified: true

tank2021:
  authors: "Tank, A., Covert, I., Foti, N., Shojaie, A., Fox, E. B."
  year: 2021
  title: "Neural Granger Causality"
  venue: "IEEE Transactions on Pattern Analysis and Machine Intelligence"
  doi: "10.1109/TPAMI.2021.3065601"
  verified: true

hyvarinen2010:
  authors: "Hyvärinen, A., Zhang, K., Shimizu, S., Hoyer, P. O."
  year: 2010
  title: "Estimation of a Structural Vector Autoregression Model Using Non-Gaussianity"
  venue: "Journal of Machine Learning Research"
  url: "https://www.jmlr.org/papers/v11/hyvarinen10a.html"
  verified: true

pamfil2020:
  authors: "Pamfil, R., Sriwattanaworachai, N., Desai, S., Pilgerstorfer, P., Georgatzis, K., Beaumont, P., Aragam, B."
  year: 2020
  title: "DYNOTEARS: Structure Learning from Time-Series Data"
  venue: "AISTATS / PMLR"
  url: "https://proceedings.mlr.press/v108/pamfil20a.html"
  verified: true

chickering2002:
  authors: "Chickering, D. M."
  year: 2002
  title: "Optimal Structure Identification With Greedy Search"
  venue: "Journal of Machine Learning Research"
  url: "https://www.jmlr.org/papers/v3/chickering02b.html"
  verified: true

spirtes1995:
  authors: "Spirtes, P., Meek, C., Richardson, T."
  year: 1995
  title: "Causal Inference in the Presence of Latent Variables and Selection Bias"
  venue: "UAI"
  url: "https://www.cs.cmu.edu/afs/cs/project/learn-43/lib/photoz/.g/web/.g/scottd/fullbook.pdf"
  verified: true
```

- [ ] **Step 2: Create `causal_algorithms_atlas/algorithms/pcmci.md`**

```markdown
---
id: pcmci
name: "PCMCI"
aliases: []
family: constraint-based
temporal_handling: native
output_type: dag
assumptions:
  - id: causal_sufficiency
    required: true
    statement: "Ausencia de confundidores latentes nao medidos."
  - id: stationarity
    required: true
    statement: "Processo estacionario no intervalo analisado."
  - id: faithfulness
    required: true
    statement: "Independencias observadas refletem a estrutura causal, nao coincidencia."
handles_latent_confounders: false
handles_nonlinearity: false
handles_contemporaneous_effects: false
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "PCMCI"
references: [runge2019]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-07"
---

## Ideia central

PCMCI combina duas etapas: primeiro, uma selecao de condicionantes (PC1) reduz o
conjunto de pais candidatos de cada variavel usando testes de independencia condicional
iterativos; depois, o teste MCI (Momentary Conditional Independence) avalia cada
relacao remanescente condicionando tanto nos pais estimados do alvo quanto nos da
origem, o que controla autocorrelacao e confundimento indireto ao mesmo tempo. O
wrapper do framework usa `ParCorr` (correlacao parcial) como teste de independencia,
portanto a dependencia condicional testada e linear mesmo que o metodo em si nao seja
restrito a isso na formulacao geral.

## Premissas

Exige suficiencia causal (nenhum confundidor latente relevante), estacionariedade no
trecho analisado e fidelidade causal. Com `ParCorr`, adicionalmente assume relacoes
lineares gaussianas para os testes de independencia terem poder estatistico adequado.

## Quando usar

Series com muitas variaveis e autocorrelacao temporal forte, quando se pode assumir
ausencia de confundidores latentes relevantes. Bom ponto de partida por ser rapido e
por retornar apenas relacoes lagged definitivamente direcionadas (sem ambiguidade de
orientacao), o que facilita interpretacao.

## Quando evitar

Quando ha suspeita forte de confundidor latente (nesse caso, considerar LPCMCI) ou
quando a relacao de interesse e genuinamente nao linear e o teste `ParCorr` mascara a
dependencia.

## Relação com outros métodos

E o predecessor direto do LPCMCI, que relaxa a suficiencia causal ao custo de retornar
marcas ambiguas em vez de arestas totalmente orientadas. Comparado a Classical Granger,
PCMCI condiciona em um conjunto de pais selecionado por dados em vez de usar todos os
lags disponiveis, o que reduz falsos positivos em redes densas.

## Notas de implementação

Wrapper em `causal_discovery/methods/pcmci.py`, usando o PCMCI do pacote `tigramite`
com `ParCorr`. Retorna apenas relacoes lagged definitivamente direcionadas.
```

- [ ] **Step 3: Create `causal_algorithms_atlas/algorithms/lpcmci.md`**

```markdown
---
id: lpcmci
name: "LPCMCI"
aliases: []
family: constraint-based
temporal_handling: native
output_type: pag
assumptions:
  - id: stationarity
    required: true
    statement: "Processo estacionario no intervalo analisado."
  - id: faithfulness
    required: true
    statement: "Independencias observadas refletem a estrutura causal, nao coincidencia."
handles_latent_confounders: true
handles_nonlinearity: false
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "LPCMCI"
references: [gerhardus2020]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-07"
---

## Ideia central

LPCMCI generaliza o PCMCI para o caso latente, retornando um DPAG (Directed Partial
Ancestral Graph) em vez de um DAG totalmente orientado. Marcas de aresta podem ficar
ambiguas (`o-o`, `o->`) quando os dados nao permitem determinar a orientacao com
seguranca na presenca de confundidores nao observados.

## Premissas

Nao exige suficiencia causal — essa e a motivacao central do metodo. Ainda exige
estacionariedade e fidelidade causal. O wrapper do framework usa `ParCorr`, herdando a
mesma limitacao a dependencia linear que o PCMCI.

## Quando usar

Quando ha suspeita razoavel de confundidor latente relevante e se aceita trabalhar com
saida parcialmente orientada (PAG) em vez de um DAG completo.

## Quando evitar

Quando se precisa de uma aresta totalmente direcionada para toda relacao candidata: o
framework nao converte marcas ambiguas ou bidirecionais do DPAG em setas causais, entao
relacoes com marca ambigua simplesmente nao aparecem como evidencia direcionada na
saida.

## Relação com outros métodos

Relaxa a suposicao mais forte do PCMCI (suficiencia causal) ao custo de ambiguidade de
orientacao, de forma analoga a como FCI generaliza PC/GES no caso atemporal.

## Notas de implementação

Wrapper em `causal_discovery/methods/lpcmci.py`, usando o LPCMCI do `tigramite` com
`ParCorr`. Marcas ambiguas ou bidirecionais nao sao convertidas em arestas causais.
```

- [ ] **Step 4: Create `causal_algorithms_atlas/algorithms/classical_granger.md`**

```markdown
---
id: classical_granger
name: "Classical Granger"
aliases: ["Granger Causality"]
family: granger-based
temporal_handling: native
output_type: signed-graph
assumptions:
  - id: stationarity
    required: true
    statement: "Series estacionarias (ou tornadas estacionarias por diferenciacao previa)."
  - id: linearity
    required: true
    statement: "Relacao preditiva linear entre os lags e o alvo."
handles_latent_confounders: false
handles_nonlinearity: false
handles_contemporaneous_effects: false
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "ClassicalGranger"
references: [granger1969]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-07"
---

## Ideia central

Testa se os valores passados de uma serie `X` melhoram a previsao linear de outra
serie `Y` alem do que os proprios valores passados de `Y` ja explicam. E um teste
bivariado e estritamente preditivo: "causa" aqui significa "tem poder preditivo
incremental", nao causalidade estrutural no sentido de intervencao.

## Premissas

Estacionariedade das series e uma forma funcional linear entre lags e alvo. O teste
conjunto usado no wrapper considera todos os lags simultaneamente, mas a significancia
de cada coeficiente individual e reportada separadamente.

## Quando usar

Como baseline rapido e interpretavel, ou quando a relacao realmente e bivariada e
aproximadamente linear. Boa referencia de comparacao para os metodos multivariados do
ensemble.

## Quando evitar

Quando ha confundidores comuns entre `X` e `Y` que nao entram no teste bivariado: nesse
caso, causalidade de Granger bivariada pode indicar uma relacao espuria induzida pelo
confundidor. Tambem inadequado se a relacao for genuinamente nao linear.

## Relação com outros métodos

E o fundamento historico de toda a familia "Granger-based" do projeto, incluindo Neural
Granger cMLP, que generaliza o teste para relacoes nao lineares multivariadas com uma
rede por alvo.

## Notas de implementação

Wrapper em `causal_discovery/methods/classical_granger.py`, usando o teste de
causalidade de Granger do Statsmodels. `signed_score=True` no registro do framework.
```

- [ ] **Step 5: Create `causal_algorithms_atlas/algorithms/neural_granger_cmlp.md`**

```markdown
---
id: neural_granger_cmlp
name: "Neural Granger cMLP"
aliases: ["cMLP", "Neural-GC"]
family: granger-based
temporal_handling: native
output_type: signed-graph
assumptions:
  - id: stationarity
    required: true
    statement: "Series aproximadamente estacionarias no intervalo analisado."
handles_latent_confounders: false
handles_nonlinearity: true
handles_contemporaneous_effects: false
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "NeuralGrangercMLP"
references: [tank2021]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-07"
---

## Ideia central

Ajusta uma rede neural (MLP) por variavel-alvo, onde os pesos da primeira camada sao
organizados por serie de origem e penalizados com uma penalizacao proximal estruturada
(`GL`, `GSGL` ou hierarquica) que zera grupos inteiros de pesos. Uma serie de origem e
considerada causa de Granger do alvo se algum peso do seu grupo permanece nao nulo apos
o ajuste, generalizando o teste classico de Granger para relacoes nao lineares.

## Premissas

Nao exige linearidade — essa e a motivacao central do metodo — mas ainda assume que a
nocao de causalidade de Granger (poder preditivo incremental a partir do passado) e a
pergunta relevante, e que a serie e razoavelmente estacionaria para a rede generalizar
entre janelas de treino e avaliacao.

## Quando usar

Quando ha suspeita de relacao nao linear entre series e se aceita a interpretacao
preditiva (Granger) de causalidade em vez de uma estrutural.

## Quando evitar

Com poucas observacoes: redes neurais por alvo precisam de dados suficientes para a
penalizacao estruturada distinguir sinal de ruido, e os resultados tendem a saida densa
com muitos falsos positivos em amostras pequenas.

## Relação com outros métodos

Generalizacao nao linear direta do Classical Granger, seguindo o codigo de referencia
Neural-GC dos mesmos autores da formulacao.

## Notas de implementação

Wrapper em `causal_discovery/methods/neural_granger.py`. Uma rede por alvo, penalizacao
proximal estruturada configuravel via `default_kwargs`.
```

- [ ] **Step 6: Create `causal_algorithms_atlas/algorithms/var_lingam.md`**

```markdown
---
id: var_lingam
name: "VAR-LiNGAM"
aliases: []
family: functional-causal-model
temporal_handling: native
output_type: signed-graph
assumptions:
  - id: linearity
    required: true
    statement: "Relacoes lagged e instantaneas lineares."
  - id: acyclicity_instantaneous
    required: true
    statement: "Estrutura instantanea (lag 0) e aciclica."
  - id: non_gaussian_errors
    required: true
    statement: "Ruidos independentes e nao gaussianos."
handles_latent_confounders: false
handles_nonlinearity: false
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "VARLiNGAM"
references: [hyvarinen2010]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-07"
---

## Ideia central

Combina um modelo VAR (Vector Autoregression) para a parte lagged com LiNGAM (Linear
Non-Gaussian Acyclic Model) para orientar a estrutura instantanea (lag 0). A
nao-gaussianidade dos residuos e o que permite identificar a direcao causal instantanea
sem depender apenas de restricoes de independencia condicional.

## Premissas

Linearidade em todas as relacoes (lagged e instantaneas), aciclicidade da estrutura
instantanea, e residuos independentes e nao gaussianos — se os residuos forem
gaussianos, a orientacao instantanea deixa de ser identificavel pela teoria do metodo.

## Quando usar

Quando ha efeitos instantaneos (lag 0) plausiveis entre as series e razao para acreditar
que os residuos nao sao gaussianos (comum em dados financeiros e alguns sensores
fisicos).

## Quando evitar

Com residuos proximos de gaussianos ou quando se suspeita de ciclos na estrutura
instantanea: a identificacao de LiNGAM depende estruturalmente da nao-gaussianidade e
da aciclicidade.

## Relação com outros métodos

E o unico metodo do ensemble que modela explicitamente efeitos instantaneos (lag 0)
alem dos lagged, o que o torna complementar a PCMCI e DYNOTEARS, que tratam
majoritariamente ou exclusivamente relacoes lagged.

## Notas de implementação

Wrapper em `causal_discovery/methods/var_lingam.py`, usando `lingam.VARLiNGAM`, incluindo
efeitos instantaneos e lagged. `signed_score=True` no registro do framework.
```

- [ ] **Step 7: Create `causal_algorithms_atlas/algorithms/dynotears.md`**

```markdown
---
id: dynotears
name: "DYNOTEARS"
aliases: []
family: continuous-optimization
temporal_handling: native
output_type: signed-graph
assumptions:
  - id: linearity
    required: true
    statement: "Relacoes lagged e instantaneas lineares."
  - id: acyclicity_instantaneous
    required: true
    statement: "Estrutura instantanea (lag 0) e aciclica, imposta via restricao continua suave."
handles_latent_confounders: false
handles_nonlinearity: false
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: both
implemented_in_framework: true
framework_method_name: "DYNOTEARS"
references: [pamfil2020]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-07"
---

## Ideia central

Formula a descoberta causal como um problema de otimizacao continua: aprende
simultaneamente as matrizes de coeficientes lagged e instantanea minimizando erro de
reconstrucao com penalizacao L1 (esparsidade) sujeita a uma restricao de aciclicidade
suave (diferenciavel) sobre a estrutura instantanea, no estilo NOTEARS estendido para o
caso temporal.

## Premissas

Linearidade de todas as relacoes e aciclicidade da estrutura instantanea, imposta
explicitamente pela restricao de otimizacao (nao apenas assumida — o metodo falha em
convergir para uma estrutura com ciclos instantaneos por construcao).

## Quando usar

Quando o numero de variaveis e moderado a grande e se aceita a formulacao linear;
tambem util quando os dados sao um painel de series (multiplas replicas curtas) em vez
de uma unica serie longa, ja que a formulacao de otimizacao aceita ambos.

## Quando evitar

Quando a restricao de aciclicidade instantanea for implausivel para o dominio (ex.: se
ha razao para esperar um ciclo de feedback instantaneo entre variaveis).

## Relação com outros métodos

E a contraparte de VAR-LiNGAM que troca a identificacao via nao-gaussianidade por uma
restricao explicita de otimizacao para orientar a estrutura instantanea; nao exige
nao-gaussianidade dos residuos.

## Notas de implementação

Implementacao local em `causal_discovery/methods/dynotears.py`: formulacao linear com
restricao suave de aciclicidade e penalizacao L1 compartilhada. `signed_score=True`.
```

- [ ] **Step 8: Create `causal_algorithms_atlas/algorithms/ges.md`**

```markdown
---
id: ges
name: "GES"
aliases: ["Greedy Equivalence Search"]
family: score-based
temporal_handling: windowed-unrolling
output_type: dag
assumptions:
  - id: causal_sufficiency
    required: true
    statement: "Ausencia de confundidores latentes nao medidos."
  - id: faithfulness
    required: true
    statement: "Independencias observadas refletem a estrutura causal, nao coincidencia."
handles_latent_confounders: false
handles_nonlinearity: false
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "GES"
references: [chickering2002]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-07"
---

## Ideia central

GES e um algoritmo geral (nao temporal) que busca greedy em duas fases (forward,
adicionando arestas; backward, removendo) sobre o espaco de classes de equivalencia de
Markov, otimizando um score (por padrao, BIC) ate atingir um otimo local que a teoria
garante ser a estrutura correta quando as premissas valem e os dados sao suficientes.

## Premissas

Suficiencia causal e fidelidade — heranca do algoritmo geral, nao especifico ao caso
temporal. Ao ser adaptado para series no framework, o score usado assume forma
funcional compativel com BIC gaussiano.

## Quando usar

Como candidato adicional ao lado dos metodos temporais nativos (PCMCI, LPCMCI), para
capturar estrutura que uma busca por score pode encontrar e uma busca baseada em
restricoes pode perder, especialmente com poucas variaveis onde a busca greedy e
barata.

## Quando evitar

GES e FCI sao algoritmos gerais para dados tabulares, nao desenhados para series
temporais. Nao devem ser tratados como algoritmos temporalmente estacionarios como
PCMCI apenas por terem sido adaptados.

## Relação com outros métodos

No framework, GES e FCI recebem a mesma adaptacao: uma matriz desenrolada no tempo
(`variavel_t`, `variavel_lag_1`, ...), e a conversao de volta ao contrato temporal
considera apenas relacoes entre uma variavel defasada e uma variavel atual. Arestas nao
orientadas entre passado e presente sao orientadas pela ordem temporal conhecida.

## Notas de implementação

Wrapper em `causal_discovery/methods/causal_learn.py`, usando o GES oficial do pacote
`causal-learn` sobre a matriz temporal expandida.
```

- [ ] **Step 9: Create `causal_algorithms_atlas/algorithms/fci.md`**

```markdown
---
id: fci
name: "FCI"
aliases: ["Heterogeneous FCI"]
family: constraint-based
temporal_handling: windowed-unrolling
output_type: pag
assumptions:
  - id: faithfulness
    required: true
    statement: "Independencias observadas refletem a estrutura causal, nao coincidencia."
handles_latent_confounders: true
handles_nonlinearity: false
handles_contemporaneous_effects: true
data_requirements:
  min_variables: 2
  min_timepoints: null
  sample_type: single-series
implemented_in_framework: true
framework_method_name: "FCI"
references: [spirtes1995]
verification: verified
verified_by: "paper-cross-check+source-code"
last_reviewed: "2026-09-07"
---

## Ideia central

FCI (Fast Causal Inference) generaliza PC para o caso com confundidores latentes e
selecao amostral, retornando um PAG (Partial Ancestral Graph) em vez de um DAG. Como
GES, e um algoritmo geral adaptado ao caso temporal no framework via desenrolamento em
uma matriz `variavel_t`, `variavel_lag_1`, ....

## Premissas

Fidelidade causal — nao exige suficiencia causal, o que e a motivacao central do
metodo frente a PC/GES. No framework, conhecimento previo proibe arestas
presente -> passado na matriz desenrolada, refletindo a ordem temporal conhecida.

## Quando usar

Quando ha suspeita de confundidor latente e se aceita saida parcialmente orientada
(PAG); por padrao o framework retorna apenas arestas PAG definitivamente orientadas,
como o LPCMCI.

## Quando evitar

Mesma ressalva do GES: e um algoritmo geral para dados tabulares, nao nativamente
temporal. A adaptacao usa o algoritmo oficial mas nao o torna estacionario como PCMCI.

## Relação com outros métodos

Papel para GES o mesmo que LPCMCI faz para PCMCI: versão que tolera confundidor latente
trocando DAG totalmente orientado por PAG parcialmente orientado.

## Notas de implementação

Wrapper em `causal_discovery/methods/causal_learn.py` e alias em
`causal_discovery/methods/heterogeneous_fci.py` (`run_heterogeneous_fci` e mantido como
alias compativel de `run_fci`). Usa o FCI oficial do `causal-learn` sobre a matriz
temporal expandida, com conhecimento previo proibindo presente -> passado. Por padrao
retorna apenas arestas PAG definitivamente orientadas.
```

- [ ] **Step 10: Write the failing content test**

```python
# tests/test_atlas_content.py
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
```

- [ ] **Step 11: Run the content test**

Run: `python -m pytest tests/test_atlas_content.py -v`
Expected: PASS (3 tests). If it fails on `missing`, re-check the card whose
`framework_method_name` doesn't exactly match the registry name (case-sensitive, e.g.
`"NeuralGrangercMLP"` not `"NeuralGrangerCMLP"`) — check `causal_discovery/registry.py`'s
`_method_name_from_function` derivation if unsure.

- [ ] **Step 12: Run full loader/schema/validate suite plus the new content test together**

Run: `python -m pytest tests/test_atlas_schema.py tests/test_atlas_loader.py tests/test_atlas_validate.py tests/test_atlas_content.py -v`
Expected: all PASS

- [ ] **Step 13: Compile check**

Run: `python -m compileall causal_algorithms_atlas`

- [ ] **Step 14: Commit**

```bash
git add causal_algorithms_atlas/references.yaml causal_algorithms_atlas/algorithms tests/test_atlas_content.py
git commit -m "feat: adicionar fichas verificadas dos 8 metodos do framework"
```

---

## Task 5: Export to RAG-ready JSONL chunks (`export.py`)

**Files:**
- Create: `causal_algorithms_atlas/export.py`
- Test: `tests/test_atlas_export.py`

**Interfaces:**
- Consumes: `causal_algorithms_atlas.schema.AlgorithmCard`, `causal_algorithms_atlas.loader.load_algorithm_cards`
- Produces: `Chunk` dataclass (`id: str`, `algorithm_id: str`, `section: str`, `text: str`,
  `metadata: dict`); `cards_to_chunks(cards: dict[str, AlgorithmCard]) -> list[Chunk]`;
  `write_chunks_jsonl(chunks: list[Chunk], path: str | Path) -> None`; `SUMMARY_SECTION_NAME:
  str = "Resumo estruturado"`

The summary chunk's text is a deterministic Portuguese sentence built from the structured
fields (family, assumptions, handles_* flags), so queries phrased in terms of data
characteristics ("método que tolera confundidor latente") can match it directly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_atlas_export.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_atlas_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'causal_algorithms_atlas.export'`

- [ ] **Step 3: Implement `causal_algorithms_atlas/export.py`**

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from causal_algorithms_atlas.schema import AlgorithmCard

SUMMARY_SECTION_NAME = "Resumo estruturado"


@dataclass(frozen=True)
class Chunk:
    id: str
    algorithm_id: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)


def _card_metadata(card: AlgorithmCard) -> dict:
    return {
        "family": card.family.value,
        "temporal_handling": card.temporal_handling.value,
        "output_type": card.output_type.value,
        "handles_latent_confounders": card.handles_latent_confounders,
        "handles_nonlinearity": card.handles_nonlinearity,
        "handles_contemporaneous_effects": card.handles_contemporaneous_effects,
        "implemented_in_framework": card.implemented_in_framework,
        "verification": card.verification.value,
    }


def _summary_text(card: AlgorithmCard) -> str:
    confundidor = "tolera" if card.handles_latent_confounders else "nao tolera"
    nao_linear = "captura relacoes nao lineares" if card.handles_nonlinearity else "assume relacoes lineares"
    instantaneo = (
        "modela efeitos instantaneos (lag 0)"
        if card.handles_contemporaneous_effects
        else "nao modela efeitos instantaneos (lag 0)"
    )
    assumption_ids = ", ".join(a.id for a in card.assumptions) or "nenhuma premissa declarada"
    return (
        f"{card.name} e um metodo da familia {card.family.value}, com tratamento temporal "
        f"{card.temporal_handling.value}, que {confundidor} confundidores latentes, "
        f"{nao_linear} e {instantaneo}. Saida do tipo {card.output_type.value}. "
        f"Premissas declaradas: {assumption_ids}."
    )


def cards_to_chunks(cards: dict[str, AlgorithmCard]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for card in cards.values():
        metadata = _card_metadata(card)
        chunks.append(
            Chunk(
                id=f"{card.id}#{SUMMARY_SECTION_NAME}",
                algorithm_id=card.id,
                section=SUMMARY_SECTION_NAME,
                text=_summary_text(card),
                metadata=metadata,
            )
        )
        for section_name, section_text in card.sections.items():
            chunks.append(
                Chunk(
                    id=f"{card.id}#{section_name}",
                    algorithm_id=card.id,
                    section=section_name,
                    text=section_text,
                    metadata=metadata,
                )
            )
    return chunks


def write_chunks_jsonl(chunks: list[Chunk], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            record = {
                "id": chunk.id,
                "algorithm_id": chunk.algorithm_id,
                "section": chunk.section,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_atlas_export.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add causal_algorithms_atlas/export.py tests/test_atlas_export.py
git commit -m "feat: adicionar export de fichas para chunks jsonl (RAG)"
```

---

## Task 6: Empirical evidence layer (`evidence/`)

**Files:**
- Create: `causal_algorithms_atlas/evidence.py`
- Create: `causal_algorithms_atlas/evidence/pcmci__toy_a_linear.yaml`
- Create: `causal_algorithms_atlas/evidence/pcmci__toy_b_nonlinear.yaml`
- Create: `causal_algorithms_atlas/evidence/classical_granger__toy_a_linear.yaml`
- Create: `causal_algorithms_atlas/evidence/ges__toy_a_linear.yaml`
- Create: `causal_algorithms_atlas/evidence/fci__toy_a_linear.yaml`
- Create: `causal_algorithms_atlas/evidence/dynotears__toy_a_linear.yaml`
- Create: `causal_algorithms_atlas/evidence/lpcmci__toy_a_linear.yaml`
- Create: `causal_algorithms_atlas/evidence/neural_granger_cmlp__toy_a_linear.yaml`
- Create: `causal_algorithms_atlas/evidence/var_lingam__toy_a_linear.yaml`
- Test: `tests/test_atlas_evidence.py`

**Interfaces:**
- Consumes: `causal_algorithms_atlas.loader.load_algorithm_cards`
- Produces: `EvidenceRecord` dataclass (`algorithm_id: str`, `dataset: str`, `source: str`,
  `metrics: dict[str, float]`, `notes: str`); `EvidenceError(ValueError)`;
  `load_evidence_records(directory: str | Path) -> list[EvidenceRecord]`;
  `validate_evidence_algorithm_ids(records: list[EvidenceRecord], known_ids: frozenset[str]) ->
  None` (raises `EvidenceError` if any `record.algorithm_id` is not in `known_ids`)

Metric values below come from `.local/results/toy_synthetic_validation/metrics.csv` (already
read during design; that file is gitignored, so these YAML files are the versioned, durable
copy of that specific measurement).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_atlas_evidence.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_atlas_evidence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'causal_algorithms_atlas.evidence'`

- [ ] **Step 3: Implement `causal_algorithms_atlas/evidence.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class EvidenceError(ValueError):
    """Raised when an evidence record is malformed or dangles a reference."""


@dataclass(frozen=True)
class EvidenceRecord:
    algorithm_id: str
    dataset: str
    source: str
    metrics: dict[str, float]
    notes: str
    source_path: str = ""


def load_evidence_records(directory: str | Path) -> list[EvidenceRecord]:
    directory = Path(directory)
    records: list[EvidenceRecord] = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            records.append(
                EvidenceRecord(
                    algorithm_id=raw["algorithm_id"],
                    dataset=raw["dataset"],
                    source=raw["source"],
                    metrics=dict(raw.get("metrics") or {}),
                    notes=str(raw.get("notes") or ""),
                    source_path=str(path),
                )
            )
        except (KeyError, yaml.YAMLError) as exc:
            raise EvidenceError(f"{path}: registro de evidencia invalido ({exc}).") from exc
    return records


def validate_evidence_algorithm_ids(
    records: list[EvidenceRecord], known_ids: frozenset[str]
) -> None:
    unknown = [r for r in records if r.algorithm_id not in known_ids]
    if unknown:
        bad = [(r.algorithm_id, r.source_path) for r in unknown]
        raise EvidenceError(f"Registros de evidencia com algorithm_id desconhecido: {bad}.")
```

- [ ] **Step 4: Run tests to verify they pass so far**

Run: `python -m pytest tests/test_atlas_evidence.py -v`
Expected: `LoadEvidenceRecordsTests` and `ValidateEvidenceAlgorithmIdsTests` PASS;
`RealEvidenceContentTests` still FAILS (no `evidence/*.yaml` content yet) — that's expected at
this point.

- [ ] **Step 5: Create the 9 evidence YAML files**

`causal_algorithms_atlas/evidence/pcmci__toy_a_linear.yaml`:

```yaml
algorithm_id: pcmci
dataset: toy_a_linear
source: "datasets/synthetic_causal/toy_a_linear.csv + toy_a_linear_gt.csv"
metrics:
  precision: 0.75
  recall: 1.0
  f1_score: 0.8571428571428571
notes: >
  Medido no benchmark sintetico do projeto (ground truth conhecido via
  toy_a_linear_gt.csv), nao e resultado do paper original. Fonte bruta:
  .local/results/toy_synthetic_validation/metrics.csv (nao versionado).
```

`causal_algorithms_atlas/evidence/pcmci__toy_b_nonlinear.yaml`:

```yaml
algorithm_id: pcmci
dataset: toy_b_nonlinear
source: "datasets/synthetic_causal/toy_b_nonlinear.csv + toy_b_nonlinear_gt.csv"
metrics:
  precision: 1.0
  recall: 1.0
  f1_score: 1.0
notes: >
  Medido no benchmark sintetico do projeto. Fonte bruta:
  .local/results/toy_synthetic_validation/metrics.csv (nao versionado).
```

`causal_algorithms_atlas/evidence/classical_granger__toy_a_linear.yaml`:

```yaml
algorithm_id: classical_granger
dataset: toy_a_linear
source: "datasets/synthetic_causal/toy_a_linear.csv + toy_a_linear_gt.csv"
metrics:
  precision: 0.75
  recall: 1.0
  f1_score: 0.8571428571428571
notes: >
  Medido no benchmark sintetico do projeto. Fonte bruta:
  .local/results/toy_synthetic_validation/metrics.csv (nao versionado).
```

`causal_algorithms_atlas/evidence/ges__toy_a_linear.yaml`:

```yaml
algorithm_id: ges
dataset: toy_a_linear
source: "datasets/synthetic_causal/toy_a_linear.csv + toy_a_linear_gt.csv"
metrics:
  precision: 1.0
  recall: 1.0
  f1_score: 1.0
notes: >
  Medido no benchmark sintetico do projeto. Fonte bruta:
  .local/results/toy_synthetic_validation/metrics.csv (nao versionado).
```

`causal_algorithms_atlas/evidence/fci__toy_a_linear.yaml`:

```yaml
algorithm_id: fci
dataset: toy_a_linear
source: "datasets/synthetic_causal/toy_a_linear.csv + toy_a_linear_gt.csv"
metrics:
  precision: 1.0
  recall: 1.0
  f1_score: 1.0
notes: >
  Medido no benchmark sintetico do projeto. Fonte bruta:
  .local/results/toy_synthetic_validation/metrics.csv (nao versionado).
```

`causal_algorithms_atlas/evidence/dynotears__toy_a_linear.yaml`:

```yaml
algorithm_id: dynotears
dataset: toy_a_linear
source: "datasets/synthetic_causal/toy_a_linear.csv + toy_a_linear_gt.csv"
metrics:
  precision: 1.0
  recall: 1.0
  f1_score: 1.0
notes: >
  Medido no benchmark sintetico do projeto. Fonte bruta:
  .local/results/toy_synthetic_validation/metrics.csv (nao versionado).
```

`causal_algorithms_atlas/evidence/lpcmci__toy_a_linear.yaml`:

```yaml
algorithm_id: lpcmci
dataset: toy_a_linear
source: "datasets/synthetic_causal/toy_a_linear.csv + toy_a_linear_gt.csv"
metrics:
  precision: 0.0
  recall: 0.0
  f1_score: 0.0
notes: >
  Medido no benchmark sintetico do projeto: LPCMCI nao retornou nenhuma aresta
  direcionada com confianca suficiente neste dataset (marcas ficaram ambiguas). Fonte
  bruta: .local/results/toy_synthetic_validation/metrics.csv (nao versionado).
```

`causal_algorithms_atlas/evidence/neural_granger_cmlp__toy_a_linear.yaml`:

```yaml
algorithm_id: neural_granger_cmlp
dataset: toy_a_linear
source: "datasets/synthetic_causal/toy_a_linear.csv + toy_a_linear_gt.csv"
metrics:
  precision: 0.3
  recall: 1.0
  f1_score: 0.4615384615384615
notes: >
  Medido no benchmark sintetico do projeto: alta taxa de falsos positivos neste
  dataset pequeno. Fonte bruta:
  .local/results/toy_synthetic_validation/metrics.csv (nao versionado).
```

`causal_algorithms_atlas/evidence/var_lingam__toy_a_linear.yaml`:

```yaml
algorithm_id: var_lingam
dataset: toy_a_linear
source: "datasets/synthetic_causal/toy_a_linear.csv + toy_a_linear_gt.csv"
metrics:
  precision: 1.0
  recall: 1.0
  f1_score: 1.0
notes: >
  Medido no benchmark sintetico do projeto. Fonte bruta:
  .local/results/toy_synthetic_validation/metrics.csv (nao versionado).
```

- [ ] **Step 6: Run the full evidence test file**

Run: `python -m pytest tests/test_atlas_evidence.py -v`
Expected: PASS (4 tests, including `RealEvidenceContentTests`)

- [ ] **Step 7: Commit**

```bash
git add causal_algorithms_atlas/evidence.py causal_algorithms_atlas/evidence tests/test_atlas_evidence.py
git commit -m "feat: adicionar camada de evidencia empirica do atlas"
```

---

## Task 7: Additional verified literature algorithms

**Files:**
- Create: `causal_algorithms_atlas/algorithms/<id>.md` (one per algorithm accepted below)
- Modify: `causal_algorithms_atlas/references.yaml` (append new reference keys)

**Interfaces:**
- Consumes: same as Task 4 (`AlgorithmCard` schema, `REQUIRED_SECTIONS`).
- Produces: no new public interface; extends the existing `algorithms/` and `references.yaml`
  content the tests from Tasks 2-6 already cover.

This task is research-gated, not fixed content: for each candidate below, verify the primary
reference (DOI/arXiv/venue) with `WebSearch`/`WebFetch` before writing the card. Only write a
card for a candidate whose reference is confirmed. Do not lower the bar to hit a count.

Candidates to attempt, in order:

1. **PC / PC-stable** — Spirtes & Glymour (1991) original PC; Colombo & Maathuis (2014) for
   PC-stable. `family: constraint-based`, `temporal_handling: none` (needs windowed unrolling
   like GES/FCI to apply to time series — document this in "Quando evitar").
2. **PCMCI+** — Runge (2020), extends PCMCI to include contemporaneous links.
   `family: constraint-based`, `temporal_handling: native`, `handles_contemporaneous_effects: true`.
3. **Transfer Entropy** — Schreiber (2000), Physical Review Letters. `family:
   information-theoretic`, `temporal_handling: native`, `handles_nonlinearity: true` (it's a
   general information-theoretic dependency measure, not restricted to linear relations).
4. **Convergent Cross Mapping (CCM)** — Sugihara et al. (2012), Science. `family:
   information-theoretic` or a dedicated state-space category if the schema's
   `AlgorithmFamily` enum needs extending (if so, add the new member to `schema.py` in this
   task and update `test_atlas_schema.py`'s enum coverage expectations accordingly).
5. **TiMINo** — Peters, Janzing, Schölkopf (2013), NeurIPS. `family: functional-causal-model`.
6. **NOTEARS** — Zheng et al. (2018), NeurIPS. `family: continuous-optimization`,
   `temporal_handling: none` (it's the atemporal ancestor of DYNOTEARS — document the
   relationship in "Relação com outros métodos").
7. **SVAR-FCI / tsFCI** — Entner & Hoyer (2010). `family: constraint-based`, `temporal_handling:
   native`, `handles_latent_confounders: true`.
8. **TCDF** — Nauta, Bucur, Seifert (2019), Machine Learning and Knowledge Extraction.
   `family: functional-causal-model` (attention-based temporal CNN), `handles_nonlinearity: true`.

- [ ] **Step 1: For each candidate, search for and confirm the primary reference**

Use `WebSearch` with a query like `"<algorithm name>" "<first author>" paper doi` or
`arxiv "<algorithm name>" time series causal discovery`. Confirm a DOI, arXiv ID, or a
publisher page (NeurIPS proceedings, PMLR, Science, etc.) actually describing that algorithm —
not a survey that merely mentions it in passing. If confirmation fails after a reasonable
search, skip that candidate (do not write a card) and note the skip in the task's commit
message or a one-line comment in the plan's progress notes.

- [ ] **Step 2: For each confirmed candidate, append its reference to `references.yaml`**

Follow the same shape as Task 4's entries (`authors`, `year`, `title`, `venue`,
`doi` or `url`, `verified: true`).

- [ ] **Step 3: For each confirmed candidate, write `causal_algorithms_atlas/algorithms/<id>.md`**

Follow the exact frontmatter shape and the 5 required section headings from Task 4's cards.
Set `implemented_in_framework: false` and `framework_method_name: null` for all of these (none
are currently implemented in `causal_discovery/`). Set `verification: verified` only if the
paper's method section (not just the abstract) was actually read/confirmed to back every
`assumptions`/`handles_*` field written; otherwise use `verification: draft` and leave
`verified_by: null` for that card.

- [ ] **Step 4: Run the full atlas test suite after each card is added**

Run: `python -m pytest tests/test_atlas_schema.py tests/test_atlas_loader.py tests/test_atlas_validate.py tests/test_atlas_content.py tests/test_atlas_export.py -v`
Expected: PASS after every addition — fix immediately if a new card breaks `references`
resolution or introduces an assumption id outside `KNOWN_ASSUMPTIONS` (extend
`KNOWN_ASSUMPTIONS` in `schema.py`, with a matching test update, if a genuinely new controlled
assumption is needed — e.g. CCM's premise of a deterministic, low-dimensional attractor isn't
covered by the existing 8 ids).

- [ ] **Step 5: Commit once, after all candidates in this task have been attempted**

```bash
git add causal_algorithms_atlas/algorithms causal_algorithms_atlas/references.yaml causal_algorithms_atlas/schema.py tests/test_atlas_schema.py
git commit -m "feat: adicionar algoritmos adicionais verificados da literatura ao atlas"
```

---

## Task 8: EDA over the atlas (`eda.py`)

**Files:**
- Create: `causal_algorithms_atlas/eda.py`
- Modify: `.gitignore` (add `causal_algorithms_atlas/eda_output/`)
- Test: `tests/test_atlas_eda.py`

**Interfaces:**
- Consumes: `causal_algorithms_atlas.loader.load_algorithm_cards`
- Produces: `cards_to_dataframe(cards: dict[str, AlgorithmCard]) -> pandas.DataFrame` (one row
  per algorithm, columns: `id`, `name`, `family`, `temporal_handling`, `output_type`,
  `handles_latent_confounders`, `handles_nonlinearity`, `handles_contemporaneous_effects`,
  `verification`, `n_assumptions`); `family_counts_figure(df: pandas.DataFrame) ->
  plotly.graph_objects.Figure`; `assumption_coverage_figure(cards: dict[str, AlgorithmCard]) ->
  plotly.graph_objects.Figure` (boolean heatmap: algorithms x `KNOWN_ASSUMPTIONS`); a
  `if __name__ == "__main__":` block that loads the real atlas, builds both figures, and writes
  them as HTML to `causal_algorithms_atlas/eda_output/`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_atlas_eda.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_atlas_eda.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'causal_algorithms_atlas.eda'`

- [ ] **Step 3: Implement `causal_algorithms_atlas/eda.py`**

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from causal_algorithms_atlas.loader import load_algorithm_cards
from causal_algorithms_atlas.schema import KNOWN_ASSUMPTIONS, AlgorithmCard

_ATLAS_ROOT = Path(__file__).resolve().parent
_OUTPUT_DIR = _ATLAS_ROOT / "eda_output"


def cards_to_dataframe(cards: dict[str, AlgorithmCard]) -> pd.DataFrame:
    rows = [
        {
            "id": card.id,
            "name": card.name,
            "family": card.family.value,
            "temporal_handling": card.temporal_handling.value,
            "output_type": card.output_type.value,
            "handles_latent_confounders": card.handles_latent_confounders,
            "handles_nonlinearity": card.handles_nonlinearity,
            "handles_contemporaneous_effects": card.handles_contemporaneous_effects,
            "verification": card.verification.value,
            "n_assumptions": len(card.assumptions),
        }
        for card in cards.values()
    ]
    return pd.DataFrame(rows)


def family_counts_figure(df: pd.DataFrame) -> go.Figure:
    counts = df["family"].value_counts().reset_index()
    counts.columns = ["family", "count"]
    return px.bar(counts, x="family", y="count", title="Algoritmos por familia")


def assumption_coverage_figure(cards: dict[str, AlgorithmCard]) -> go.Figure:
    known = sorted(KNOWN_ASSUMPTIONS)
    ids = sorted(cards)
    z = [
        [1 if any(a.id == assumption for a in cards[algo_id].assumptions) else 0 for assumption in known]
        for algo_id in ids
    ]
    return go.Figure(
        data=go.Heatmap(z=z, x=known, y=ids, colorscale="Blues", showscale=False),
    ).update_layout(title="Cobertura de premissas por algoritmo")


if __name__ == "__main__":
    cards = load_algorithm_cards(_ATLAS_ROOT / "algorithms")
    df = cards_to_dataframe(cards)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    family_counts_figure(df).write_html(_OUTPUT_DIR / "family_counts.html")
    assumption_coverage_figure(cards).write_html(_OUTPUT_DIR / "assumption_coverage.html")
    print(f"EDA salva em {_OUTPUT_DIR}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_atlas_eda.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Generate the actual EDA output once, manually**

Run: `python -m causal_algorithms_atlas.eda`
Expected: prints `EDA salva em ...` and creates `causal_algorithms_atlas/eda_output/family_counts.html`
and `assumption_coverage.html`. Open one in a browser to sanity-check it renders.

- [ ] **Step 6: Add the generated-output directory to `.gitignore`**

```gitignore
# Causal algorithms atlas generated output
causal_algorithms_atlas/eda_output/
```

- [ ] **Step 7: Commit**

```bash
git add causal_algorithms_atlas/eda.py tests/test_atlas_eda.py .gitignore
git commit -m "feat: adicionar EDA sobre a base do atlas de algoritmos"
```

---

## Task 9: Debug RAG chat backed by local Llama (`rag_chat.py`)

**Files:**
- Create: `causal_algorithms_atlas/rag_chat.py`
- Test: `tests/test_atlas_rag_chat.py`

**Interfaces:**
- Consumes: `causal_algorithms_atlas.export.{Chunk, cards_to_chunks}`,
  `causal_algorithms_atlas.loader.load_algorithm_cards`
- Produces: `RagChatError(RuntimeError)`; `build_retriever(chunks: list[Chunk]) ->
  TfidfRetriever` (a small class wrapping a fitted `TfidfVectorizer` + the chunk list);
  `TfidfRetriever.top_k(query: str, k: int = 4) -> list[tuple[Chunk, float]]` (chunk + cosine
  similarity score, descending); `build_prompt(query: str, retrieved: list[tuple[Chunk, float]])
  -> str`; `call_ollama(prompt: str, *, model: str = "llama3.1:8b", base_url: str =
  "http://localhost:11434") -> str` (POSTs to `{base_url}/api/generate` with
  `{"model": model, "prompt": prompt, "stream": false}` via `urllib.request`, returns the
  `"response"` field; raises `RagChatError` wrapping any `URLError`/`HTTPError` with a message
  telling the user to run `ollama serve` and `ollama pull llama3.1:8b`); a `main(argv:
  list[str] | None = None) -> None` CLI entry point that takes the query as `argv[0]`, loads all
  cards, retrieves, prints retrieved chunks with scores, calls `call_ollama`, and prints the
  answer.

- [ ] **Step 1: Write the failing tests (retrieval and prompt-building only — no live Ollama call in automated tests)**

```python
# tests/test_atlas_rag_chat.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_atlas_rag_chat.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'causal_algorithms_atlas.rag_chat'`

- [ ] **Step 3: Implement `causal_algorithms_atlas/rag_chat.py`**

```python
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
_DEFAULT_MODEL = "llama3.1:8b"
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
    prompt: str, *, model: str = _DEFAULT_MODEL, base_url: str = _DEFAULT_BASE_URL
) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_atlas_rag_chat.py -v`
Expected: PASS (5 tests) — none of these require a live Ollama server (the error-path test
mocks `urlopen`; the retrieval/prompt tests are pure TF-IDF/string logic).

- [ ] **Step 5: Manual smoke test with a live Ollama server**

Run: `ollama pull llama3.1:8b` (one-time download, several GB)
Run (separate terminal or background): `ollama serve` (if not already running as a service —
check with `ollama list` first; on Windows the Ollama app usually keeps the server running in
the background already)
Run: `python -m causal_algorithms_atlas.rag_chat "Qual metodo do framework tolera confundidor latente?"`
Expected: prints retrieved chunks (should surface `lpcmci` and/or `fci`) followed by a
generated answer. This is the debug query the user asked for — read the output and confirm the
retrieval makes sense; this step is exploratory, not pass/fail.

- [ ] **Step 6: Commit**

```bash
git add causal_algorithms_atlas/rag_chat.py tests/test_atlas_rag_chat.py
git commit -m "feat: adicionar chat RAG de debug com Llama local via Ollama"
```

---

## Task 10: Dependency, documentation, and final validation

**Files:**
- Modify: `requirements.txt`
- Modify: `README.md`

**Interfaces:** none (documentation and dependency-manifest only).

- [ ] **Step 1: Add `pyyaml` to `requirements.txt`**

```text
pandas
numpy
scipy
statsmodels
scikit-learn
tigramite
lingam
causal-learn
torch
plotly
ipywidgets
nbformat
pyyaml
```

- [ ] **Step 2: Install it in the active environment**

Run: `pip install pyyaml`
Expected: installs successfully (if already present transitively, this is a no-op).

- [ ] **Step 3: Add a short section to `README.md` describing the new package**

Insert after the existing "Adicionando um algoritmo" section:

```markdown
## Base de conhecimento de algoritmos (causal_algorithms_atlas)

O pacote `causal_algorithms_atlas/` mantém uma base de conhecimento verificada sobre
algoritmos de causal discovery em séries temporais, independente do pipeline de
execução em `causal_discovery/`. Cada algoritmo tem uma ficha em
`causal_algorithms_atlas/algorithms/<id>.md` (frontmatter YAML com premissas,
requisitos de dados e referências verificadas + prosa explicativa em português). A
camada `causal_algorithms_atlas/evidence/` guarda resultados empíricos medidos neste
projeto, separados do conteúdo de literatura.

Comandos úteis:

- `python -m pytest tests/test_atlas_content.py -v` — confere que todo método
  registrado no framework tem ficha `verified` no atlas.
- `python -m causal_algorithms_atlas.eda` — gera gráficos (HTML) descrevendo a
  cobertura da base por família de algoritmo e por premissa.
- `python -m causal_algorithms_atlas.rag_chat "pergunta"` — consulta de debug via RAG
  (TF-IDF + Ollama local, modelo `llama3.1:8b`) para inspecionar manualmente a
  recuperação antes de qualquer uso mais sério.
```

- [ ] **Step 4: Full validation pass**

Run: `python -m compileall causal_discovery causal_algorithms_atlas`
Expected: no errors

Run: `python -m pytest tests -q`
Expected: all tests pass (existing `causal_discovery` suite unaffected, all new
`test_atlas_*.py` files pass)

- [ ] **Step 5: Commit**

```bash
git add requirements.txt README.md
git commit -m "docs: documentar causal_algorithms_atlas e adicionar pyyaml as dependencias"
```
