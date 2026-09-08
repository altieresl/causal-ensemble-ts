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
