from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def normalize_expert_knowledge(knowledge: pd.DataFrame | list[dict] | None) -> pd.DataFrame:
    if knowledge is None:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "lag",
                "relation",
                "confidence",
                "constraint",
                "prior_probability",
            ]
        )

    frame = pd.DataFrame(knowledge).copy()
    if frame.empty:
        return normalize_expert_knowledge(None)

    defaults = {
        "lag": np.nan,
        "relation": "weak",
        "confidence": 0.5,
        "constraint": "soft",
        "prior_probability": np.nan,
    }
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value

    frame["relation"] = frame["relation"].astype(str).str.lower().str.strip()
    frame["constraint"] = frame["constraint"].astype(str).str.lower().str.strip()
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    frame["prior_probability"] = pd.to_numeric(frame["prior_probability"], errors="coerce")

    valid_relation = frame["relation"].isin({"strong", "weak", "none"})
    valid_constraint = frame["constraint"].isin({"soft", "hard"})
    frame = frame[valid_relation & valid_constraint].copy()

    if frame.empty:
        return normalize_expert_knowledge(None)

    return frame[["source", "target", "lag", "relation", "confidence", "constraint", "prior_probability"]]


def extract_method_weights(method_weights: Mapping[str, float] | None = None) -> dict[str, float]:
    if not method_weights:
        return {}
    return {str(name): max(float(weight), 0.0) for name, weight in method_weights.items()}


def _matches_rule(edge_row: pd.Series, rule_row: pd.Series) -> bool:
    if str(edge_row.get("source")) != str(rule_row.get("source")):
        return False
    if str(edge_row.get("target")) != str(rule_row.get("target")):
        return False

    rule_lag = rule_row.get("lag")
    if pd.isna(rule_lag):
        return True
    try:
        return int(edge_row.get("lag")) == int(rule_lag)
    except Exception:
        return False


def _apply_single_rule(edge_probability: float, rule_row: pd.Series) -> tuple[float, str]:
    relation = str(rule_row["relation"])
    constraint = str(rule_row["constraint"])
    confidence = float(rule_row["confidence"])
    prior_probability = rule_row.get("prior_probability", np.nan)

    if relation == "none":
        if constraint == "hard":
            return 0.0, "hard_forbidden"
        return max(0.0, edge_probability * (1.0 - 0.9 * confidence)), "soft_forbidden"

    if relation == "strong":
        target_prior = float(prior_probability) if pd.notna(prior_probability) else 0.9
    else:
        target_prior = float(prior_probability) if pd.notna(prior_probability) else 0.35

    target_prior = float(np.clip(target_prior, 0.0, 1.0))

    # Combinamos a probabilidade inferida pelos dados com o conhecimento especialista.
    blend = confidence if constraint == "soft" else min(1.0, 0.5 + 0.5 * confidence)
    updated = (1.0 - blend) * edge_probability + blend * target_prior
    return float(np.clip(updated, 0.0, 1.0)), f"{constraint}_{relation}"


def apply_expert_knowledge_to_summary(
    summary: pd.DataFrame,
    knowledge: pd.DataFrame | list[dict] | None,
    *,
    hard_filter: bool = True,
) -> pd.DataFrame:
    if summary is None or summary.empty:
        return summary

    expert = normalize_expert_knowledge(knowledge)
    if expert.empty:
        output = summary.copy()
        if "expert_adjustment" not in output.columns:
            output["expert_adjustment"] = "none"
        return output

    output = summary.copy()
    if "expert_adjustment" not in output.columns:
        output["expert_adjustment"] = "none"
    if "expert_confidence" not in output.columns:
        output["expert_confidence"] = 0.0

    for idx, edge_row in output.iterrows():
        matched_rules = expert[expert.apply(lambda rule: _matches_rule(edge_row, rule), axis=1)]
        if matched_rules.empty:
            continue

        current_probability = float(np.clip(edge_row.get("edge_probability", 0.0), 0.0, 1.0))
        applied_labels: list[str] = []
        max_confidence = float(edge_row.get("expert_confidence", 0.0))

        for _, rule in matched_rules.iterrows():
            current_probability, label = _apply_single_rule(current_probability, rule)
            applied_labels.append(label)
            max_confidence = max(max_confidence, float(rule["confidence"]))

        output.at[idx, "edge_probability"] = current_probability
        output.at[idx, "uncertainty"] = 1.0 - current_probability
        output.at[idx, "expert_adjustment"] = "|".join(applied_labels)
        output.at[idx, "expert_confidence"] = float(max_confidence)

        if "posterior_probability" in output.columns:
            output.at[idx, "posterior_probability"] = current_probability

    if hard_filter:
        forbidden = output["expert_adjustment"].astype(str).str.contains("hard_forbidden", regex=False)
        output = output[~forbidden].copy()

    return output.reset_index(drop=True)
