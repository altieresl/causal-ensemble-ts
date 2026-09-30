"""Avaliacoes estruturais reutilizaveis (portadas de Series_Temporais.ipynb).

Funcoes puras sobre DataFrames: comparacao ensemble x algoritmos avulsos, evidencia de painel
multi-trajetoria e avaliacao de uma estrategia contra o grafo verdadeiro. O grafo verdadeiro
so entra aqui, depois da selecao (nunca influencia quais metodos formam o ensemble).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

EMPTY_EDGES = ["source", "target", "lag"]


def drop_self_links(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=EMPTY_EDGES)
    return frame.loc[frame["source"].ne(frame["target"])].reset_index(drop=True)


def ranking_evidence(frame: pd.DataFrame) -> str:
    """Usa ``1 - p_value`` quando o metodo reporta significancia; senao, a magnitude do score."""
    if "p_value" in frame.columns:
        p_values = pd.to_numeric(frame["p_value"], errors="coerce")
        if np.isfinite(p_values).any():
            return "one_minus_p_value"
    return "absolute_score"


def undirected_pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    if len(frame) == 0:
        return set()
    return {
        tuple(sorted((str(s), str(t))))
        for s, t in zip(frame["source"], frame["target"])
        if s != t
    }


def selected_frame(summary: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Arestas detectadas pelo ensemble: ``ensemble_selected`` ou, na falta, o limiar de probabilidade."""
    if "ensemble_selected" in summary.columns:
        return summary.loc[summary["ensemble_selected"].fillna(False).astype(bool)].reset_index(drop=True)
    return summary.loc[pd.to_numeric(summary["edge_probability"], errors="coerce") >= threshold].reset_index(drop=True)


def evaluated_pair_set(relations: Sequence[tuple[str, str]]) -> set[tuple[str, str]]:
    return {tuple(sorted((s, t))) for s, t in relations if s != t}


def _filter_pairs(pair_scores: pd.DataFrame, pairs: set[tuple[str, str]]) -> pd.DataFrame:
    keep = [tuple(sorted((s, t))) in pairs for s, t in zip(pair_scores["source"], pair_scores["target"])]
    return pair_scores.loc[keep].reset_index(drop=True)


def base_method_outputs(selection: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Saidas individuais ja calculadas pela selecao robusta (nenhum metodo e reexecutado)."""
    outputs = selection.get("precomputed_outputs") or {}
    if outputs:
        return dict(outputs)
    outputs = {}
    for evaluation in selection["all_evaluations"].values():
        for name, output in evaluation["outputs"].items():
            outputs.setdefault(name, output)
    return outputs


def evaluate_strategy(
    frame: pd.DataFrame,
    *,
    ground_truth: pd.DataFrame,
    nodes: Sequence[str],
    relations: Sequence[tuple[str, str]],
    binary_frame: pd.DataFrame | None = None,
    evidence: str,
) -> dict[str, Any]:
    """Metricas binarias (esqueleto nao direcionado) + ranking sem limiar de uma estrategia."""
    from causal_discovery import (
        build_complete_undirected_pair_scores,
        compute_ranked_undirected_skeleton_metrics,
        compute_undirected_skeleton_metrics,
    )

    binary = compute_undirected_skeleton_metrics(
        frame if binary_frame is None else binary_frame,
        ground_truth,
        prob_threshold=0.0,
        nodes=list(nodes),
        evaluated_relations=list(relations),
    )
    pair_scores = build_complete_undirected_pair_scores(frame, list(nodes), evidence=evidence)
    pair_scores = _filter_pairs(pair_scores, evaluated_pair_set(relations))
    try:
        ranked = compute_ranked_undirected_skeleton_metrics(pair_scores, ground_truth)
    except ValueError:
        ranked = {"average_precision": float("nan"), "roc_auc": float("nan")}
    return {"binary": binary, "ranked": ranked}


def compare_strategies(
    selection: dict[str, Any],
    summary: pd.DataFrame,
    *,
    ground_truth: pd.DataFrame,
    nodes: Sequence[str],
    relations: Sequence[tuple[str, str]],
    threshold: float,
) -> dict[str, Any]:
    """Ensemble (ENSEMBLE_AUTO) versus cada algoritmo avulso, com a mesma avaliacao estrutural."""
    from causal_discovery import compute_undirected_skeleton_metrics

    pairs_evaluated = evaluated_pair_set(relations)
    individual = {name: drop_self_links(frame) for name, frame in sorted(base_method_outputs(selection).items())}
    ensemble_summary = drop_self_links(summary)
    ensemble_detected = selected_frame(ensemble_summary, threshold)

    frames = dict(individual)
    frames["ENSEMBLE_AUTO"] = ensemble_summary
    evidence = {name: ranking_evidence(frame) for name, frame in individual.items()}
    evidence["ENSEMBLE_AUTO"] = "ensemble_score"

    has_truth = not ground_truth.empty
    rows: list[dict[str, Any]] = []
    detected: dict[str, set[tuple[str, str]]] = {}
    for strategy, frame in frames.items():
        binary_frame = ensemble_detected if strategy == "ENSEMBLE_AUTO" else frame
        pairs = undirected_pairs(binary_frame) & pairs_evaluated
        detected[strategy] = pairs
        row: dict[str, Any] = {
            "strategy": strategy,
            "returned_edges": int(len(binary_frame)),
            "detected_pairs": len(pairs),
        }
        if has_truth:
            outcome = evaluate_strategy(
                frame, ground_truth=ground_truth, nodes=nodes, relations=relations,
                binary_frame=binary_frame, evidence=evidence[strategy],
            )
            binary, ranked = outcome["binary"], outcome["ranked"]
            row.update(
                precision=binary["precision"], recall=binary["recall"], f1_score=binary["f1_score"],
                f1_minus_baseline=binary["f1_score"] - binary["all_pairs_baseline_f1"],
                structural_hamming_distance=binary["structural_hamming_distance"],
                true_positives=binary["true_positives"], false_positives=binary["false_positives"],
                false_negatives=binary["false_negatives"],
                average_precision=ranked["average_precision"], roc_auc=ranked["roc_auc"],
                false_positive_pairs=sorted(binary["false_positive_pairs"]),
                false_negative_pairs=sorted(binary["false_negative_pairs"]),
            )
        rows.append(row)
    if has_truth:
        rows.sort(key=lambda r: r["f1_score"], reverse=True)

    ensemble_pairs = detected["ENSEMBLE_AUTO"]
    overlap = []
    for strategy, pairs in detected.items():
        if strategy == "ENSEMBLE_AUTO":
            continue
        union = pairs | ensemble_pairs
        overlap.append(
            {
                "strategy": strategy,
                "shared_pairs": len(pairs & ensemble_pairs),
                "only_method": len(pairs - ensemble_pairs),
                "only_ensemble": len(ensemble_pairs - pairs),
                "jaccard": (len(pairs & ensemble_pairs) / len(union)) if union else None,
            }
        )

    reference = None
    if has_truth:
        ref = compute_undirected_skeleton_metrics(
            pd.DataFrame(columns=EMPTY_EDGES), ground_truth, nodes=list(nodes), evaluated_relations=list(relations)
        )
        reference = {
            "ground_truth_pairs": ref["ground_truth_pairs"],
            "ground_truth_prevalence": ref["ground_truth_prevalence"],
            "all_pairs_baseline_f1": ref["all_pairs_baseline_f1"],
        }
    return {
        "rows": rows,
        "overlap": overlap,
        "evaluated_pairs": len(pairs_evaluated),
        "reference": reference,
    }


def panel_evidence(
    dataset: Any,
    *,
    relations: Sequence[tuple[str, str]],
    max_lag: int,
    ground_truth: pd.DataFrame,
) -> dict[str, Any] | None:
    """Ranking sem limiar com PCMCI sobre todas as trajetorias independentes (nao altera o ensemble)."""
    from causal_discovery import compute_ranked_undirected_skeleton_metrics, run_pcmci_multiple_trajectories

    if dataset.trajectory_count <= 1 or ground_truth.empty:
        return None
    scores = run_pcmci_multiple_trajectories(
        dataset.observed_trajectories(), dataset.available_columns, max_lag=max_lag, standardize=True
    )
    scores = _filter_pairs(scores, evaluated_pair_set(relations))
    metrics: dict[str, Any] | None
    error: str | None = None
    try:
        metrics = compute_ranked_undirected_skeleton_metrics(scores, ground_truth)
    except ValueError as exc:
        metrics, error = None, str(exc)
    return {
        "trajectory_count": dataset.trajectory_count,
        "max_lag": max_lag,
        "context_nodes": len(dataset.available_columns),
        "ranking_metrics": metrics,
        "error": error,
        "top_pairs": scores.sort_values("score", ascending=False).head(20).to_dict(orient="records"),
    }
