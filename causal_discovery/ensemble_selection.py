from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .ensemble import (
    _deduplicate_method_edges,
    _label_method_output,
    run_method_suite,
    summarize_ensemble,
    summarize_probabilistic_ensemble,
)
from .expert_knowledge import apply_expert_knowledge_to_summary, extract_method_weights

MetricScoreFn = Callable[[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame], float]


def _run_method_suite_fast(
    data: pd.DataFrame,
    methods: Mapping[str, Callable[..., pd.DataFrame]],
    *,
    method_kwargs: Mapping[str, dict] | None = None,
    parallel_jobs: int = 1,
) -> dict[str, pd.DataFrame]:
    method_kwargs = method_kwargs or {}
    if parallel_jobs <= 1 or len(methods) <= 1:
        return run_method_suite(data, methods, method_kwargs=method_kwargs)

    outputs: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(parallel_jobs))) as executor:
        futures = {
            executor.submit(method, data, **method_kwargs.get(name, {})): name
            for name, method in methods.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            outputs[name] = _label_method_output(
                name,
                future.result(),
                data_columns=data.columns,
            )

    return outputs


def _empty_stability_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source",
            "target",
            "lag",
            "selected_count",
            "selection_frequency",
            "appearance_frequency",
            "mean_edge_probability",
            "mean_confidence",
            "stability_selected",
        ]
    )


def _to_edge_key(source: object, target: object, lag: object) -> tuple[str, str, int] | None:
    if pd.isna(source) or pd.isna(target) or pd.isna(lag):
        return None
    try:
        lag_int = int(lag)
    except Exception:
        return None
    return str(source), str(target), lag_int


def _extract_edge_set(frame: pd.DataFrame) -> set[tuple[str, str, int]]:
    if frame is None or frame.empty:
        return set()
    required = {"source", "target", "lag"}
    if not required.issubset(frame.columns):
        return set()

    keys: set[tuple[str, str, int]] = set()
    for _, row in frame[["source", "target", "lag"]].iterrows():
        key = _to_edge_key(row["source"], row["target"], row["lag"])
        if key is not None:
            keys.add(key)
    return keys


def compute_method_consistency(results_by_method: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    methods = list(results_by_method.keys())
    if not methods:
        return pd.DataFrame()

    edge_sets = {name: _extract_edge_set(frame) for name, frame in results_by_method.items()}
    matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)

    for left in methods:
        for right in methods:
            set_left = edge_sets[left]
            set_right = edge_sets[right]
            union = set_left | set_right
            if not union:
                score = 1.0
            else:
                score = len(set_left & set_right) / len(union)
            matrix.at[left, right] = float(score)

    return matrix


def _jaccard_similarity(
    left: set[tuple[str, str, int]],
    right: set[tuple[str, str, int]],
) -> float:
    union = left | right
    if not union:
        return float("nan")
    return float(len(left & right) / len(union))


def estimate_adaptive_method_weights(
    results_by_method: Mapping[str, pd.DataFrame],
    bootstrap_outputs: list[Mapping[str, pd.DataFrame]] | None = None,
    *,
    base_weights: Mapping[str, float] | None = None,
    stability_power: float = 1.0,
    diversity_bonus: float = 0.15,
    density_penalty: float = 0.5,
    minimum_weight: float = 0.05,
    redundancy_penalty: float = 0.0,
) -> pd.DataFrame:
    """Estima pesos sem usar o ground truth.

    A confiabilidade vem da repetibilidade das arestas sob reamostragem. Um
    bonus moderado preserva informacao de metodos nao redundantes, enquanto a
    penalizacao de densidade impede que um grafo muito denso domine apenas por
    emitir mais arestas.
    """
    names = list(results_by_method)
    columns = [
        "method",
        "prior_weight",
        "bootstrap_stability",
        "diversity",
        "mean_edge_count",
        "relative_density",
        "density_factor",
        "redundancy",
        "redundancy_factor",
        "adaptive_weight",
    ]
    if not names:
        return pd.DataFrame(columns=columns)

    stability_power = max(float(stability_power), 0.0)
    diversity_bonus = max(float(diversity_bonus), 0.0)
    density_penalty = max(float(density_penalty), 0.0)
    minimum_weight = float(np.clip(minimum_weight, 1e-6, 1.0))
    redundancy_penalty = max(float(redundancy_penalty), 0.0)
    base_weights = base_weights or {}
    bootstrap_outputs = list(bootstrap_outputs or [])

    edge_sets_by_method: dict[str, list[set[tuple[str, str, int]]]] = {}
    mean_edge_counts: dict[str, float] = {}
    for name in names:
        samples = [_extract_edge_set(results_by_method.get(name, pd.DataFrame()))]
        samples.extend(
            _extract_edge_set(iteration.get(name, pd.DataFrame()))
            for iteration in bootstrap_outputs
        )
        edge_sets_by_method[name] = samples
        mean_edge_counts[name] = float(np.mean([len(edges) for edges in samples]))

    positive_counts = [count for count in mean_edge_counts.values() if count > 0.0]
    reference_count = float(np.median(positive_counts)) if positive_counts else 1.0
    reference_count = max(reference_count, 1e-12)

    rows: list[dict[str, float | str]] = []
    for name in names:
        samples = edge_sets_by_method[name]
        repeatabilities = [
            similarity
            for left_index, left in enumerate(samples)
            for right in samples[left_index + 1 :]
            if np.isfinite(similarity := _jaccard_similarity(left, right))
        ]
        bootstrap_stability = float(np.mean(repeatabilities)) if repeatabilities else 0.0

        cross_method_similarities: list[float] = []
        for other in names:
            if other == name:
                continue
            paired_similarities = [
                similarity
                for left, right in zip(samples, edge_sets_by_method[other])
                if np.isfinite(similarity := _jaccard_similarity(left, right))
            ]
            if paired_similarities:
                cross_method_similarities.append(float(np.mean(paired_similarities)))
        diversity = (
            1.0 - float(np.mean(cross_method_similarities))
            if cross_method_similarities
            else 0.0
        )
        redundancy = (
            float(np.mean(cross_method_similarities))
            if cross_method_similarities
            else 0.0
        )

        relative_density = mean_edge_counts[name] / reference_count
        density_factor = 1.0 / (
            1.0 + density_penalty * max(relative_density - 1.0, 0.0)
        )
        prior_weight = max(float(base_weights.get(name, 1.0)), 0.0)
        reliability = max(bootstrap_stability, minimum_weight) ** stability_power
        rows.append(
            {
                "method": name,
                "prior_weight": prior_weight,
                "bootstrap_stability": bootstrap_stability,
                "diversity": diversity,
                "mean_edge_count": mean_edge_counts[name],
                "relative_density": relative_density,
                "density_factor": density_factor,
                "redundancy": redundancy,
                "redundancy_factor": 1.0 / (1.0 + redundancy_penalty * redundancy),
                "adaptive_weight": prior_weight
                * reliability
                * (1.0 + diversity_bonus * diversity)
                * density_factor,
            }
        )

    for row in rows:
        row["adaptive_weight"] = float(row["adaptive_weight"]) * float(
            row["redundancy_factor"]
        )

    weights = np.asarray([float(row["adaptive_weight"]) for row in rows])
    positive = weights[weights > 0.0]
    normalization = float(positive.mean()) if positive.size else 1.0
    for row in rows:
        row["adaptive_weight"] = float(row["adaptive_weight"]) / normalization

    return pd.DataFrame(rows, columns=columns)


def _add_bootstrap_consensus(
    summary: pd.DataFrame,
    base_outputs: Mapping[str, pd.DataFrame],
    bootstrap_outputs: list[Mapping[str, pd.DataFrame]] | None,
    *,
    method_names: list[str],
    method_weights: Mapping[str, float],
    stability_weight: float,
    local_expert_weight: float,
) -> pd.DataFrame:
    frame = summary.copy()
    if frame.empty:
        frame["base_edge_probability"] = pd.Series(dtype=float)
        frame["normalized_evidence_probability"] = pd.Series(dtype=float)
        frame["bootstrap_probability"] = pd.Series(dtype=float)
        frame["bootstrap_evidence_score"] = pd.Series(dtype=float)
        frame["consensus_score"] = pd.Series(dtype=float)
        frame["local_expert_score"] = pd.Series(dtype=float)
        frame["dominant_method"] = pd.Series(dtype=object)
        frame["dominant_method_reliability"] = pd.Series(dtype=float)
        frame["dominant_edge_stability"] = pd.Series(dtype=float)
        frame["ensemble_score"] = pd.Series(dtype=float)
        return frame

    iterations = list(bootstrap_outputs or [])
    stability_weight = float(np.clip(stability_weight, 0.0, 1.0))
    local_expert_weight = float(np.clip(local_expert_weight, 0.0, 1.0))
    weights = {name: max(float(method_weights.get(name, 1.0)), 0.0) for name in method_names}
    total_weight = sum(weights.values())
    if total_weight <= 0.0:
        weights = {name: 1.0 for name in method_names}
        total_weight = float(max(len(method_names), 1))
    maximum_method_weight = max(max(weights.values(), default=1.0), 1e-12)

    def normalized_evidence(output: pd.DataFrame) -> dict[tuple[str, str, int], float]:
        if output is None or output.empty:
            return {}
        normalized = _deduplicate_method_edges(output)
        if normalized.empty:
            return {}
        p_values = pd.to_numeric(normalized.get("p_value"), errors="coerce")
        scores = pd.to_numeric(normalized.get("score"), errors="coerce").abs()
        raw_evidence = (1.0 - p_values).where(p_values.notna(), scores)
        finite = raw_evidence[np.isfinite(raw_evidence)]
        if finite.empty:
            return {}
        if finite.nunique() == 1:
            ranks = pd.Series(1.0, index=finite.index)
        else:
            ranks = finite.rank(method="average", pct=True)

        evidence: dict[tuple[str, str, int], float] = {}
        for index, value in ranks.items():
            row = normalized.loc[index]
            key = _to_edge_key(row.get("source"), row.get("target"), row.get("lag"))
            if key is not None:
                evidence[key] = max(evidence.get(key, 0.0), float(value))
        return evidence

    base_evidence = {
        name: normalized_evidence(base_outputs.get(name, pd.DataFrame()))
        for name in method_names
    }
    bootstrap_evidence = [
        {
            name: normalized_evidence(iteration.get(name, pd.DataFrame()))
            for name in method_names
        }
        for iteration in iterations
    ]
    counts: dict[str, dict[tuple[str, str, int], int]] = {
        name: {} for name in method_names
    }
    for iteration in iterations:
        for name in method_names:
            for edge in _extract_edge_set(iteration.get(name, pd.DataFrame())):
                counts[name][edge] = counts[name].get(edge, 0) + 1

    normalized_base_probabilities: list[float] = []
    bootstrap_probabilities: list[float] = []
    bootstrap_evidence_scores: list[float] = []
    local_expert_scores: list[float] = []
    dominant_methods: list[str | None] = []
    dominant_method_reliabilities: list[float] = []
    dominant_edge_stabilities: list[float] = []
    for _, row in frame.iterrows():
        key = _to_edge_key(row.get("source"), row.get("target"), row.get("lag"))
        if key is None:
            normalized_base_probabilities.append(0.0)
            bootstrap_probabilities.append(0.0)
            bootstrap_evidence_scores.append(0.0)
            local_expert_scores.append(0.0)
            dominant_methods.append(None)
            dominant_method_reliabilities.append(0.0)
            dominant_edge_stabilities.append(0.0)
            continue
        normalized_base_probabilities.append(
            float(
                sum(
                    weights[name] * base_evidence[name].get(key, 0.0)
                    for name in method_names
                )
                / total_weight
            )
        )
        local_scores: dict[str, float] = {}
        method_edge_stabilities: dict[str, float] = {}
        method_reliabilities: dict[str, float] = {}
        for name in method_names:
            base_strength = base_evidence[name].get(key, 0.0)
            if bootstrap_evidence:
                edge_stability = counts[name].get(key, 0) / len(bootstrap_evidence)
                bootstrap_strength = float(
                    np.mean(
                        [
                            evidence_by_method[name].get(key, 0.0)
                            for evidence_by_method in bootstrap_evidence
                        ]
                    )
                )
                stable_strength = math.sqrt(
                    max(bootstrap_strength, 0.0) * max(edge_stability, 0.0)
                )
            else:
                edge_stability = 1.0 if base_strength > 0.0 else 0.0
                stable_strength = base_strength

            reliability = weights[name] / maximum_method_weight
            reliability_factor = 0.5 + 0.5 * reliability
            local_scores[name] = reliability_factor * (
                (1.0 - stability_weight) * base_strength
                + stability_weight * stable_strength
            )
            method_edge_stabilities[name] = float(edge_stability)
            method_reliabilities[name] = float(reliability)

        dominant_method = max(local_scores, key=local_scores.get) if local_scores else None
        local_expert_scores.append(
            float(local_scores.get(dominant_method, 0.0)) if dominant_method else 0.0
        )
        dominant_methods.append(dominant_method)
        dominant_method_reliabilities.append(
            method_reliabilities.get(dominant_method, 0.0) if dominant_method else 0.0
        )
        dominant_edge_stabilities.append(
            method_edge_stabilities.get(dominant_method, 0.0) if dominant_method else 0.0
        )

        if not bootstrap_evidence:
            bootstrap_probabilities.append(float(row.get("weighted_support_ratio", 0.0)))
            bootstrap_evidence_scores.append(normalized_base_probabilities[-1])
            continue
        weighted_frequency = sum(
            weights[name] * counts[name].get(key, 0) / len(iterations)
            for name in method_names
        )
        bootstrap_probabilities.append(float(weighted_frequency / total_weight))
        bootstrap_evidence_scores.append(
            float(
                np.mean(
                    [
                        sum(
                            weights[name] * evidence_by_method[name].get(key, 0.0)
                            for name in method_names
                        )
                        / total_weight
                        for evidence_by_method in bootstrap_evidence
                    ]
                )
            )
        )

    frame["base_edge_probability"] = pd.to_numeric(
        frame["edge_probability"], errors="coerce"
    ).fillna(0.0)
    frame["normalized_evidence_probability"] = normalized_base_probabilities
    frame["bootstrap_probability"] = bootstrap_probabilities
    frame["bootstrap_evidence_score"] = bootstrap_evidence_scores
    frame["edge_probability"] = (
        stability_weight * frame["bootstrap_probability"]
        + (1.0 - stability_weight) * frame["base_edge_probability"]
    ).clip(0.0, 1.0)
    frame["consensus_score"] = (
        stability_weight * frame["bootstrap_evidence_score"]
        + (1.0 - stability_weight) * frame["normalized_evidence_probability"]
    ).clip(0.0, 1.0)
    frame["local_expert_score"] = local_expert_scores
    frame["dominant_method"] = dominant_methods
    frame["dominant_method_reliability"] = dominant_method_reliabilities
    frame["dominant_edge_stability"] = dominant_edge_stabilities
    frame["ensemble_score"] = (
        local_expert_weight * frame["local_expert_score"]
        + (1.0 - local_expert_weight) * frame["consensus_score"]
    ).clip(0.0, 1.0)
    frame["uncertainty"] = 1.0 - frame["edge_probability"]
    return frame.sort_values(
        ["edge_probability", "bootstrap_probability", "votes"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _ridge_validation_mse(
    features: np.ndarray,
    target: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    alpha: float,
) -> float:
    train_features = features[train_indices]
    validation_features = features[validation_indices]
    train_target = target[train_indices]
    validation_target = target[validation_indices]

    feature_mean = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    train_features = (train_features - feature_mean) / feature_scale
    validation_features = (validation_features - feature_mean) / feature_scale

    target_mean = float(train_target.mean())
    system = train_features.T @ train_features + alpha * np.eye(train_features.shape[1])
    coefficients = np.linalg.solve(
        system,
        train_features.T @ (train_target - target_mean),
    )
    prediction = target_mean + validation_features @ coefficients
    return float(np.mean((validation_target - prediction) ** 2))


def _expanding_validation_splits(
    sample_count: int,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    validation_size = max(3, sample_count // 6)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fraction in np.linspace(0.50, 0.80, max(1, int(n_splits))):
        train_end = max(6, int(sample_count * float(fraction)))
        validation_end = min(sample_count, train_end + validation_size)
        if validation_end > train_end:
            splits.append((np.arange(train_end), np.arange(train_end, validation_end)))
    return splits


def add_predictive_validation_score(
    summary: pd.DataFrame,
    data: pd.DataFrame,
    *,
    validation_weight: float = 0.75,
    max_lag: int = 2,
    n_splits: int = 3,
    ridge_alpha: float = 1.0,
    conditional_parents: int = 0,
    uncertainty_penalty: float = 0.0,
    rank_exponent: float = 0.5,
) -> pd.DataFrame:
    """Penaliza no ranking arestas sem ganho preditivo temporal fora da amostra.

    A validacao compara um modelo autorregressivo do alvo com o mesmo modelo
    acrescido da fonte na defasagem da aresta. Opcionalmente, os modelos incluem
    os pais candidatos mais fortes do alvo, aproximando o ganho de uma
    contribuicao condicional. Os cortes sao expansivos e preservam a ordem
    temporal. ``uncertainty_penalty`` troca a media pelo limite conservador
    ``media - penalidade * erro_padrao``. ``rank_exponent`` controla a
    intensidade do gate sobre a posicao percentual do ganho. O resultado
    ajusta apenas ``ensemble_score``; ``edge_probability`` e a decisao binaria
    mantem seus significados.

    Esta medida e evidencia preditiva auxiliar, nao uma prova de causalidade.
    """
    frame = summary.copy()
    if "ensemble_score" not in frame.columns:
        frame["ensemble_score"] = pd.Series(np.nan, index=frame.index, dtype=float)
    frame["pre_validation_ensemble_score"] = pd.to_numeric(
        frame["ensemble_score"], errors="coerce"
    ).fillna(0.0)
    frame["predictive_gain"] = 0.0
    frame["predictive_gain_mean"] = 0.0
    frame["predictive_gain_std"] = 0.0
    frame["predictive_gain_standard_error"] = 0.0
    frame["predictive_positive_split_ratio"] = 0.0
    frame["predictive_rank"] = 0.0

    if frame.empty or data is None or data.empty:
        return frame

    validation_weight = float(np.clip(validation_weight, 0.0, 1.0))
    max_lag = max(1, int(max_lag))
    n_splits = max(1, int(n_splits))
    ridge_alpha = max(float(ridge_alpha), 1e-12)
    conditional_parents = max(0, int(conditional_parents))
    uncertainty_penalty = max(0.0, float(uncertainty_penalty))
    rank_exponent = max(float(rank_exponent), 1e-6)
    numeric_data = data.apply(pd.to_numeric, errors="coerce")
    if len(numeric_data) - max_lag < 12:
        return frame

    candidate_score_column = (
        "pre_validation_ensemble_score"
        if "pre_validation_ensemble_score" in frame
        else "ensemble_score"
    )
    candidate_rows: dict[str, list[tuple[str, int, float]]] = {}
    if conditional_parents:
        for _, candidate in frame.iterrows():
            source = str(candidate.get("source"))
            target = str(candidate.get("target"))
            try:
                lag = max(0, int(candidate.get("lag")))
                score = float(candidate.get(candidate_score_column, 0.0))
            except (TypeError, ValueError, OverflowError):
                continue
            if (
                source in numeric_data
                and target in numeric_data
                and source != target
                and lag <= max_lag
                and np.isfinite(score)
            ):
                candidate_rows.setdefault(target, []).append((source, lag, score))
        for target in candidate_rows:
            candidate_rows[target].sort(key=lambda item: item[2], reverse=True)

    gains: list[float] = []
    mean_gains: list[float] = []
    gain_stds: list[float] = []
    gain_standard_errors: list[float] = []
    positive_split_ratios: list[float] = []

    def append_empty_gain() -> None:
        gains.append(0.0)
        mean_gains.append(0.0)
        gain_stds.append(0.0)
        gain_standard_errors.append(0.0)
        positive_split_ratios.append(0.0)

    for _, row in frame.iterrows():
        source = str(row.get("source"))
        target = str(row.get("target"))
        try:
            edge_lag = max(0, int(row.get("lag")))
        except (TypeError, ValueError, OverflowError):
            append_empty_gain()
            continue
        if source not in numeric_data or target not in numeric_data or edge_lag > max_lag:
            append_empty_gain()
            continue

        target_values = numeric_data[target].to_numpy(dtype=float)
        source_values = numeric_data[source].to_numpy(dtype=float)
        target_vector = target_values[max_lag:]
        autoregressive = np.column_stack(
            [
                target_values[max_lag - lag : len(numeric_data) - lag]
                for lag in range(1, max_lag + 1)
            ]
        )
        source_feature = source_values[
            max_lag - edge_lag : len(numeric_data) - edge_lag if edge_lag else None
        ]
        baseline_features = [autoregressive]
        seen_parent_features: set[tuple[str, int]] = set()
        for parent_source, parent_lag, _ in candidate_rows.get(target, []):
            parent_key = (parent_source, parent_lag)
            if parent_source == source or parent_key in seen_parent_features:
                continue
            parent_values = numeric_data[parent_source].to_numpy(dtype=float)
            parent_feature = parent_values[
                max_lag - parent_lag :
                len(numeric_data) - parent_lag if parent_lag else None
            ]
            baseline_features.append(parent_feature.reshape(-1, 1))
            seen_parent_features.add(parent_key)
            if len(seen_parent_features) >= conditional_parents:
                break
        baseline = np.column_stack(baseline_features)
        augmented = np.column_stack([baseline, source_feature])
        if not (
            np.isfinite(target_vector).all()
            and np.isfinite(baseline).all()
            and np.isfinite(augmented).all()
        ):
            append_empty_gain()
            continue

        sample_count = len(target_vector)
        split_gains: list[float] = []
        for train_indices, validation_indices in _expanding_validation_splits(
            sample_count, n_splits
        ):
            baseline_mse = _ridge_validation_mse(
                baseline,
                target_vector,
                train_indices,
                validation_indices,
                alpha=ridge_alpha,
            )
            augmented_mse = _ridge_validation_mse(
                augmented,
                target_vector,
                train_indices,
                validation_indices,
                alpha=ridge_alpha,
            )
            split_gains.append(
                (baseline_mse - augmented_mse) / max(baseline_mse, 1e-12)
            )
        if not split_gains:
            append_empty_gain()
            continue
        mean_gain = float(np.mean(split_gains))
        gain_std = float(np.std(split_gains, ddof=1)) if len(split_gains) > 1 else 0.0
        gain_standard_error = gain_std / math.sqrt(len(split_gains))
        conservative_gain = mean_gain - uncertainty_penalty * gain_standard_error
        gains.append(max(0.0, conservative_gain))
        mean_gains.append(mean_gain)
        gain_stds.append(gain_std)
        gain_standard_errors.append(gain_standard_error)
        positive_split_ratios.append(float(np.mean(np.asarray(split_gains) > 0.0)))

    frame["predictive_gain"] = gains
    frame["predictive_gain_mean"] = mean_gains
    frame["predictive_gain_std"] = gain_stds
    frame["predictive_gain_standard_error"] = gain_standard_errors
    frame["predictive_positive_split_ratio"] = positive_split_ratios
    positive = frame["predictive_gain"] > 0.0
    if positive.any():
        frame.loc[positive, "predictive_rank"] = frame.loc[
            positive, "predictive_gain"
        ].rank(method="average", pct=True)

    # O piso preserva parte de toda evidencia causal. O expoente permite
    # controlar quanto uma validacao ruidosa em series curtas altera o ranking.
    predictive_gate = frame["predictive_rank"].clip(0.0, 1.0) ** rank_exponent
    frame["ensemble_score"] = frame["pre_validation_ensemble_score"] * (
        (1.0 - validation_weight) + validation_weight * predictive_gate
    )
    return frame


def add_ranked_structure_selection(
    summary: pd.DataFrame,
    *,
    nodes: Sequence[str] | None = None,
    max_pair_density: float = 0.5,
    ranking_column: str = "ensemble_score",
    selection_column: str = "ensemble_selected",
    rescue_bootstrap_min: float | None = None,
    rescue_predictive_rank_min: float | None = None,
    rescue_support_min: float | None = None,
) -> pd.DataFrame:
    """Projeta o ranking do ensemble em uma estrutura binaria esparsa.

    A selecao ocorre no nivel de pares nao direcionados para evitar que varias
    direcoes ou defasagens do mesmo par consumam o orcamento estrutural. O
    ``edge_probability`` original e preservado; a decisao fica na coluna
    booleana ``selection_column``.

    ``max_pair_density`` e um teto, nao uma estimativa da densidade verdadeira.
    Pares fora do teto podem ser resgatados quando satisfazem simultaneamente
    limites de estabilidade bootstrap, ganho preditivo e suporte entre metodos.
    Pares sem evidencia positiva nao sao adicionados apenas para preencher o
    orcamento.
    """
    frame = summary.copy()
    frame[selection_column] = False
    frame["ranked_selection_pair_score"] = 0.0
    frame["ranked_selection_pair_rank"] = pd.Series(
        pd.NA, index=frame.index, dtype="Int64"
    )
    if frame.empty:
        return frame
    if not 0.0 <= float(max_pair_density) <= 1.0:
        raise ValueError("max_pair_density deve estar entre 0 e 1.")
    if ranking_column not in frame:
        raise ValueError(f"Coluna de ranking ausente: {ranking_column}")

    observed_nodes = list(dict.fromkeys(str(node) for node in (nodes or [])))
    if not observed_nodes:
        observed_nodes = list(
            dict.fromkeys(
                [*frame["source"].astype(str), *frame["target"].astype(str)]
            )
        )
    possible_pairs = len(observed_nodes) * max(len(observed_nodes) - 1, 0) // 2
    pair_budget = int(math.ceil(float(max_pair_density) * possible_pairs))
    if pair_budget <= 0:
        return frame

    pair_keys = [
        tuple(sorted((str(source), str(target))))
        for source, target in zip(frame["source"], frame["target"])
    ]
    scores = pd.to_numeric(frame[ranking_column], errors="coerce").fillna(0.0)
    pair_table = pd.DataFrame({"pair_key": pair_keys, "pair_score": scores})
    pair_table = pair_table.loc[
        [source != target for source, target in pair_table["pair_key"]]
    ]
    pair_table = (
        pair_table.groupby("pair_key", as_index=False)["pair_score"]
        .max()
        .loc[lambda current: current["pair_score"] > 0.0]
        .sort_values(["pair_score", "pair_key"], ascending=[False, True])
        .reset_index(drop=True)
    )
    pair_table["pair_rank"] = np.arange(1, len(pair_table) + 1)
    selected_pairs = set(pair_table.head(pair_budget)["pair_key"])
    rescue_thresholds = (
        rescue_bootstrap_min,
        rescue_predictive_rank_min,
        rescue_support_min,
    )
    if any(value is not None for value in rescue_thresholds):
        if not all(value is not None for value in rescue_thresholds):
            raise ValueError("Os tres limiares de resgate devem ser informados juntos.")
        required_columns = {
            "bootstrap_probability",
            "predictive_rank",
            "support_ratio",
        }
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise ValueError(
                "Colunas ausentes para resgate estrutural: "
                + ", ".join(missing_columns)
            )
        rescue_table = pd.DataFrame(
            {
                "pair_key": pair_keys,
                "bootstrap_probability": pd.to_numeric(
                    frame["bootstrap_probability"], errors="coerce"
                ).fillna(0.0),
                "predictive_rank": pd.to_numeric(
                    frame["predictive_rank"], errors="coerce"
                ).fillna(0.0),
                "support_ratio": pd.to_numeric(
                    frame["support_ratio"], errors="coerce"
                ).fillna(0.0),
            }
        ).groupby("pair_key", as_index=False).max()
        rescued_pairs = rescue_table.loc[
            (rescue_table["bootstrap_probability"] >= float(rescue_bootstrap_min))
            & (rescue_table["predictive_rank"] >= float(rescue_predictive_rank_min))
            & (rescue_table["support_ratio"] >= float(rescue_support_min)),
            "pair_key",
        ]
        selected_pairs.update(rescued_pairs)
    score_map = dict(zip(pair_table["pair_key"], pair_table["pair_score"]))
    rank_map = dict(zip(pair_table["pair_key"], pair_table["pair_rank"]))

    frame[selection_column] = [key in selected_pairs for key in pair_keys]
    frame["ranked_selection_pair_score"] = [score_map.get(key, 0.0) for key in pair_keys]
    frame["ranked_selection_pair_rank"] = pd.array(
        [rank_map.get(key, pd.NA) for key in pair_keys], dtype="Int64"
    )
    return frame


def _moving_block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)

    block = max(1, min(int(block_size), n))
    n_blocks = math.ceil(n / block)

    # Em series temporais, usamos blocos contiguos para preservar dependencia local.
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    indices = [np.arange(start, start + block) for start in starts]
    merged = np.concatenate(indices)[:n]
    return merged.astype(int)


def _bootstrap_sample(data: pd.DataFrame, *, block_size: int, rng: np.random.Generator) -> pd.DataFrame:
    indices = _moving_block_bootstrap_indices(len(data), block_size, rng)
    sampled = data.iloc[indices].copy()
    sampled.reset_index(drop=True, inplace=True)
    return sampled


def run_bootstrap_stability_selection(
    data: pd.DataFrame,
    methods: Mapping[str, Callable[..., pd.DataFrame]],
    *,
    method_kwargs: Mapping[str, dict] | None = None,
    method_weights: Mapping[str, float] | None = None,
    expert_knowledge: pd.DataFrame | list[dict] | None = None,
    precomputed_bootstrap_outputs: list[Mapping[str, pd.DataFrame]] | None = None,
    selected_method_names: list[str] | None = None,
    parallel_jobs: int = 1,
    max_bootstrap_seconds: float | None = None,
    n_bootstrap: int = 30,
    block_size: int | None = None,
    min_votes: int = 2,
    use_probabilistic: bool = True,
    selection_probability_threshold: float = 0.5,
    stability_threshold: float = 0.6,
    prior_edge_probability: float = 0.1,
    posterior_weight: float = 0.7,
    confidence_level: float = 0.95,
    random_state: int | None = 42,
) -> pd.DataFrame:
    if n_bootstrap <= 0 and not precomputed_bootstrap_outputs:
        raise ValueError("n_bootstrap deve ser maior que zero.")

    if (data is None or data.empty) and not precomputed_bootstrap_outputs:
        return _empty_stability_frame()

    method_kwargs = method_kwargs or {}
    weight_map = extract_method_weights(method_weights)
    if precomputed_bootstrap_outputs:
        iterations = list(precomputed_bootstrap_outputs)
    else:
        block = block_size if block_size is not None else max(2, len(data) // 10)
        rng = np.random.default_rng(random_state)
        iterations = []
        start_time = time.perf_counter()
        for _ in range(n_bootstrap):
            if (
                max_bootstrap_seconds is not None
                and len(iterations) > 0
                and (time.perf_counter() - start_time) >= float(max_bootstrap_seconds)
            ):
                break
            sampled_data = _bootstrap_sample(data, block_size=block, rng=rng)
            outputs = _run_method_suite_fast(
                sampled_data,
                methods,
                method_kwargs=method_kwargs,
                parallel_jobs=parallel_jobs,
            )
            iterations.append(outputs)

    if not iterations:
        return _empty_stability_frame()

    n_iterations = len(iterations)
    rng = np.random.default_rng(random_state)

    stats: dict[tuple[str, str, int], dict[str, float]] = {}

    for outputs_full in iterations:
        if selected_method_names is not None:
            outputs = {
                name: outputs_full[name]
                for name in selected_method_names
                if name in outputs_full
            }
        else:
            outputs = dict(outputs_full)

        if not outputs:
            continue

        results = list(outputs.values())

        if use_probabilistic:
            summary = summarize_probabilistic_ensemble(
                results,
                min_votes=min_votes,
                prior_edge_probability=prior_edge_probability,
                posterior_weight=posterior_weight,
                confidence_level=confidence_level,
                method_weights=weight_map,
                method_names=list(outputs.keys()),
            )
            summary = apply_expert_knowledge_to_summary(summary, expert_knowledge, hard_filter=True)
            selected = summary[summary["edge_probability"] >= selection_probability_threshold]
        else:
            summary = summarize_ensemble(results, min_votes=min_votes)
            selected = summary

        selected_keys = _extract_edge_set(selected)
        for _, row in summary.iterrows():
            key = _to_edge_key(row.get("source"), row.get("target"), row.get("lag"))
            if key is None:
                continue

            bucket = stats.setdefault(
                key,
                {
                    "selected_count": 0.0,
                    "appearance_count": 0.0,
                    "prob_sum": 0.0,
                    "prob_count": 0.0,
                    "confidence_sum": 0.0,
                    "confidence_count": 0.0,
                },
            )

            bucket["appearance_count"] += 1.0
            if key in selected_keys:
                bucket["selected_count"] += 1.0

            edge_probability = row.get("edge_probability", np.nan)
            if pd.notna(edge_probability):
                bucket["prob_sum"] += float(edge_probability)
                bucket["prob_count"] += 1.0

            confidence = row.get("confidence", np.nan)
            if pd.notna(confidence):
                bucket["confidence_sum"] += float(confidence)
                bucket["confidence_count"] += 1.0

    records: list[dict[str, Any]] = []
    for (source, target, lag), bucket in stats.items():
        selected_count = int(bucket["selected_count"])
        selection_frequency = selected_count / n_iterations
        appearance_frequency = bucket["appearance_count"] / n_iterations
        mean_edge_probability = (
            bucket["prob_sum"] / bucket["prob_count"] if bucket["prob_count"] > 0 else np.nan
        )
        mean_confidence = (
            bucket["confidence_sum"] / bucket["confidence_count"]
            if bucket["confidence_count"] > 0
            else np.nan
        )

        records.append(
            {
                "source": source,
                "target": target,
                "lag": lag,
                "selected_count": selected_count,
                "selection_frequency": float(selection_frequency),
                "appearance_frequency": float(appearance_frequency),
                "mean_edge_probability": float(mean_edge_probability)
                if pd.notna(mean_edge_probability)
                else np.nan,
                "mean_confidence": float(mean_confidence) if pd.notna(mean_confidence) else np.nan,
                "stability_selected": bool(selection_frequency >= stability_threshold),
            }
        )

    if not records:
        return _empty_stability_frame()

    return (
        pd.DataFrame(records)
        .sort_values(["selection_frequency", "appearance_frequency"], ascending=[False, False])
        .reset_index(drop=True)
    )


def _mean_upper_triangle(matrix: pd.DataFrame) -> float:
    if matrix is None or matrix.empty or len(matrix) <= 1:
        return 1.0
    values = matrix.to_numpy(dtype=float)
    upper = values[np.triu_indices_from(values, k=1)]
    if upper.size == 0:
        return 1.0
    finite = upper[np.isfinite(upper)]
    if finite.size == 0:
        return 0.0
    return float(finite.mean())


def _default_performance_score(metrics: dict[str, float]) -> float:
    if metrics["num_edges"] <= 0.0:
        return 0.0
    # A selecao e guiada por repetibilidade e evidencia; parcimonia evita que
    # combinacoes densas sejam premiadas apenas por produzirem mais arestas.
    return float(
        0.40 * metrics["mean_stability"]
        + 0.20 * metrics["mean_confidence"]
        + 0.15 * metrics["mean_edge_probability"]
        + 0.15 * metrics["stable_edge_ratio"]
        + 0.10 * (1.0 - math.sqrt(metrics["edge_density"]))
    )


def evaluate_method_combination(
    data: pd.DataFrame,
    methods: Mapping[str, Callable[..., pd.DataFrame]],
    *,
    method_kwargs: Mapping[str, dict] | None = None,
    method_weights: Mapping[str, float] | None = None,
    expert_knowledge: pd.DataFrame | list[dict] | None = None,
    precomputed_outputs: Mapping[str, pd.DataFrame] | None = None,
    precomputed_bootstrap_outputs: list[Mapping[str, pd.DataFrame]] | None = None,
    parallel_jobs: int = 1,
    max_bootstrap_seconds: float | None = None,
    min_votes: int = 2,
    n_bootstrap: int = 30,
    block_size: int | None = None,
    stability_threshold: float = 0.6,
    selection_probability_threshold: float = 0.5,
    prior_edge_probability: float = 0.1,
    posterior_weight: float = 0.7,
    confidence_level: float = 0.95,
    adaptive_method_weights: bool = True,
    stability_weight: float = 0.65,
    local_expert_weight: float = 0.60,
    predictive_validation_weight: float = 0.0,
    predictive_validation_max_lag: int | None = None,
    predictive_validation_splits: int = 3,
    predictive_validation_ridge_alpha: float = 1.0,
    predictive_validation_conditional_parents: int = 0,
    predictive_validation_uncertainty_penalty: float = 0.0,
    predictive_validation_rank_exponent: float = 0.5,
    ranked_selection_max_pair_density: float | None = None,
    ranked_selection_rescue_bootstrap_min: float | None = None,
    ranked_selection_rescue_predictive_rank_min: float | None = None,
    ranked_selection_rescue_support_min: float | None = None,
    method_redundancy_penalty: float = 0.0,
    method_stability_power: float = 1.0,
    method_diversity_bonus: float = 0.15,
    method_density_penalty: float = 0.5,
    minimum_method_weight: float = 0.05,
    random_state: int | None = 42,
    score_fn: MetricScoreFn | None = None,
) -> dict[str, Any]:
    method_kwargs = method_kwargs or {}
    weight_map = extract_method_weights(method_weights)
    if precomputed_outputs is not None:
        outputs = dict(precomputed_outputs)
    else:
        outputs = _run_method_suite_fast(
            data,
            methods,
            method_kwargs=method_kwargs,
            parallel_jobs=parallel_jobs,
        )
    results = list(outputs.values())
    if precomputed_bootstrap_outputs is None:
        block = block_size if block_size is not None else max(2, len(data) // 10)
        rng = np.random.default_rng(random_state)
        generated_bootstrap_outputs: list[dict[str, pd.DataFrame]] = []
        start_time = time.perf_counter()
        for _ in range(n_bootstrap):
            if (
                max_bootstrap_seconds is not None
                and generated_bootstrap_outputs
                and (time.perf_counter() - start_time) >= float(max_bootstrap_seconds)
            ):
                break
            sampled_data = _bootstrap_sample(data, block_size=block, rng=rng)
            generated_bootstrap_outputs.append(
                _run_method_suite_fast(
                    sampled_data,
                    methods,
                    method_kwargs=method_kwargs,
                    parallel_jobs=parallel_jobs,
                )
            )
        precomputed_bootstrap_outputs = generated_bootstrap_outputs
    selected_bootstrap_outputs = [
        {name: iteration[name] for name in outputs if name in iteration}
        for iteration in list(precomputed_bootstrap_outputs or [])
    ]
    adaptive_weight_table = estimate_adaptive_method_weights(
        outputs,
        selected_bootstrap_outputs,
        base_weights=weight_map,
        stability_power=method_stability_power,
        diversity_bonus=method_diversity_bonus,
        density_penalty=method_density_penalty,
        minimum_weight=minimum_method_weight,
        redundancy_penalty=method_redundancy_penalty,
    )
    if adaptive_method_weights:
        effective_weights = dict(
            zip(
                adaptive_weight_table["method"],
                adaptive_weight_table["adaptive_weight"],
            )
        )
    else:
        effective_weights = {
            name: max(float(weight_map.get(name, 1.0)), 0.0)
            for name in outputs
        }

    probabilistic_summary = summarize_probabilistic_ensemble(
        results,
        min_votes=min_votes,
        prior_edge_probability=prior_edge_probability,
        posterior_weight=posterior_weight,
        confidence_level=confidence_level,
        method_weights=effective_weights,
        method_names=list(outputs.keys()),
    )
    probabilistic_summary = _add_bootstrap_consensus(
        probabilistic_summary,
        outputs,
        selected_bootstrap_outputs,
        method_names=list(outputs),
        method_weights=effective_weights,
        stability_weight=stability_weight,
        local_expert_weight=local_expert_weight,
    )
    if predictive_validation_weight > 0.0:
        predictive_max_lag = predictive_validation_max_lag
        if predictive_max_lag is None:
            observed_lags: list[int] = []
            for output in outputs.values():
                if output is None or output.empty or "lag" not in output:
                    continue
                numeric_lags = pd.to_numeric(output["lag"], errors="coerce").dropna()
                if not numeric_lags.empty:
                    observed_lags.append(int(numeric_lags.max()))
            predictive_max_lag = max(observed_lags or [1])
        probabilistic_summary = add_predictive_validation_score(
            probabilistic_summary,
            data,
            validation_weight=predictive_validation_weight,
            max_lag=predictive_max_lag,
            n_splits=predictive_validation_splits,
            ridge_alpha=predictive_validation_ridge_alpha,
            conditional_parents=predictive_validation_conditional_parents,
            uncertainty_penalty=predictive_validation_uncertainty_penalty,
            rank_exponent=predictive_validation_rank_exponent,
        )
    probabilistic_summary = apply_expert_knowledge_to_summary(
        probabilistic_summary,
        expert_knowledge,
        hard_filter=True,
    )
    if ranked_selection_max_pair_density is not None:
        probabilistic_summary = add_ranked_structure_selection(
            probabilistic_summary,
            nodes=list(data.columns),
            max_pair_density=ranked_selection_max_pair_density,
            rescue_bootstrap_min=ranked_selection_rescue_bootstrap_min,
            rescue_predictive_rank_min=(
                ranked_selection_rescue_predictive_rank_min
            ),
            rescue_support_min=ranked_selection_rescue_support_min,
        )

    stability = run_bootstrap_stability_selection(
        data,
        methods,
        method_kwargs=method_kwargs,
        method_weights=effective_weights,
        expert_knowledge=expert_knowledge,
        precomputed_bootstrap_outputs=precomputed_bootstrap_outputs,
        selected_method_names=list(methods.keys()),
        parallel_jobs=parallel_jobs,
        max_bootstrap_seconds=max_bootstrap_seconds,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        min_votes=min_votes,
        use_probabilistic=True,
        selection_probability_threshold=selection_probability_threshold,
        stability_threshold=stability_threshold,
        prior_edge_probability=prior_edge_probability,
        posterior_weight=posterior_weight,
        confidence_level=confidence_level,
        random_state=random_state,
    )

    consistency = compute_method_consistency(outputs)

    if "ensemble_selected" in probabilistic_summary:
        selected_summary = probabilistic_summary.loc[
            probabilistic_summary["ensemble_selected"].fillna(False).astype(bool)
        ].copy()
    else:
        selected_summary = probabilistic_summary.loc[
            probabilistic_summary["edge_probability"] >= selection_probability_threshold
        ].copy()
    max_observed_lag = 1
    for output in outputs.values():
        if output is not None and not output.empty and "lag" in output.columns:
            numeric_lags = pd.to_numeric(output["lag"], errors="coerce").dropna()
            if not numeric_lags.empty:
                max_observed_lag = max(max_observed_lag, int(numeric_lags.max()))
    possible_edges = max(
        len(data.columns) * max(len(data.columns) - 1, 0) * max_observed_lag,
        1,
    )
    edge_density = float(np.clip(len(selected_summary) / possible_edges, 0.0, 1.0))
    selected_bootstrap_probability = pd.to_numeric(
        selected_summary.get("bootstrap_probability", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    selected_stability_ratio = (
        float((selected_bootstrap_probability >= stability_threshold).mean())
        if not selected_bootstrap_probability.empty
        else 0.0
    )

    metrics: dict[str, float] = {
        "num_methods": float(len(methods)),
        "num_edges": float(len(selected_summary)),
        "num_candidate_edges": float(len(probabilistic_summary)),
        "edge_density": edge_density,
        "mean_edge_probability": float(selected_summary["edge_probability"].mean())
        if not selected_summary.empty
        else 0.0,
        "mean_confidence": float(selected_summary["confidence"].mean())
        if not selected_summary.empty
        else 0.0,
        "mean_uncertainty": float(selected_summary["uncertainty"].mean())
        if not selected_summary.empty
        else 1.0,
        "mean_stability": float(selected_bootstrap_probability.mean())
        if not selected_bootstrap_probability.empty
        else 0.0,
        "stable_edge_ratio": selected_stability_ratio,
        "mean_method_agreement": _mean_upper_triangle(consistency),
        "mean_method_stability": float(adaptive_weight_table["bootstrap_stability"].mean()),
        "mean_method_diversity": float(adaptive_weight_table["diversity"].mean()),
    }

    scorer = score_fn or (lambda current, *_: _default_performance_score(current))
    performance_score = float(scorer(metrics, probabilistic_summary, stability, consistency))
    metrics["performance_score"] = performance_score

    return {
        "methods": list(methods.keys()),
        "outputs": outputs,
        "probabilistic_summary": probabilistic_summary,
        "stability": stability,
        "consistency": consistency,
        "method_weight_diagnostics": adaptive_weight_table,
        "effective_method_weights": effective_weights,
        "metrics": metrics,
    }


def select_robust_ensemble_combination(
    data: pd.DataFrame,
    methods: Mapping[str, Callable[..., pd.DataFrame]],
    *,
    method_kwargs: Mapping[str, dict] | None = None,
    method_weights: Mapping[str, float] | None = None,
    expert_knowledge: pd.DataFrame | list[dict] | None = None,
    precomputed_outputs: Mapping[str, pd.DataFrame] | None = None,
    precomputed_bootstrap_outputs: Sequence[Mapping[str, pd.DataFrame]] | None = None,
    precompute_runs: bool = True,
    parallel_jobs: int = 1,
    max_bootstrap_seconds: float | None = None,
    min_methods: int = 2,
    max_methods: int | None = None,
    min_votes: int = 2,
    n_bootstrap: int = 30,
    block_size: int | None = None,
    stability_threshold: float = 0.6,
    selection_probability_threshold: float = 0.5,
    prior_edge_probability: float = 0.1,
    posterior_weight: float = 0.7,
    confidence_level: float = 0.95,
    adaptive_method_weights: bool = True,
    stability_weight: float = 0.65,
    local_expert_weight: float = 0.60,
    predictive_validation_weight: float = 0.0,
    predictive_validation_max_lag: int | None = None,
    predictive_validation_splits: int = 3,
    predictive_validation_ridge_alpha: float = 1.0,
    predictive_validation_conditional_parents: int = 0,
    predictive_validation_uncertainty_penalty: float = 0.0,
    predictive_validation_rank_exponent: float = 0.5,
    ranked_selection_max_pair_density: float | None = None,
    ranked_selection_rescue_bootstrap_min: float | None = None,
    ranked_selection_rescue_predictive_rank_min: float | None = None,
    ranked_selection_rescue_support_min: float | None = None,
    method_redundancy_penalty: float = 0.0,
    method_stability_power: float = 1.0,
    method_diversity_bonus: float = 0.15,
    method_density_penalty: float = 0.5,
    minimum_method_weight: float = 0.05,
    random_state: int | None = 42,
    score_fn: MetricScoreFn | None = None,
) -> dict[str, Any]:
    if not methods:
        raise ValueError("methods nao pode ser vazio.")

    names = list(methods.keys())
    max_size = len(names) if max_methods is None else min(max_methods, len(names))
    min_size = max(1, min_methods)
    if min_size > max_size:
        raise ValueError("min_methods nao pode ser maior que max_methods.")

    method_kwargs = method_kwargs or {}
    evaluations: dict[str, dict[str, Any]] = {}
    ranking_rows: list[dict[str, Any]] = []

    base_outputs_all = (
        dict(precomputed_outputs) if precomputed_outputs is not None else None
    )
    cached_bootstrap_outputs = (
        [dict(iteration) for iteration in precomputed_bootstrap_outputs]
        if precomputed_bootstrap_outputs is not None
        else None
    )

    if precompute_runs and base_outputs_all is None:
        # Reuso de resultados evita executar métodos pesados repetidamente por combinação.
        base_outputs_all = _run_method_suite_fast(
            data,
            methods,
            method_kwargs=method_kwargs,
            parallel_jobs=parallel_jobs,
        )
    if cached_bootstrap_outputs is not None:
        cached_bootstrap_outputs = cached_bootstrap_outputs[: max(0, int(n_bootstrap))]
    if precompute_runs and (
        cached_bootstrap_outputs is None
        or len(cached_bootstrap_outputs) < max(0, int(n_bootstrap))
    ):
        block = block_size if block_size is not None else max(2, len(data) // 10)
        rng = np.random.default_rng(random_state)
        cached_bootstrap_outputs = cached_bootstrap_outputs or []
        # Avanca o gerador pelas amostras ja persistidas para que estender um
        # cache produza a mesma sequencia de uma execucao integral.
        for _ in range(len(cached_bootstrap_outputs)):
            _bootstrap_sample(data, block_size=block, rng=rng)
        start_time = time.perf_counter()
        for _ in range(len(cached_bootstrap_outputs), max(0, int(n_bootstrap))):
            if (
                max_bootstrap_seconds is not None
                and len(cached_bootstrap_outputs) > 0
                and (time.perf_counter() - start_time) >= float(max_bootstrap_seconds)
            ):
                break
            sampled_data = _bootstrap_sample(data, block_size=block, rng=rng)
            cached_bootstrap_outputs.append(
                _run_method_suite_fast(
                    sampled_data,
                    methods,
                    method_kwargs=method_kwargs,
                    parallel_jobs=parallel_jobs,
                )
            )

    for size in range(min_size, max_size + 1):
        for combo in combinations(names, size):
            combo_methods = {name: methods[name] for name in combo}
            combo_kwargs = {name: method_kwargs.get(name, {}) for name in combo}

            evaluation = evaluate_method_combination(
                data,
                combo_methods,
                method_kwargs=combo_kwargs,
                method_weights={name: method_weights[name] for name in combo if method_weights and name in method_weights},
                expert_knowledge=expert_knowledge,
                precomputed_outputs={name: base_outputs_all[name] for name in combo} if base_outputs_all else None,
                precomputed_bootstrap_outputs=cached_bootstrap_outputs,
                parallel_jobs=parallel_jobs,
                max_bootstrap_seconds=max_bootstrap_seconds,
                min_votes=min_votes,
                n_bootstrap=n_bootstrap,
                block_size=block_size,
                stability_threshold=stability_threshold,
                selection_probability_threshold=selection_probability_threshold,
                prior_edge_probability=prior_edge_probability,
                posterior_weight=posterior_weight,
                confidence_level=confidence_level,
                adaptive_method_weights=adaptive_method_weights,
                stability_weight=stability_weight,
                local_expert_weight=local_expert_weight,
                predictive_validation_weight=predictive_validation_weight,
                predictive_validation_max_lag=predictive_validation_max_lag,
                predictive_validation_splits=predictive_validation_splits,
                predictive_validation_ridge_alpha=predictive_validation_ridge_alpha,
                predictive_validation_conditional_parents=(
                    predictive_validation_conditional_parents
                ),
                predictive_validation_uncertainty_penalty=(
                    predictive_validation_uncertainty_penalty
                ),
                predictive_validation_rank_exponent=(
                    predictive_validation_rank_exponent
                ),
                ranked_selection_max_pair_density=ranked_selection_max_pair_density,
                ranked_selection_rescue_bootstrap_min=(
                    ranked_selection_rescue_bootstrap_min
                ),
                ranked_selection_rescue_predictive_rank_min=(
                    ranked_selection_rescue_predictive_rank_min
                ),
                ranked_selection_rescue_support_min=(
                    ranked_selection_rescue_support_min
                ),
                method_redundancy_penalty=method_redundancy_penalty,
                method_stability_power=method_stability_power,
                method_diversity_bonus=method_diversity_bonus,
                method_density_penalty=method_density_penalty,
                minimum_method_weight=minimum_method_weight,
                random_state=random_state,
                score_fn=score_fn,
            )

            key = " + ".join(combo)
            evaluations[key] = evaluation
            metrics = evaluation["metrics"]
            ranking_rows.append(
                {
                    "combination": key,
                    "num_methods": int(metrics["num_methods"]),
                    "num_edges": int(metrics["num_edges"]),
                    "edge_density": float(metrics["edge_density"]),
                    "mean_stability": float(metrics["mean_stability"]),
                    "stable_edge_ratio": float(metrics["stable_edge_ratio"]),
                    "mean_edge_probability": float(metrics["mean_edge_probability"]),
                    "mean_confidence": float(metrics["mean_confidence"]),
                    "mean_method_agreement": float(metrics["mean_method_agreement"]),
                    "mean_method_stability": float(metrics["mean_method_stability"]),
                    "mean_method_diversity": float(metrics["mean_method_diversity"]),
                    "performance_score": float(metrics["performance_score"]),
                }
            )

    ranking = pd.DataFrame(ranking_rows)
    if ranking.empty:
        raise RuntimeError("Nenhuma combinacao foi avaliada.")

    ranking = ranking.sort_values(
        ["performance_score", "mean_stability", "mean_confidence"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best_key = str(ranking.iloc[0]["combination"])
    best_eval = evaluations[best_key]

    return {
        "best_combination": best_eval["methods"],
        "best_evaluation": best_eval,
        "ranking": ranking,
        "all_evaluations": evaluations,
        "precomputed_outputs": base_outputs_all,
        "precomputed_bootstrap_outputs": cached_bootstrap_outputs,
    }
