import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


def _undirected_pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    """Converte arestas em pares sem direção, removendo autorrelações."""
    if frame.empty:
        return set()

    required_columns = {"source", "target"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            "A tabela de arestas não possui as colunas obrigatórias: "
            f"{sorted(missing_columns)}"
        )

    return {
        tuple(sorted((str(source), str(target))))
        for source, target in zip(frame["source"], frame["target"])
        if pd.notna(source) and pd.notna(target) and str(source) != str(target)
    }


def compute_undirected_skeleton_metrics(
    predicted_summary: pd.DataFrame,
    ground_truth: pd.DataFrame,
    prob_threshold: float = 0.5,
    nodes: list[str] | tuple[str, ...] | None = None,
    evaluated_relations: list[tuple[str, str]] | None = None,
) -> dict[str, object]:
    """Avalia adjacências quando o ground truth não informa direção ou lag.

    Arestas em direções ou lags diferentes são reduzidas ao mesmo par de nós. Quando
    há mais de uma previsão para o par, basta que uma delas alcance o limiar de
    ``edge_probability`` para que a adjacência seja considerada prevista. Se
    ``evaluated_relations`` for informado, a comparação considera somente esses pares.
    """
    if not 0.0 <= prob_threshold <= 1.0:
        raise ValueError("prob_threshold deve estar entre 0 e 1.")

    if predicted_summary.empty:
        predictions = predicted_summary
    elif "edge_probability" in predicted_summary.columns:
        probabilities = pd.to_numeric(
            predicted_summary["edge_probability"], errors="coerce"
        )
        predictions = predicted_summary.loc[probabilities >= prob_threshold]
    else:
        predictions = predicted_summary

    predicted_pairs = _undirected_pairs(predictions)
    true_pairs = _undirected_pairs(ground_truth)
    evaluated_pairs = (
        {
            tuple(sorted((str(source), str(target))))
            for source, target in evaluated_relations
            if str(source) != str(target)
        }
        if evaluated_relations is not None
        else None
    )
    if evaluated_pairs is not None:
        predicted_pairs &= evaluated_pairs
        true_pairs &= evaluated_pairs
    true_positive_pairs = predicted_pairs & true_pairs
    false_positive_pairs = predicted_pairs - true_pairs
    false_negative_pairs = true_pairs - predicted_pairs

    true_positives = len(true_positive_pairs)
    false_positives = len(false_positive_pairs)
    false_negatives = len(false_negative_pairs)
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    if nodes is None:
        node_set = {
            node
            for pair in predicted_pairs | true_pairs
            for node in pair
        }
    else:
        node_set = {str(node) for node in nodes}

    candidate_pairs = (
        len(evaluated_pairs)
        if evaluated_pairs is not None
        else len(node_set) * (len(node_set) - 1) // 2
    )
    ground_truth_prevalence = (
        len(true_pairs) / candidate_pairs if candidate_pairs else 0.0
    )
    all_pairs_baseline_f1 = (
        2 * ground_truth_prevalence / (1.0 + ground_truth_prevalence)
        if ground_truth_prevalence
        else 0.0
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "structural_hamming_distance": false_positives + false_negatives,
        "candidate_pairs": candidate_pairs,
        "ground_truth_pairs": len(true_pairs),
        "ground_truth_prevalence": ground_truth_prevalence,
        "all_pairs_baseline_f1": all_pairs_baseline_f1,
        "true_positive_pairs": sorted(true_positive_pairs),
        "false_positive_pairs": sorted(false_positive_pairs),
        "false_negative_pairs": sorted(false_negative_pairs),
    }


def compute_ranked_undirected_skeleton_metrics(
    pair_scores: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    score_column: str = "score",
) -> dict[str, float | int]:
    """Avalia o ranking completo de pares sem depender de limiar binario.

    AUROC mede a ordenacao entre positivos e negativos. Average precision (AP),
    equivalente a area sob a curva precision-recall em sua definicao por degraus,
    e mais informativa quando o esqueleto e esparso.
    """
    required_columns = {"source", "target", score_column}
    missing_columns = required_columns - set(pair_scores.columns)
    if missing_columns:
        raise ValueError(
            "A tabela de escores nao possui as colunas obrigatorias: "
            f"{sorted(missing_columns)}"
        )
    if pair_scores.empty:
        raise ValueError("pair_scores nao pode ser vazio.")

    true_pairs = _undirected_pairs(ground_truth)
    normalized = pair_scores.copy()
    normalized["_pair"] = [
        tuple(sorted((str(source), str(target))))
        for source, target in zip(normalized["source"], normalized["target"])
    ]
    if normalized["_pair"].duplicated().any():
        raise ValueError("pair_scores deve conter exatamente uma linha por par.")
    if any(source == target for source, target in normalized["_pair"]):
        raise ValueError("pair_scores nao deve conter autorrelacoes.")

    scores = pd.to_numeric(normalized[score_column], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores.to_numpy()).all():
        raise ValueError(f"{score_column} deve conter apenas numeros finitos.")
    labels = normalized["_pair"].isin(true_pairs).astype(int).to_numpy()
    if np.unique(labels).size < 2:
        raise ValueError(
            "AUROC/AUPRC exigem ao menos um par positivo e um par negativo."
        )

    prevalence = float(labels.mean())
    return {
        "candidate_pairs": int(len(labels)),
        "positive_pairs": int(labels.sum()),
        "negative_pairs": int(len(labels) - labels.sum()),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "random_average_precision": prevalence,
    }


def build_complete_undirected_pair_scores(
    predictions: pd.DataFrame,
    nodes: list[str] | tuple[str, ...],
    *,
    evidence: str,
    default_score: float = 0.0,
) -> pd.DataFrame:
    """Reduz arestas direcionadas a um ranking completo de pares de nos.

    ``evidence`` aceita ``probability`` (maior ``edge_probability``),
    ``ensemble_score`` (ranking normalizado do ensemble),
    ``one_minus_p_value`` (maior ``1 - p_value``) ou ``absolute_score``
    (maior magnitude de ``score``). Pares ausentes recebem ``default_score``.
    """
    node_names = tuple(str(node) for node in nodes)
    if len(node_names) < 2:
        raise ValueError("nodes deve conter ao menos dois nos.")
    if len(set(node_names)) != len(node_names):
        raise ValueError("nodes nao pode conter nomes duplicados.")
    if not np.isfinite(float(default_score)):
        raise ValueError("default_score deve ser finito.")

    modes = {
        "probability": "edge_probability",
        "ensemble_score": "ensemble_score",
        "one_minus_p_value": "p_value",
        "absolute_score": "score",
    }
    normalized_evidence = str(evidence).strip().lower()
    if normalized_evidence not in modes:
        raise ValueError(
            "evidence deve ser 'probability', 'ensemble_score', "
            "'one_minus_p_value' ou 'absolute_score'."
        )

    required = {"source", "target"}
    if not predictions.empty:
        required.add(modes[normalized_evidence])
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            "A tabela de previsoes nao possui as colunas obrigatorias: "
            f"{sorted(missing)}"
        )

    allowed_nodes = set(node_names)
    pair_scores: dict[tuple[str, str], float] = {}
    if not predictions.empty:
        value_column = modes[normalized_evidence]
        for source, target, raw_value in predictions[
            ["source", "target", value_column]
        ].itertuples(index=False, name=None):
            source_name = str(source)
            target_name = str(target)
            if (
                source_name == target_name
                or source_name not in allowed_nodes
                or target_name not in allowed_nodes
            ):
                continue
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric_value):
                continue
            if normalized_evidence == "one_minus_p_value":
                if not 0.0 <= numeric_value <= 1.0:
                    continue
                score = 1.0 - numeric_value
            elif normalized_evidence == "absolute_score":
                score = abs(numeric_value)
            else:
                score = numeric_value

            pair = tuple(sorted((source_name, target_name)))
            pair_scores[pair] = max(pair_scores.get(pair, float("-inf")), score)

    records = []
    for source_index, source in enumerate(node_names):
        for target in node_names[source_index + 1 :]:
            pair = tuple(sorted((source, target)))
            records.append(
                {
                    "source": source,
                    "target": target,
                    "score": float(pair_scores.get(pair, default_score)),
                }
            )
    return pd.DataFrame(records, columns=["source", "target", "score"])


def compute_paired_superiority_statistics(
    results: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    metric: str,
    higher_is_better: bool = True,
    trajectory_column: str = "trajectory_index",
    strategy_column: str = "strategy",
    confidence_level: float = 0.95,
    n_bootstrap: int = 10_000,
    random_state: int | None = 42,
) -> dict[str, float | int | str]:
    """Resume uma comparacao pareada entre duas estrategias.

    Diferencas positivas sempre favorecem ``candidate``. O intervalo usa
    bootstrap pareado da media e o Wilcoxon e bilateral. Correcoes para
    comparacoes multiplas devem ser aplicadas pelo chamador.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level deve estar entre 0 e 1.")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap deve ser maior que zero.")

    required = {trajectory_column, strategy_column, metric}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(
            "A tabela de resultados nao possui as colunas obrigatorias: "
            f"{sorted(missing)}"
        )

    selected = results.loc[
        results[strategy_column].astype(str).isin([str(candidate), str(baseline)]),
        [trajectory_column, strategy_column, metric],
    ].copy()
    if selected.duplicated([trajectory_column, strategy_column]).any():
        raise ValueError(
            "Deve existir no maximo uma observacao por trajetoria e estrategia."
        )
    selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
    paired = selected.pivot(
        index=trajectory_column,
        columns=strategy_column,
        values=metric,
    )
    if str(candidate) not in paired.columns or str(baseline) not in paired.columns:
        raise ValueError("candidate e baseline devem possuir resultados pareados.")
    paired = paired[[str(candidate), str(baseline)]].dropna()
    if paired.empty:
        raise ValueError("Nenhuma trajetoria pareada disponivel para comparacao.")

    candidate_values = paired[str(candidate)].to_numpy(dtype=float)
    baseline_values = paired[str(baseline)].to_numpy(dtype=float)
    raw_difference = candidate_values - baseline_values
    oriented_difference = raw_difference if higher_is_better else -raw_difference

    rng = np.random.default_rng(random_state)
    sample_indices = rng.integers(
        0,
        len(oriented_difference),
        size=(int(n_bootstrap), len(oriented_difference)),
    )
    bootstrap_means = oriented_difference[sample_indices].mean(axis=1)
    alpha = 1.0 - float(confidence_level)
    ci_low, ci_high = np.quantile(
        bootstrap_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )

    if np.allclose(oriented_difference, 0.0):
        wilcoxon_p_value = 1.0
    else:
        wilcoxon_p_value = float(
            wilcoxon(
                oriented_difference,
                alternative="two-sided",
                zero_method="wilcox",
            ).pvalue
        )

    standard_deviation = (
        float(np.std(oriented_difference, ddof=1)) if len(paired) > 1 else 0.0
    )
    mean_improvement = float(np.mean(oriented_difference))
    standardized_effect = (
        float(mean_improvement / standard_deviation)
        if standard_deviation > 0.0
        else (float("inf") if mean_improvement > 0.0 else 0.0)
    )
    return {
        "candidate": str(candidate),
        "baseline": str(baseline),
        "metric": str(metric),
        "paired_trajectories": int(len(paired)),
        "candidate_mean": float(np.mean(candidate_values)),
        "baseline_mean": float(np.mean(baseline_values)),
        "mean_improvement": mean_improvement,
        "median_improvement": float(np.median(oriented_difference)),
        "confidence_interval_low": float(ci_low),
        "confidence_interval_high": float(ci_high),
        "win_rate": float(np.mean(oriented_difference > 0.0)),
        "tie_rate": float(np.mean(np.isclose(oriented_difference, 0.0))),
        "standardized_effect": standardized_effect,
        "wilcoxon_p_value": wilcoxon_p_value,
    }


def generate_synthetic_timeseries(
    n_samples: int = 500,
    noise_std: float = 0.1,
    random_state: int | None = 42,
):
    """Gera séries temporais sintéticas com estrutura causal conhecida."""
    rng = np.random.default_rng(random_state)
    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    z = np.zeros(n_samples)

    for time in range(2, n_samples):
        x[time] = 0.7 * x[time - 1] + rng.normal(0, noise_std)
        y[time] = 0.5 * y[time - 1] + 0.8 * x[time - 2] + rng.normal(0, noise_std)
        z[time] = (
            0.6 * z[time - 1]
            + 0.9 * y[time - 1]
            - 0.5 * x[time - 1]
            + rng.normal(0, noise_std)
        )

    ground_truth = pd.DataFrame(
        [
            {"source": "X", "target": "X", "lag": 1},
            {"source": "Y", "target": "Y", "lag": 1},
            {"source": "Z", "target": "Z", "lag": 1},
            {"source": "X", "target": "Y", "lag": 2},
            {"source": "Y", "target": "Z", "lag": 1},
            {"source": "X", "target": "Z", "lag": 1},
        ]
    )
    return pd.DataFrame({"X": x, "Y": y, "Z": z}), ground_truth


def compute_structural_metrics(
    predicted_summary: pd.DataFrame,
    ground_truth: pd.DataFrame,
    prob_threshold: float = 0.5,
):
    """Calcula precision, recall, F1 e SHD para arestas direcionadas com lag."""
    if "edge_probability" in predicted_summary.columns:
        predictions = predicted_summary[
            predicted_summary["edge_probability"] >= prob_threshold
        ]
    else:
        predictions = predicted_summary

    def _to_set(frame: pd.DataFrame) -> set[tuple]:
        if frame.empty:
            return set()
        return set(map(tuple, frame[["source", "target", "lag"]].to_numpy()))

    predicted_edges = _to_set(predictions)
    true_edges = _to_set(ground_truth)
    matches = predicted_edges & true_edges
    unmatched_predictions = predicted_edges - matches
    unmatched_truth = true_edges - matches

    reversed_edges = {
        edge
        for edge in unmatched_predictions
        if edge[0] != edge[1] and (edge[1], edge[0], edge[2]) in unmatched_truth
    }

    true_positives = len(matches)
    false_positives = len(unmatched_predictions)
    false_negatives = len(unmatched_truth)
    reversals = len(reversed_edges)

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # Uma reversão exige uma operação; sem este ajuste seria contada como FP + FN.
    shd = false_positives + false_negatives - reversals

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "reversed_edges": reversals,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "structural_hamming_distance": shd,
    }


def inject_noise_regime_change(
    df: pd.DataFrame,
    index_change: int = 250,
    noise_multiplier: float = 3.0,
    random_state: int | None = 42,
):
    """Injeta uma mudança no regime de ruído após uma posição temporal."""
    if not 0 <= index_change < len(df):
        raise ValueError("index_change deve estar dentro dos limites do DataFrame.")
    if noise_multiplier < 0.0:
        raise ValueError("noise_multiplier nao pode ser negativo.")

    rng = np.random.default_rng(random_state)
    try:
        noisy = df.astype(float).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError("Todas as colunas devem ser numericas.") from exc

    for column_index in range(noisy.shape[1]):
        values = noisy.iloc[:, column_index].to_numpy()
        noise = rng.normal(
            0,
            np.std(values) * noise_multiplier,
            len(noisy) - index_change,
        )
        noisy.iloc[index_change:, column_index] = values[index_change:] + noise

    return noisy
