import numpy as np
import pandas as pd


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
    """Injeta uma quebra de regime por aumento de ruído após um índice."""
    if not 0 <= index_change < len(df):
        raise ValueError("index_change deve estar dentro dos limites do DataFrame.")

    rng = np.random.default_rng(random_state)
    noisy = df.copy()
    for column in noisy.columns:
        noise = rng.normal(
            0,
            np.std(noisy[column]) * noise_multiplier,
            len(noisy) - index_change,
        )
        noisy.loc[index_change:, column] += noise

    return noisy
