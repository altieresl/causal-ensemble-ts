"""Benchmark sintetico com ground truth conhecido + robustez a mudanca no regime de ruido.

Port das celulas 21-23 de Series_Temporais.ipynb: consenso de 2 entre 3 metodos, sobre a serie
limpa e sobre a mesma serie com ruido multiplicado a partir de ``index_change``.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from ..adapters.serialization import frame_records, to_jsonable
from ..domain import DomainError

ProgressFn = Callable[[int, int, str], None]

DEFAULTS: dict[str, Any] = {
    "n_samples": 500,
    "noise_multiplier": 3.0,
    "index_change": 250,
    "n_bootstrap": 20,
    "max_lag": 2,
    "selection_threshold": 0.6,
    "probability_threshold": 0.5,
    "random_state": 42,
    "methods": None,
}


def normalize_params(raw: dict[str, Any]) -> dict[str, Any]:
    params = {**DEFAULTS, **{k: v for k, v in raw.items() if v is not None or k == "methods"}}
    if not 100 <= params["n_samples"] <= 5000:
        raise DomainError("n_samples deve estar entre 100 e 5000.")
    if not 0 <= params["index_change"] < params["n_samples"]:
        raise DomainError("index_change deve estar dentro da serie (0 <= index_change < n_samples).")
    if params["noise_multiplier"] < 0:
        raise DomainError("noise_multiplier nao pode ser negativo.")
    if not 1 <= params["n_bootstrap"] <= 100:
        raise DomainError("n_bootstrap deve estar entre 1 e 100.")
    if not 1 <= params["max_lag"] <= 10:
        raise DomainError("max_lag deve estar entre 1 e 10.")
    if params["methods"] is not None and len(params["methods"]) < 2:
        raise DomainError("Selecione ao menos 2 metodos.")
    return params


def _evaluate(frame: pd.DataFrame, ground_truth: pd.DataFrame, threshold: float) -> dict[str, Any]:
    from causal_discovery.benchmark import compute_structural_metrics

    metrics = compute_structural_metrics(frame, ground_truth, prob_threshold=threshold)
    keys = ["source", "target", "lag"]
    predicted = {tuple(r) for r in frame.loc[frame["edge_probability"] >= threshold, keys].to_numpy()}
    truth = {tuple(r) for r in ground_truth[keys].to_numpy()}
    return {
        "metrics": metrics,
        "edges": frame_records(frame, [*keys, "edge_probability"]),
        "true_positives": sorted(predicted & truth),
        "false_positives": sorted(predicted - truth),
        "false_negatives": sorted(truth - predicted),
    }


def run_benchmark(raw_params: dict[str, Any], progress: ProgressFn) -> dict[str, Any]:
    from causal_discovery import (
        get_registered_method_kwargs,
        get_registered_method_weights,
        get_registered_methods,
    )
    from causal_discovery.benchmark import generate_synthetic_timeseries, inject_noise_regime_change
    from causal_discovery.ensemble_selection import select_robust_ensemble_combination

    params = normalize_params(raw_params)
    registered = get_registered_methods()
    names = list(params["methods"] or registered)
    unknown = sorted(set(names) - set(registered))
    if unknown:
        raise DomainError(f"Metodos desconhecidos: {unknown}")
    methods = {name: registered[name] for name in names}
    kwargs = {name: dict(k) for name, k in get_registered_method_kwargs(params["max_lag"]).items() if name in methods}
    if "NeuralGrangercMLP" in kwargs:
        kwargs["NeuralGrangercMLP"]["max_iter"] = 200  # mais leve, so no benchmark auxiliar
    weights = {n: w for n, w in get_registered_method_weights().items() if n in methods}
    size = min(3, len(names))  # consenso de 2 entre 3 metodos

    df, truth = generate_synthetic_timeseries(n_samples=params["n_samples"], random_state=params["random_state"])
    truth = truth.loc[truth["source"].ne(truth["target"])].reset_index(drop=True)

    def select(frame: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
        selection = select_robust_ensemble_combination(
            frame, methods, method_kwargs=kwargs, method_weights=weights,
            min_methods=size, max_methods=size, min_votes=2 if size >= 2 else 1,
            n_bootstrap=params["n_bootstrap"],
            selection_probability_threshold=params["selection_threshold"],
            random_state=params["random_state"],
        )
        summary = selection["best_evaluation"]["probabilistic_summary"]
        return summary.loc[summary["source"].ne(summary["target"])].reset_index(drop=True), selection["best_combination"]

    noisy = inject_noise_regime_change(
        df, index_change=params["index_change"], noise_multiplier=params["noise_multiplier"],
        random_state=params["random_state"],
    )
    # As duas selecoes sao independentes (series diferentes, sem estado compartilhado): em paralelo.
    done = {"count": 0}

    def run_one(frame: pd.DataFrame, label: str):
        outcome = select(frame)
        done["count"] += 1
        progress(done["count"], 2, f"Concluida: {label}")
        return outcome

    progress(0, 2, "Series limpa e ruidosa em paralelo")
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="bench") as pool:
        clean_future = pool.submit(run_one, df, "serie limpa")
        noisy_future = pool.submit(run_one, noisy, "serie com ruido severo")
        clean_summary, clean_combo = clean_future.result()
        noisy_summary, noisy_combo = noisy_future.result()

    threshold = params["probability_threshold"]
    clean = _evaluate(clean_summary, truth, threshold)
    corrupted = _evaluate(noisy_summary, truth, threshold)
    return to_jsonable(
        {
            "params": params,
            "ground_truth": frame_records(truth, ["source", "target", "lag"]),
            "clean": {**clean, "best_combination": clean_combo},
            "noisy": {**corrupted, "best_combination": noisy_combo},
            "delta": {
                key: corrupted["metrics"][key] - clean["metrics"][key]
                for key in ("f1_score", "structural_hamming_distance")
                if key in clean["metrics"] and key in corrupted["metrics"]
            },
        }
    )
