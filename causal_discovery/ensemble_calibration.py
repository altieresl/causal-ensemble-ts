from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _validate_score_tables(
    score_tables: Sequence[pd.DataFrame],
    targets: Sequence[Sequence[int] | np.ndarray],
) -> tuple[list[pd.DataFrame], list[np.ndarray], list[str]]:
    tables = [table.copy() for table in score_tables]
    labels = [np.asarray(target, dtype=int).reshape(-1) for target in targets]
    if not tables or len(tables) != len(labels):
        raise ValueError("score_tables e targets devem ter o mesmo tamanho nao nulo.")

    candidate_names = [str(column) for column in tables[0].columns]
    if not candidate_names:
        raise ValueError("Ao menos um candidato deve ser informado.")
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("Os nomes dos candidatos devem ser unicos.")

    for table, target in zip(tables, labels):
        if [str(column) for column in table.columns] != candidate_names:
            raise ValueError("Todas as tabelas devem ter os mesmos candidatos e ordem.")
        if len(table) != len(target):
            raise ValueError("Cada tabela deve ter uma classe por linha.")
        values = table.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Os escores dos candidatos devem ser finitos.")
        if not set(np.unique(target)).issubset({0, 1}):
            raise ValueError("targets deve conter apenas classes binarias 0/1.")
        if np.unique(target).size < 2:
            raise ValueError("Cada trajetória deve conter exemplos positivos e negativos.")
    return tables, labels, candidate_names


def _mean_average_precision(
    scores: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
) -> float:
    return float(
        np.mean(
            [
                average_precision_score(target, score)
                for target, score in zip(targets, scores)
            ]
        )
    )


def _greedy_path(
    score_tables: Sequence[pd.DataFrame],
    targets: Sequence[np.ndarray],
    *,
    max_members: int,
) -> list[dict[str, Any]]:
    candidate_names = [str(column) for column in score_tables[0].columns]
    running_sums = [np.zeros(len(table), dtype=float) for table in score_tables]
    counts = {name: 0 for name in candidate_names}
    path: list[dict[str, Any]] = []

    for member_count in range(1, max_members + 1):
        candidates: list[tuple[float, str, list[np.ndarray]]] = []
        for candidate_index, candidate_name in enumerate(candidate_names):
            combined = [
                (running + table.iloc[:, candidate_index].to_numpy(dtype=float))
                / member_count
                for running, table in zip(running_sums, score_tables)
            ]
            candidates.append(
                (
                    _mean_average_precision(combined, targets),
                    candidate_name,
                    combined,
                )
            )

        # O nome resolve empates de forma deterministica sem consultar o teste.
        best_score, best_name, _ = sorted(
            candidates, key=lambda item: (-item[0], item[1])
        )[0]
        best_index = candidate_names.index(best_name)
        for trajectory_index, table in enumerate(score_tables):
            running_sums[trajectory_index] += table.iloc[:, best_index].to_numpy(
                dtype=float
            )
        counts[best_name] += 1
        path.append(
            {
                "member_count": member_count,
                "selected_candidate": best_name,
                "mean_average_precision": best_score,
                "weights": {
                    name: counts[name] / member_count for name in candidate_names
                },
            }
        )
    return path


def combine_candidate_scores(
    score_table: pd.DataFrame,
    weights: Mapping[str, float],
) -> np.ndarray:
    """Combina candidatos por uma media convexa nao negativa."""
    normalized = {
        str(name): max(float(weight), 0.0) for name, weight in weights.items()
    }
    total = sum(normalized.values())
    if total <= 0.0:
        raise ValueError("Ao menos um peso deve ser positivo.")
    missing = sorted(set(normalized) - set(map(str, score_table.columns)))
    if missing:
        raise ValueError("Candidatos ausentes: " + ", ".join(missing))
    result = np.zeros(len(score_table), dtype=float)
    for name, weight in normalized.items():
        result += score_table[name].to_numpy(dtype=float) * weight / total
    return result


def apply_calibrated_pair_ensemble(
    summary: pd.DataFrame,
    pair_candidate_scores: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    selected_pair_count: int,
    score_column: str = "ensemble_score",
    selection_column: str = "ensemble_selected",
) -> pd.DataFrame:
    """Aplica pesos congelados e um corte top-k no nivel do par nao orientado.

    ``pair_candidate_scores`` deve ter ``source``, ``target`` e uma coluna por
    candidato. A funcao nao consulta classes verdadeiras; pesos e ``k`` devem
    ter sido calibrados apenas no desenvolvimento.
    """
    frame = summary.copy()
    required = {"source", "target"}
    if not required.issubset(pair_candidate_scores.columns):
        raise ValueError("pair_candidate_scores requer source e target.")
    if frame.empty:
        frame["pre_calibration_ensemble_score"] = pd.Series(dtype=float)
        frame[selection_column] = pd.Series(dtype=bool)
        return frame

    candidate_columns = [str(name) for name in weights]
    missing = sorted(set(candidate_columns) - set(pair_candidate_scores.columns))
    if missing:
        raise ValueError("Candidatos ausentes: " + ", ".join(missing))
    pair_table = pair_candidate_scores.copy()
    pair_table["pair_key"] = [
        tuple(sorted((str(source), str(target))))
        for source, target in zip(pair_table["source"], pair_table["target"])
    ]
    pair_table = pair_table.drop_duplicates("pair_key", keep="first").reset_index(
        drop=True
    )
    pair_table["calibrated_pair_score"] = combine_candidate_scores(
        pair_table[candidate_columns], weights
    )
    pair_table = pair_table.sort_values(
        ["calibrated_pair_score", "pair_key"], ascending=[False, True]
    ).reset_index(drop=True)
    pair_table["calibrated_pair_rank"] = np.arange(1, len(pair_table) + 1)
    selected_pair_count = int(np.clip(selected_pair_count, 0, len(pair_table)))
    selected_pairs = set(pair_table.head(selected_pair_count)["pair_key"])
    score_lookup = dict(
        zip(pair_table["pair_key"], pair_table["calibrated_pair_score"])
    )
    rank_lookup = dict(zip(pair_table["pair_key"], pair_table["calibrated_pair_rank"]))
    frame_keys = [
        tuple(sorted((str(source), str(target))))
        for source, target in zip(frame["source"], frame["target"])
    ]
    existing_score = (
        frame[score_column]
        if score_column in frame
        else pd.Series(0.0, index=frame.index, dtype=float)
    )
    frame["pre_calibration_ensemble_score"] = pd.to_numeric(
        existing_score, errors="coerce"
    ).fillna(0.0)
    frame[score_column] = [score_lookup.get(key, 0.0) for key in frame_keys]
    frame[selection_column] = [key in selected_pairs for key in frame_keys]
    frame["calibrated_pair_rank"] = pd.array(
        [rank_lookup.get(key, pd.NA) for key in frame_keys], dtype="Int64"
    )
    return frame


def fit_cross_validated_greedy_ensemble(
    score_tables: Sequence[pd.DataFrame],
    targets: Sequence[Sequence[int] | np.ndarray],
    *,
    max_members: int = 20,
) -> dict[str, Any]:
    """Aprende pesos por ensemble selection agrupado por trajetória.

    A biblioteca e combinada por selecao gulosa com reposicao. O numero de
    membros e escolhido por leave-one-trajectory-out, impedindo que a mesma
    trajetória escolha e avalie a complexidade do ensemble. O ajuste requer
    previsoes de desenvolvimento com ground truth; nao deve ser executado no
    conjunto reservado para avaliacao final.

    A regra segue o ensemble selection de Caruana et al. (ICML 2004,
    doi:10.1145/1015330.1015432), usando Average Precision por causa do
    desbalanceamento estrutural.
    """
    tables, labels, candidate_names = _validate_score_tables(score_tables, targets)
    max_members = max(1, int(max_members))
    if len(tables) < 2:
        raise ValueError("Sao necessarias ao menos duas trajetórias para validacao.")

    fold_rows: list[dict[str, Any]] = []
    for held_out_index in range(len(tables)):
        training_tables = [
            table for index, table in enumerate(tables) if index != held_out_index
        ]
        training_targets = [
            target for index, target in enumerate(labels) if index != held_out_index
        ]
        path = _greedy_path(
            training_tables,
            training_targets,
            max_members=max_members,
        )
        for path_entry in path:
            held_out_scores = combine_candidate_scores(
                tables[held_out_index], path_entry["weights"]
            )
            fold_rows.append(
                {
                    "held_out_index": held_out_index,
                    "member_count": path_entry["member_count"],
                    "average_precision": average_precision_score(
                        labels[held_out_index], held_out_scores
                    ),
                }
            )

    cross_validation = pd.DataFrame(fold_rows)
    cv_summary = (
        cross_validation.groupby("member_count", as_index=False)
        .agg(
            mean_average_precision=("average_precision", "mean"),
            std_average_precision=("average_precision", "std"),
        )
        .sort_values(
            ["mean_average_precision", "member_count"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
    selected_size = int(cv_summary.iloc[0]["member_count"])
    final_path = _greedy_path(tables, labels, max_members=selected_size)
    final_entry = final_path[-1]
    return {
        "candidate_names": candidate_names,
        "selected_size": selected_size,
        "weights": final_entry["weights"],
        "development_average_precision": final_entry["mean_average_precision"],
        "cross_validation": cross_validation,
        "cross_validation_summary": cv_summary,
        "selection_path": pd.DataFrame(final_path),
    }


def calibrate_top_k_by_f1(
    scores: Sequence[Sequence[float] | np.ndarray],
    targets: Sequence[Sequence[int] | np.ndarray],
    *,
    minimum_k: int = 1,
    maximum_k: int | None = None,
    minimum_precision: float | None = None,
    minimum_recall: float | None = None,
    maximum_structural_hamming_distance: float | None = None,
    fallback_k: int | None = None,
) -> dict[str, Any]:
    """Calibra a decisao binaria para maximizar F1 no desenvolvimento.

    O limiar e expresso como quantidade ranqueada para ser comparavel entre
    trajetórias cujas escalas de escore diferem. Em empates de F1, a estrutura
    menor e escolhida. Limites opcionais permitem maximizar F1 sem degradar
    outras funcoes da matriz de confusao, seguindo o paradigma de classificacao
    com perdas complexas e restricoes de Narasimhan (AISTATS 2018,
    https://proceedings.mlr.press/v84/narasimhan18a.html). A avaliacao final
    deve permanecer separada.
    """
    score_arrays = [np.asarray(score, dtype=float).reshape(-1) for score in scores]
    label_arrays = [np.asarray(target, dtype=int).reshape(-1) for target in targets]
    if not score_arrays or len(score_arrays) != len(label_arrays):
        raise ValueError("scores e targets devem ter o mesmo tamanho nao nulo.")
    pair_count = len(score_arrays[0])
    if any(len(score) != pair_count for score in score_arrays):
        raise ValueError("Todos os vetores de escore devem ter o mesmo tamanho.")
    if any(len(label) != pair_count for label in label_arrays):
        raise ValueError("Cada vetor de classes deve acompanhar seus escores.")
    minimum_k = max(1, int(minimum_k))
    maximum_k = pair_count if maximum_k is None else min(int(maximum_k), pair_count)
    if minimum_k > maximum_k:
        raise ValueError("Intervalo de k invalido.")

    rows: list[dict[str, float | int]] = []
    for k in range(minimum_k, maximum_k + 1):
        trajectory_metrics = []
        for score, target in zip(score_arrays, label_arrays):
            selected = np.zeros(pair_count, dtype=int)
            selected[np.argsort(score, kind="stable")[-k:]] = 1
            true_positives = int(np.sum((selected == 1) & (target == 1)))
            false_positives = int(np.sum((selected == 1) & (target == 0)))
            false_negatives = int(np.sum((selected == 0) & (target == 1)))
            precision = true_positives / max(true_positives + false_positives, 1)
            recall = true_positives / max(true_positives + false_negatives, 1)
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0.0
                else 0.0
            )
            trajectory_metrics.append(
                (precision, recall, f1, false_positives + false_negatives)
            )
        metric_array = np.asarray(trajectory_metrics, dtype=float)
        rows.append(
            {
                "k": k,
                "precision": float(metric_array[:, 0].mean()),
                "recall": float(metric_array[:, 1].mean()),
                "f1_score": float(metric_array[:, 2].mean()),
                "structural_hamming_distance": float(metric_array[:, 3].mean()),
            }
        )
    all_results = pd.DataFrame(rows)
    feasible = pd.Series(True, index=all_results.index)
    if minimum_precision is not None:
        feasible &= all_results["precision"] >= float(minimum_precision) - 1e-12
    if minimum_recall is not None:
        feasible &= all_results["recall"] >= float(minimum_recall) - 1e-12
    if maximum_structural_hamming_distance is not None:
        feasible &= all_results["structural_hamming_distance"] <= (
            float(maximum_structural_hamming_distance) + 1e-12
        )
    all_results["constraints_satisfied"] = feasible
    feasible_results = all_results.loc[feasible]
    used_fallback = feasible_results.empty
    if used_fallback:
        if fallback_k is None:
            raise ValueError("Nenhum corte satisfaz as restricoes informadas.")
        fallback_k = int(fallback_k)
        feasible_results = all_results.loc[all_results["k"].eq(fallback_k)]
        if feasible_results.empty:
            raise ValueError("fallback_k esta fora do intervalo avaliado.")
    table = feasible_results.sort_values(
        ["f1_score", "k"], ascending=[False, True]
    ).reset_index(drop=True)
    ordered_results = pd.concat(
        [
            table,
            all_results.loc[~all_results["k"].isin(table["k"])].sort_values(
                ["constraints_satisfied", "f1_score", "k"],
                ascending=[False, False, True],
            ),
        ],
        ignore_index=True,
    )
    return {
        "selected_k": int(table.iloc[0]["k"]),
        "constraints_satisfied": not used_fallback,
        "used_fallback": used_fallback,
        "results": ordered_results,
    }
