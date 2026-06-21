from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor

from ..types import canonical_links_to_dataframe
from ..utils import build_target_dataset, parse_lagged_name, validate_numeric_dataframe


class _CMLPNetwork(nn.Module):
    def __init__(
        self,
        num_series: int,
        lag: int,
        hidden: tuple[int, ...],
        activation: str,
    ):
        super().__init__()
        if not hidden:
            raise ValueError("hidden_layer_sizes nao pode ser vazio.")
        layers: list[nn.Conv1d] = [nn.Conv1d(num_series, hidden[0], lag)]
        layers.extend(
            nn.Conv1d(input_size, output_size, 1)
            for input_size, output_size in zip(hidden, (*hidden[1:], 1))
        )
        self.layers = nn.ModuleList(layers)
        self.activation = nn.ReLU() if activation == "relu" else nn.Tanh()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = values.transpose(2, 1)
        for index, layer in enumerate(self.layers):
            if index:
                output = self.activation(output)
            output = layer(output)
        return output.transpose(2, 1)


class _CMLP(nn.Module):
    def __init__(
        self,
        num_series: int,
        lag: int,
        hidden: tuple[int, ...],
        activation: str,
    ):
        super().__init__()
        self.num_series = num_series
        self.lag = lag
        self.networks = nn.ModuleList(
            _CMLPNetwork(num_series, lag, hidden, activation)
            for _ in range(num_series)
        )

    def group_norms(self) -> torch.Tensor:
        return torch.stack(
            [
                torch.linalg.vector_norm(network.layers[0].weight, dim=(0, 2))
                for network in self.networks
            ]
        )

    def lag_norms(self) -> torch.Tensor:
        return torch.stack(
            [
                torch.linalg.vector_norm(network.layers[0].weight, dim=0)
                for network in self.networks
            ]
        )


def _structured_prox(
    network: _CMLPNetwork,
    amount: float,
    penalty: str,
) -> None:
    weights = network.layers[0].weight
    if penalty == "GL":
        norms = torch.linalg.vector_norm(weights, dim=(0, 2), keepdim=True)
        scale = torch.clamp(norms - amount, min=0.0) / torch.clamp(norms, min=amount)
        weights.data.mul_(scale)
    elif penalty == "GSGL":
        lag_norms = torch.linalg.vector_norm(weights, dim=0, keepdim=True)
        lag_scale = torch.clamp(lag_norms - amount, min=0.0) / torch.clamp(
            lag_norms, min=amount
        )
        weights.data.mul_(lag_scale)
        group_norms = torch.linalg.vector_norm(weights, dim=(0, 2), keepdim=True)
        group_scale = torch.clamp(group_norms - amount, min=0.0) / torch.clamp(
            group_norms, min=amount
        )
        weights.data.mul_(group_scale)
    elif penalty == "H":
        for index in range(weights.shape[2]):
            prefix = weights[:, :, : index + 1]
            norms = torch.linalg.vector_norm(prefix, dim=(0, 2), keepdim=True)
            scale = torch.clamp(norms - amount, min=0.0) / torch.clamp(
                norms, min=amount
            )
            weights.data[:, :, : index + 1].mul_(scale)
    else:
        raise ValueError("penalty deve ser 'GL', 'GSGL' ou 'H'.")


def _structured_penalty(
    network: _CMLPNetwork,
    coefficient: float,
    penalty: str,
) -> torch.Tensor:
    weights = network.layers[0].weight
    if penalty == "GL":
        value = torch.sum(torch.linalg.vector_norm(weights, dim=(0, 2)))
    elif penalty == "GSGL":
        value = torch.sum(torch.linalg.vector_norm(weights, dim=(0, 2)))
        value = value + torch.sum(torch.linalg.vector_norm(weights, dim=0))
    elif penalty == "H":
        value = sum(
            torch.sum(
                torch.linalg.vector_norm(
                    weights[:, :, : index + 1],
                    dim=(0, 2),
                )
            )
            for index in range(weights.shape[2])
        )
    else:
        raise ValueError("penalty deve ser 'GL', 'GSGL' ou 'H'.")
    return coefficient * value


def _ridge_penalty(network: _CMLPNetwork, coefficient: float) -> torch.Tensor:
    return coefficient * sum(
        torch.sum(layer.weight**2) for layer in network.layers[1:]
    )


def _fit_cmlp_ista(
    model: _CMLP,
    values: torch.Tensor,
    *,
    lambda_group: float,
    lambda_ridge: float,
    penalty: str,
    learning_rate: float,
    max_iter: int,
    check_every: int,
    patience: int,
    tolerance: float,
) -> tuple[float, int]:
    loss_fn = nn.MSELoss(reduction="mean")
    best_loss = float("inf")
    best_model = deepcopy(model.state_dict())
    checks_without_improvement = 0
    final_loss = float("inf")

    for iteration in range(1, max_iter + 1):
        smooth_loss = torch.zeros((), dtype=values.dtype, device=values.device)
        for target_index, network in enumerate(model.networks):
            prediction = network(values[:, :-1])
            target = values[:, model.lag :, target_index : target_index + 1]
            smooth_loss = smooth_loss + loss_fn(prediction, target)
            smooth_loss = smooth_loss + _ridge_penalty(network, lambda_ridge)

        if not torch.isfinite(smooth_loss):
            break
        smooth_loss.backward()

        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-learning_rate)
            if lambda_group > 0.0:
                for network in model.networks:
                    _structured_prox(
                        network,
                        amount=learning_rate * lambda_group,
                        penalty=penalty,
                    )
        model.zero_grad(set_to_none=True)

        if iteration % check_every == 0 or iteration == max_iter:
            with torch.no_grad():
                checked_smooth = torch.zeros(
                    (),
                    dtype=values.dtype,
                    device=values.device,
                )
                for target_index, network in enumerate(model.networks):
                    prediction = network(values[:, :-1])
                    target = values[
                        :, model.lag :, target_index : target_index + 1
                    ]
                    checked_smooth = checked_smooth + loss_fn(prediction, target)
                    checked_smooth = checked_smooth + _ridge_penalty(
                        network,
                        lambda_ridge,
                    )
                nonsmooth = sum(
                    _structured_penalty(network, lambda_group, penalty)
                    for network in model.networks
                )
                final_loss = float((checked_smooth + nonsmooth).item())

            if final_loss < best_loss - tolerance:
                best_loss = final_loss
                best_model = deepcopy(model.state_dict())
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
                if checks_without_improvement >= patience:
                    break

    model.load_state_dict(best_model)
    return best_loss, iteration


def run_mlp_temporal_importance(
    data: pd.DataFrame,
    max_lag: int,
    *,
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    max_iter: int = 200,
    perm_repeats: int = 10,
    score_threshold: float = 0.0,
    test_fraction: float = 0.2,
) -> pd.DataFrame:
    """Estimate nonlinear temporal predictive links with an MLP heuristic.

    This is not the published cMLP/cLSTM Neural Granger algorithm of Tank
    et al. (2021), which uses structured group sparsity. Here, chronological
    holdout permutation importance is used as an exploratory approximation.
    """
    validated = validate_numeric_dataframe(data, min_rows=max_lag + 5)
    if not 0.0 < test_fraction < 0.5:
        raise ValueError("test_fraction deve estar entre 0 e 0.5.")
    records: list[dict] = []

    for target_name in validated.columns:
        predictors, target = build_target_dataset(validated, target_name, max_lag)
        if predictors.empty or target.empty:
            continue

        split_index = int(len(predictors) * (1.0 - test_fraction))
        if split_index < 5 or len(predictors) - split_index < 3:
            continue
        train_x = predictors.iloc[:split_index]
        test_x = predictors.iloc[split_index:]
        train_y = target.iloc[:split_index]
        test_y = target.iloc[split_index:]

        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42,
        )

        try:
            model.fit(train_x, train_y)
        except Exception:
            continue

        importance = permutation_importance(
            model,
            test_x,
            test_y,
            n_repeats=perm_repeats,
            random_state=42,
            n_jobs=-1,
        )

        for feature_name, score in zip(predictors.columns, importance.importances_mean):
            if score <= score_threshold:
                continue

            parsed = parse_lagged_name(feature_name)
            if parsed is None:
                continue

            source_name, lag = parsed
            records.append(
                {
                    "source": source_name,
                    "target": target_name,
                    "lag": lag,
                    "score": float(score),
                    "p_value": np.nan,
                    "method": "MLPPermutationHeuristic",
                    "implementation": "mlp_holdout_permutation_heuristic",
                }
            )

    return canonical_links_to_dataframe(records)


def run_neural_granger(
    data: pd.DataFrame,
    max_lag: int,
    *,
    hidden_layer_sizes: tuple[int, ...] = (64,),
    activation: str = "relu",
    lambda_group: float = 0.01,
    lambda_ridge: float = 0.01,
    penalty: str = "H",
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    check_every: int = 100,
    patience: int = 5,
    tolerance: float = 1e-5,
    min_group_norm: float = 1e-6,
    min_lag_norm: float = 1e-6,
    standardize: bool = True,
    random_state: int = 42,
    device: str = "cpu",
) -> pd.DataFrame:
    """Run the canonical cMLP Neural Granger model with group lasso.

    The architecture and proximal group-lasso training follow Tank et al.
    (2021) and the authors' Neural-GC reference implementation.
    """
    if max_lag < 1:
        return canonical_links_to_dataframe([])
    if activation not in {"relu", "tanh"}:
        raise ValueError("activation deve ser 'relu' ou 'tanh'.")
    if penalty not in {"GL", "GSGL", "H"}:
        raise ValueError("penalty deve ser 'GL', 'GSGL' ou 'H'.")
    if max_iter <= 0 or check_every <= 0:
        raise ValueError("max_iter e check_every devem ser maiores que zero.")

    validated = validate_numeric_dataframe(data, min_rows=max_lag + 10)
    values = validated.to_numpy(dtype=np.float32)
    if standardize:
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        std[std == 0.0] = 1.0
        values = (values - mean) / std

    torch.manual_seed(random_state)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA solicitado, mas nao esta disponivel.")
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device).unsqueeze(0)
    model = _CMLP(
        num_series=values.shape[1],
        lag=max_lag,
        hidden=hidden_layer_sizes,
        activation=activation,
    ).to(device)
    training_loss, iterations = _fit_cmlp_ista(
        model,
        tensor,
        lambda_group=lambda_group,
        lambda_ridge=lambda_ridge,
        penalty=penalty,
        learning_rate=learning_rate,
        max_iter=max_iter,
        check_every=check_every,
        patience=patience,
        tolerance=tolerance,
    )

    group_norms = model.group_norms().detach().cpu().numpy()
    lag_norms = model.lag_norms().detach().cpu().numpy()
    names = validated.columns.tolist()
    records: list[dict] = []

    for target_index, target in enumerate(names):
        for source_index, source in enumerate(names):
            group_norm = float(group_norms[target_index, source_index])
            if group_norm <= min_group_norm:
                continue
            for kernel_index in range(max_lag):
                lag_score = float(lag_norms[target_index, source_index, kernel_index])
                if lag_score <= min_lag_norm:
                    continue
                records.append(
                    {
                        "source": source,
                        "target": target,
                        "lag": max_lag - kernel_index,
                        "score": lag_score,
                        "p_value": np.nan,
                        "method": "NeuralGrangercMLP",
                        "group_norm": group_norm,
                        "penalty": penalty,
                        "lambda_group": float(lambda_group),
                        "lambda_ridge": float(lambda_ridge),
                        "training_loss": float(training_loss),
                        "training_iterations": int(iterations),
                    }
                )

    return canonical_links_to_dataframe(records)
