from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


def _clip_probability(value: float, *, eps: float = 1e-12) -> float:
    return float(min(max(value, eps), 1.0 - eps))


def combine_p_values_fisher(p_values: pd.Series) -> float:
    numeric = pd.to_numeric(p_values, errors="coerce").dropna()
    numeric = numeric[(numeric > 0.0) & (numeric <= 1.0)]
    if numeric.empty:
        return float("nan")

    clipped = numeric.clip(1e-16, 1.0)
    statistic = float(-2.0 * np.log(clipped).sum())
    dof = 2 * len(clipped)
    return float(chi2.sf(statistic, dof))


def bayes_factor_from_p_value(p_value: float) -> float:
    if not np.isfinite(p_value):
        return float("nan")

    clipped = _clip_probability(float(p_value), eps=1e-16)
    if clipped >= (1.0 / math.e):
        return 1.0

    bf01 = -math.e * clipped * math.log(clipped)
    if bf01 <= 0.0 or not np.isfinite(bf01):
        return float("nan")

    return float(1.0 / bf01)


def posterior_probability_from_bayes_factor(
    bayes_factor_10: float,
    *,
    prior_edge_probability: float = 0.1,
) -> float:
    prior = _clip_probability(prior_edge_probability)
    if not np.isfinite(bayes_factor_10):
        return prior

    prior_odds = prior / (1.0 - prior)
    posterior_odds = prior_odds * max(float(bayes_factor_10), 1e-16)
    posterior = posterior_odds / (1.0 + posterior_odds)
    return _clip_probability(float(posterior))


def score_to_probability(score: float, *, scale: float = 1.0) -> float:
    if not np.isfinite(score):
        return 0.5

    scale = max(float(scale), 1e-12)
    magnitude = abs(float(score)) / scale
    return _clip_probability(float(1.0 / (1.0 + math.exp(-magnitude))))


def wilson_support_interval(
    votes: int,
    total_methods: int,
    *,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    if total_methods <= 0:
        return 0.0, 0.0

    p_hat = votes / total_methods
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    denominator = 1.0 + (z**2 / total_methods)
    center = (p_hat + z**2 / (2.0 * total_methods)) / denominator
    margin = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) / total_methods) + (z**2 / (4.0 * total_methods**2)))
        / denominator
    )
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return float(low), float(high)
