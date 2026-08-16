"""Exact coordinatewise maximal-coupling arithmetic for P2N."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from fdca.couplings.p1l_conditional_maximal import inverse_cdf, probability_vector, total_variation


@dataclass(frozen=True)
class NaturalDraw:
    x: int
    y: int
    tv: float
    accepted_equal: bool


def maximal_from_uniforms(p: Sequence[float], q: Sequence[float], proposal_u: float, accept_u: float, residual_u: float) -> NaturalDraw:
    left, right = probability_vector(p), probability_vector(q)
    x = inverse_cdf(left, proposal_u)
    ratio = float(right[x] / left[x]) if left[x] > 0 else 0.0
    tv = total_variation(left, right)
    if accept_u <= min(1.0, ratio):
        return NaturalDraw(x, x, tv, True)
    residual = np.maximum(right - left, 0.0)
    mass = float(residual.sum())
    if mass <= 1e-15:
        return NaturalDraw(x, x, tv, True)
    y = inverse_cdf(residual / mass, residual_u)
    return NaturalDraw(x, y, tv, False)


def fixed_x_from_uniforms(p: Sequence[float], q: Sequence[float], x: int, accept_u: float, residual_u: float) -> NaturalDraw:
    left, right = probability_vector(p), probability_vector(q)
    x = int(x)
    if left[x] <= 0:
        raise ValueError("fixed x has zero p mass")
    ratio = float(right[x] / left[x])
    tv = total_variation(left, right)
    if accept_u <= min(1.0, ratio):
        return NaturalDraw(x, x, tv, True)
    residual = np.maximum(right - left, 0.0)
    mass = float(residual.sum())
    if mass <= 1e-15:
        return NaturalDraw(x, x, tv, True)
    return NaturalDraw(x, inverse_cdf(residual / mass, residual_u), tv, False)


def poisson_binomial(tv: Sequence[float]) -> np.ndarray:
    probabilities = np.asarray(tv, dtype=np.float64)
    law = np.array([1.0], dtype=np.float64)
    for value in probabilities:
        law = np.convolve(law, np.array([1.0 - value, value], dtype=np.float64))
    law /= law.sum()
    return law


def exact_seed_summary(tv: Sequence[float]) -> dict[str, float | list[float]]:
    values = np.asarray(tv, dtype=np.float64)
    law = poisson_binomial(values)
    return {
        "split_incidence": float(1.0 - law[0]),
        "expected_seed_count": float(values.sum()),
        "zero_seed_probability": float(law[0]),
        "single_seed_probability": float(law[1]) if len(law) > 1 else 0.0,
        "multi_seed_probability": float(law[2:].sum()) if len(law) > 2 else 0.0,
        "seed_count_law": law.tolist(),
    }


def independent_split_probability(tv: Sequence[float]) -> float:
    product = 1.0
    for value in map(float, tv):
        product *= 1.0 - value
    return 1.0 - product
