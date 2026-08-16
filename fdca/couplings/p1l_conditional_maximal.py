"""Conditional-form coordinatewise categorical maximal coupling for P1L."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fdca.rng.p1l_event_addressed import EventKey, EventRegistry


def probability_vector(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("categorical law must be a nonempty vector")
    if not np.all(np.isfinite(vector)) or np.min(vector) < -1e-12:
        raise ValueError("invalid categorical probabilities")
    vector = np.maximum(vector, 0.0)
    total = float(vector.sum())
    if total <= 0.0 or not np.isclose(total, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"categorical law sums to {total}")
    return vector / total


def inverse_cdf(probabilities: Sequence[float], uniform: float) -> int:
    if not 0.0 <= uniform < 1.0:
        raise ValueError(uniform)
    vector = probability_vector(probabilities)
    return min(int(np.searchsorted(np.cumsum(vector), uniform, side="right")), len(vector) - 1)


def total_variation(p: Sequence[float], q: Sequence[float]) -> float:
    left, right = probability_vector(p), probability_vector(q)
    if left.shape != right.shape:
        raise ValueError("support mismatch")
    return float(np.abs(left - right).sum() / 2.0)


def joint_law(p: Sequence[float], q: Sequence[float]) -> np.ndarray:
    left, right = probability_vector(p), probability_vector(q)
    common = np.minimum(left, right)
    rp = np.maximum(left - right, 0.0)
    rq = np.maximum(right - left, 0.0)
    tv = float(rp.sum())
    joint = np.diag(common)
    if tv > 1e-15:
        joint += np.outer(rp, rq) / tv
    return joint


@dataclass(frozen=True)
class CoupledDraw:
    x: int
    y: int
    accepted: bool
    tv: float
    p_sum_error: float
    q_sum_error: float


def conditional_maximal_from_fixed_x(
    p: Sequence[float],
    q: Sequence[float],
    x: int,
    accept_key: EventKey,
    residual_key: EventKey,
    registry: EventRegistry,
) -> CoupledDraw:
    """Draw Y conditionally on an already-addressed canonical X draw."""

    left, right = probability_vector(p), probability_vector(q)
    if left.shape != right.shape or not 0 <= int(x) < len(left):
        raise ValueError("support/x mismatch")
    if left[int(x)] <= 0.0:
        raise ValueError("fixed X has zero probability under p")
    accept_u = registry.uniform(accept_key)
    ratio = float(right[int(x)] / left[int(x)])
    tv = total_variation(left, right)
    if accept_u <= min(1.0, ratio):
        return CoupledDraw(int(x), int(x), True, tv, abs(float(left.sum()) - 1.0), abs(float(right.sum()) - 1.0))
    residual = np.maximum(right - left, 0.0)
    mass = float(residual.sum())
    if mass <= 1e-15:
        return CoupledDraw(int(x), int(x), True, tv, abs(float(left.sum()) - 1.0), abs(float(right.sum()) - 1.0))
    y = inverse_cdf(residual / mass, registry.uniform(residual_key))
    return CoupledDraw(int(x), int(y), False, tv, abs(float(left.sum()) - 1.0), abs(float(right.sum()) - 1.0))


def conditional_maximal_sample(
    p: Sequence[float],
    q: Sequence[float],
    proposal_key: EventKey,
    accept_key: EventKey,
    residual_key: EventKey,
    registry: EventRegistry,
) -> CoupledDraw:
    left = probability_vector(p)
    x = inverse_cdf(left, registry.uniform(proposal_key))
    return conditional_maximal_from_fixed_x(left, q, x, accept_key, residual_key, registry)
