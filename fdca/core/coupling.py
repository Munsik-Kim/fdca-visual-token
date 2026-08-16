"""Categorical couplings and deterministic event-addressed sampling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _probability_vector(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("categorical law must be a nonempty vector")
    if not np.all(np.isfinite(vector)) or np.min(vector) < -1e-15:
        raise ValueError(f"invalid categorical law {vector}")
    vector = np.maximum(vector, 0.0)
    total = float(vector.sum())
    if not np.isclose(total, 1.0, atol=1e-12, rtol=0.0):
        raise ValueError(f"categorical law sums to {total}, not 1")
    return vector / total


def total_variation(p: Sequence[float], q: Sequence[float]) -> float:
    p_vector = _probability_vector(p)
    q_vector = _probability_vector(q)
    if p_vector.shape != q_vector.shape:
        raise ValueError("categorical laws have different support sizes")
    return float(0.5 * np.abs(p_vector - q_vector).sum())


def conditional_maximal_joint(p: Sequence[float], q: Sequence[float]) -> np.ndarray:
    """Exact joint law of the conditional-form categorical maximal coupling.

    The diagonal contains the common mass. Conditional on rejection, the
    positive residual of p is coupled independently to the positive residual of
    q. The residual supports are disjoint, hence mismatch probability is TV.
    """

    p_vector = _probability_vector(p)
    q_vector = _probability_vector(q)
    if p_vector.shape != q_vector.shape:
        raise ValueError("categorical laws have different support sizes")
    common = np.minimum(p_vector, q_vector)
    residual_p = np.maximum(p_vector - q_vector, 0.0)
    residual_q = np.maximum(q_vector - p_vector, 0.0)
    tv = float(residual_p.sum())
    joint = np.diag(common)
    if tv > 1e-15:
        joint = joint + np.outer(residual_p, residual_q) / tv
    joint[np.abs(joint) < 1e-16] = 0.0
    return joint


def inverse_cdf_crn_joint(p: Sequence[float], q: Sequence[float]) -> np.ndarray:
    """Joint categorical law induced by a shared U(0,1) inverse-CDF draw."""

    p_vector = _probability_vector(p)
    q_vector = _probability_vector(q)
    if p_vector.shape != q_vector.shape:
        raise ValueError("categorical laws have different support sizes")
    p_edges = np.concatenate(([0.0], np.cumsum(p_vector)))
    q_edges = np.concatenate(([0.0], np.cumsum(q_vector)))
    p_edges[-1] = 1.0
    q_edges[-1] = 1.0
    joint = np.zeros((p_vector.size, p_vector.size), dtype=np.float64)
    for i in range(p_vector.size):
        for j in range(q_vector.size):
            overlap = min(p_edges[i + 1], q_edges[j + 1]) - max(p_edges[i], q_edges[j])
            joint[i, j] = max(0.0, float(overlap))
    return joint


def joint_for_name(name: str, p: Sequence[float], q: Sequence[float]) -> np.ndarray:
    if name in {"maximal", "conditional_maximal"}:
        return conditional_maximal_joint(p, q)
    if name in {"crn", "inverse_cdf_crn"}:
        return inverse_cdf_crn_joint(p, q)
    raise ValueError(f"unknown coupling {name!r}")


@dataclass(frozen=True)
class CouplingAudit:
    x_marginal_error: float
    y_marginal_error: float
    normalization_error: float
    mismatch_probability: float
    tv: float


def audit_joint(joint: np.ndarray, p: Sequence[float], q: Sequence[float]) -> CouplingAudit:
    p_vector = _probability_vector(p)
    q_vector = _probability_vector(q)
    matrix = np.asarray(joint, dtype=np.float64)
    if matrix.shape != (p_vector.size, q_vector.size):
        raise ValueError("joint matrix has the wrong shape")
    if np.min(matrix) < -1e-15:
        raise ValueError("joint matrix contains negative mass")
    mismatch = float(matrix.sum() - np.trace(matrix))
    return CouplingAudit(
        x_marginal_error=float(np.max(np.abs(matrix.sum(axis=1) - p_vector))),
        y_marginal_error=float(np.max(np.abs(matrix.sum(axis=0) - q_vector))),
        normalization_error=abs(float(matrix.sum()) - 1.0),
        mismatch_probability=mismatch,
        tv=total_variation(p_vector, q_vector),
    )


def uniform_from_key(*parts: object) -> float:
    """Map a semantic event key to a deterministic uniform in [0,1)."""

    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(1 << 64)


def inverse_cdf_sample(probabilities: Sequence[float], uniform: float) -> int:
    vector = _probability_vector(probabilities)
    if not 0.0 <= uniform < 1.0:
        raise ValueError("uniform must be in [0,1)")
    return int(np.searchsorted(np.cumsum(vector), uniform, side="right"))


def conditional_maximal_sample(
    p: Sequence[float], q: Sequence[float], event_key: tuple[object, ...]
) -> tuple[int, int]:
    """Conditional-form maximal sample with separately addressed random events."""

    p_vector = _probability_vector(p)
    q_vector = _probability_vector(q)
    if p_vector.shape != q_vector.shape:
        raise ValueError("categorical laws have different support sizes")
    x = inverse_cdf_sample(p_vector, uniform_from_key(*event_key, "proposal_x"))
    ratio = q_vector[x] / p_vector[x]
    if uniform_from_key(*event_key, "proposal_accept") <= min(1.0, float(ratio)):
        return x, x
    residual = np.maximum(q_vector - p_vector, 0.0)
    tv = float(residual.sum())
    if tv <= 1e-15:
        return x, x
    y = inverse_cdf_sample(
        residual / tv, uniform_from_key(*event_key, "proposal_resid")
    )
    return x, y
