"""Clipped finite-DAG and linear nilpotent resolvent envelopes."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from fdca.synthetic.models import SyntheticDecoder


def _vector(values: Sequence[float], n: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (n,):
        raise ValueError(f"expected vector shape {(n,)}, got {vector.shape}")
    return vector


def clipped_envelope(
    model: SyntheticDecoder, source: Sequence[float], matrix: np.ndarray
) -> np.ndarray:
    source_vector = _vector(source, model.n_positions)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (model.n_positions, model.n_positions):
        raise ValueError("matrix shape mismatch")
    envelope = np.zeros(model.n_positions, dtype=np.float64)
    for position in model.topological_positions:
        envelope[position] = min(
            1.0,
            float(source_vector[position] + np.dot(matrix[position], envelope)),
        )
    return envelope


def conditional_envelope(
    model: SyntheticDecoder,
    source: Sequence[float],
    matrix: np.ndarray,
    split_round: int,
    seed: Iterable[int],
) -> np.ndarray:
    """First-split envelope with fixed boundary and future source terms."""

    if not 0 <= split_round < len(model.schedule):
        raise ValueError("split round outside schedule")
    source_vector = _vector(source, model.n_positions)
    matrix = np.asarray(matrix, dtype=np.float64)
    seed_set = frozenset(seed)
    current_batch = set(model.schedule[split_round])
    if not seed_set or not seed_set.issubset(current_batch):
        raise ValueError("seed must be a nonempty subset of the split batch")
    envelope = np.zeros(model.n_positions, dtype=np.float64)
    for position in seed_set:
        envelope[position] = 1.0
    for round_index, batch in enumerate(model.schedule):
        if round_index <= split_round:
            continue
        for position in batch:
            envelope[position] = min(
                1.0,
                float(source_vector[position] + np.dot(matrix[position], envelope)),
            )
    return envelope


def finite_resolvent_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    n = matrix.shape[0]
    result = np.eye(n, dtype=np.float64)
    power = np.eye(n, dtype=np.float64)
    for _ in range(1, n):
        power = power @ matrix
        result = result + power
    return result


def inverse_resolvent_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.linalg.inv(np.eye(matrix.shape[0], dtype=np.float64) - matrix)


def linear_envelope(matrix: np.ndarray, source: Sequence[float]) -> np.ndarray:
    source_vector = np.asarray(source, dtype=np.float64)
    return inverse_resolvent_matrix(matrix) @ source_vector


def recursion_rhs(source: Sequence[float], matrix: np.ndarray, profile: Sequence[float]) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) + np.asarray(matrix, dtype=np.float64) @ np.asarray(
        profile, dtype=np.float64
    )
