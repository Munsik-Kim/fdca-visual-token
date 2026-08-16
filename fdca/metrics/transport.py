"""Finite-support Wasserstein distance under weighted token Hamming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

from fdca.synthetic.coupled_paths import weighted_hamming
from fdca.synthetic.state_space import State


@dataclass(frozen=True)
class TransportResult:
    distance: float
    success: bool
    status: int
    message: str
    max_marginal_residual: float
    support_left: int
    support_right: int
    variables: int


def exact_wasserstein(
    reference: Mapping[State, float],
    approximate: Mapping[State, float],
    weights: Sequence[float],
    mass_tolerance: float = 1e-12,
) -> TransportResult:
    """Solve the full finite transportation LP with SciPy HiGHS."""

    left_states = tuple(state for state, mass in sorted(reference.items()) if mass > 0.0)
    right_states = tuple(state for state, mass in sorted(approximate.items()) if mass > 0.0)
    left_mass = np.asarray([reference[state] for state in left_states], dtype=np.float64)
    right_mass = np.asarray([approximate[state] for state in right_states], dtype=np.float64)
    if abs(float(left_mass.sum()) - 1.0) > mass_tolerance:
        raise ValueError("reference law is not normalized")
    if abs(float(right_mass.sum()) - 1.0) > mass_tolerance:
        raise ValueError("approximate law is not normalized")
    n_left = len(left_states)
    n_right = len(right_states)
    costs = np.asarray(
        [
            weighted_hamming(left_state, right_state, weights)
            for left_state in left_states
            for right_state in right_states
        ],
        dtype=np.float64,
    )
    constraints = np.zeros((n_left + n_right, n_left * n_right), dtype=np.float64)
    for left_index in range(n_left):
        constraints[left_index, left_index * n_right : (left_index + 1) * n_right] = 1.0
    for right_index in range(n_right):
        constraints[n_left + right_index, right_index::n_right] = 1.0
    targets = np.concatenate((left_mass, right_mass))
    result = linprog(
        costs,
        A_eq=constraints,
        b_eq=targets,
        bounds=(0.0, None),
        method="highs",
        options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"transport LP failed: {result.status} {result.message}")
    residual = float(np.max(np.abs(constraints @ result.x - targets)))
    return TransportResult(
        distance=float(result.fun),
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        max_marginal_residual=residual,
        support_left=n_left,
        support_right=n_right,
        variables=n_left * n_right,
    )
