"""Conditional first-divergence consequence accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fdca.core.dag_envelope import conditional_envelope
from fdca.synthetic.coupled_paths import (
    CoupledPath,
    conditional_paths,
    coordinate_mismatch_profile,
    normalized_weights,
    paired_cost,
)
from fdca.synthetic.models import SyntheticDecoder


@dataclass(frozen=True)
class ConditionalConsequence:
    split_round: int
    seed: frozenset[int]
    event_mass: float
    mismatch_profile: np.ndarray
    paired_cost: float
    persistent_envelope: np.ndarray
    seed_only_envelope: np.ndarray
    persistent_cost_bound: float
    seed_only_cost_bound: float


def evaluate_conditional_consequence(
    model: SyntheticDecoder,
    paths: tuple[CoupledPath, ...],
    direct_source: np.ndarray,
    matrix: np.ndarray,
    split_round: int,
    seed: frozenset[int],
) -> ConditionalConsequence:
    event_mass = float(
        sum(
            path.probability
            for path in paths
            if path.first_divergence_round == split_round and path.first_seed == seed
        )
    )
    conditioned = conditional_paths(paths, split_round, seed)
    weights = normalized_weights(model.n_positions)
    mismatch = coordinate_mismatch_profile(conditioned, model.n_positions)
    persistent = conditional_envelope(model, direct_source, matrix, split_round, seed)
    seed_only = conditional_envelope(
        model, np.zeros(model.n_positions, dtype=np.float64), matrix, split_round, seed
    )
    return ConditionalConsequence(
        split_round=split_round,
        seed=seed,
        event_mass=event_mass,
        mismatch_profile=mismatch,
        paired_cost=paired_cost(conditioned, weights),
        persistent_envelope=persistent,
        seed_only_envelope=seed_only,
        persistent_cost_bound=float(weights @ persistent),
        seed_only_cost_bound=float(weights @ seed_only),
    )
