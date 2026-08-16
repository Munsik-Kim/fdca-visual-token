"""Exhaustive joint trajectory enumeration for coordinatewise couplings."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np

from fdca.core.coupling import joint_for_name
from fdca.synthetic.enumerate_laws import enumerate_terminal_law_by_round
from fdca.synthetic.models import SyntheticDecoder
from fdca.synthetic.state_space import State


@dataclass(frozen=True)
class BranchPlan:
    left: tuple[str, ...]
    right: tuple[str, ...]
    label: str

    def validate(self, rounds: int) -> None:
        if len(self.left) != rounds or len(self.right) != rounds:
            raise ValueError("branch plan must contain one kernel per round")
        if any(branch not in {"p", "q"} for branch in self.left + self.right):
            raise ValueError("branch plan entries must be 'p' or 'q'")


def persistent_plan(model: SyntheticDecoder) -> BranchPlan:
    rounds = len(model.schedule)
    return BranchPlan(("p",) * rounds, ("q",) * rounds, "persistent_p_vs_q")


def single_shock_plan(model: SyntheticDecoder, shock_round: int) -> BranchPlan:
    if not 0 <= shock_round < len(model.schedule):
        raise ValueError("shock round outside schedule")
    left = []
    right = []
    for round_index in range(len(model.schedule)):
        if round_index < shock_round:
            left.append("p")
            right.append("p")
        elif round_index == shock_round:
            left.append("p")
            right.append("q")
        else:
            left.append("q")
            right.append("q")
    return BranchPlan(tuple(left), tuple(right), f"single_shock_round_{shock_round}")


@dataclass(frozen=True)
class CoupledPath:
    probability: float
    x_history: tuple[State, ...]
    y_history: tuple[State, ...]
    first_divergence_round: int | None
    first_seed: frozenset[int]

    @property
    def x_terminal(self) -> State:
        return self.x_history[-1]

    @property
    def y_terminal(self) -> State:
        return self.y_history[-1]


def _coordinate_outcomes(
    model: SyntheticDecoder,
    coupling: str,
    left_branch: str,
    right_branch: str,
    position: int,
    x_state: State,
    y_state: State,
) -> tuple[tuple[int, int, float], ...]:
    p = model.law(left_branch, position, x_state)
    q = model.law(right_branch, position, y_state)
    joint = joint_for_name(coupling, p, q)
    outcomes: list[tuple[int, int, float]] = []
    for x_index, x_token in enumerate(model.vocabulary):
        for y_index, y_token in enumerate(model.vocabulary):
            probability = float(joint[x_index, y_index])
            if probability > 0.0:
                outcomes.append((x_token, y_token, probability))
    return tuple(outcomes)


def enumerate_coupled_paths(
    model: SyntheticDecoder,
    coupling: str = "maximal",
    plan: BranchPlan | None = None,
) -> tuple[CoupledPath, ...]:
    """Enumerate every positive-mass paired trajectory under a branch plan."""

    branch_plan = plan or persistent_plan(model)
    branch_plan.validate(len(model.schedule))
    paths: tuple[CoupledPath, ...] = (
        CoupledPath(1.0, (model.initial_state,), (model.initial_state,), None, frozenset()),
    )
    for round_index, batch in enumerate(model.schedule):
        next_paths: list[CoupledPath] = []
        for path in paths:
            x_state = path.x_history[-1]
            y_state = path.y_history[-1]
            per_coordinate = [
                _coordinate_outcomes(
                    model,
                    coupling,
                    branch_plan.left[round_index],
                    branch_plan.right[round_index],
                    position,
                    x_state,
                    y_state,
                )
                for position in batch
            ]
            for batch_outcome in product(*per_coordinate):
                x_updated = list(x_state)
                y_updated = list(y_state)
                transition_probability = 1.0
                for position, (x_token, y_token, probability) in zip(
                    batch, batch_outcome, strict=True
                ):
                    x_updated[position] = x_token
                    y_updated[position] = y_token
                    transition_probability *= probability
                x_next = tuple(x_updated)
                y_next = tuple(y_updated)
                first_round = path.first_divergence_round
                first_seed = path.first_seed
                if first_round is None and x_next != y_next:
                    first_round = round_index
                    first_seed = frozenset(
                        position for position in batch if x_next[position] != y_next[position]
                    )
                next_paths.append(
                    CoupledPath(
                        probability=path.probability * transition_probability,
                        x_history=path.x_history + (x_next,),
                        y_history=path.y_history + (y_next,),
                        first_divergence_round=first_round,
                        first_seed=first_seed,
                    )
                )
        paths = tuple(next_paths)
    return paths


def terminal_joint_law(paths: Iterable[CoupledPath]) -> dict[tuple[State, State], float]:
    law: defaultdict[tuple[State, State], float] = defaultdict(float)
    for path in paths:
        law[(path.x_terminal, path.y_terminal)] += path.probability
    return dict(law)


def joint_marginals(
    joint_law: dict[tuple[State, State], float]
) -> tuple[dict[State, float], dict[State, float]]:
    left: defaultdict[State, float] = defaultdict(float)
    right: defaultdict[State, float] = defaultdict(float)
    for (x_state, y_state), probability in joint_law.items():
        left[x_state] += probability
        right[y_state] += probability
    return dict(left), dict(right)


def maximum_law_error(actual: dict[State, float], expected: dict[State, float]) -> float:
    states = set(actual).union(expected)
    return max((abs(actual.get(state, 0.0) - expected.get(state, 0.0)) for state in states), default=0.0)


def marginal_errors(
    model: SyntheticDecoder, paths: Iterable[CoupledPath], plan: BranchPlan
) -> tuple[float, float]:
    left_actual, right_actual = joint_marginals(terminal_joint_law(paths))
    left_expected = enumerate_terminal_law_by_round(model, plan.left)
    right_expected = enumerate_terminal_law_by_round(model, plan.right)
    return (
        maximum_law_error(left_actual, left_expected),
        maximum_law_error(right_actual, right_expected),
    )


def coordinate_mismatch_profile(
    paths: Iterable[CoupledPath], n_positions: int
) -> np.ndarray:
    profile = np.zeros(n_positions, dtype=np.float64)
    for path in paths:
        for position in range(n_positions):
            if path.x_terminal[position] != path.y_terminal[position]:
                profile[position] += path.probability
    return profile


def normalized_weights(n_positions: int) -> np.ndarray:
    return np.full(n_positions, 1.0 / n_positions, dtype=np.float64)


def weighted_hamming(x_state: State, y_state: State, weights: Sequence[float]) -> float:
    weight_vector = np.asarray(weights, dtype=np.float64)
    if weight_vector.shape != (len(x_state),) or len(y_state) != len(x_state):
        raise ValueError("state/weight shape mismatch")
    return float(
        sum(weight_vector[position] for position in range(len(x_state)) if x_state[position] != y_state[position])
    )


def paired_cost(paths: Iterable[CoupledPath], weights: Sequence[float]) -> float:
    return float(
        sum(
            path.probability * weighted_hamming(path.x_terminal, path.y_terminal, weights)
            for path in paths
        )
    )


def stable_set_pathwise_violations(
    model: SyntheticDecoder,
    paths: Iterable[CoupledPath],
    weights: Sequence[float],
    tolerance: float = 1e-12,
) -> list[dict[str, object]]:
    """Audit terminal Hamming <= unresolved mass before the first split round."""

    weight_vector = np.asarray(weights, dtype=np.float64)
    violations: list[dict[str, object]] = []
    for path_index, path in enumerate(paths):
        terminal_cost = weighted_hamming(path.x_terminal, path.y_terminal, weight_vector)
        if path.first_divergence_round is None:
            residual_mass = 0.0
        else:
            unresolved = tuple(
                position
                for batch in model.schedule[path.first_divergence_round :]
                for position in batch
            )
            residual_mass = float(weight_vector[list(unresolved)].sum())
        if terminal_cost > residual_mass + tolerance:
            violations.append(
                {
                    "path_index": path_index,
                    "probability": path.probability,
                    "first_divergence_round": path.first_divergence_round,
                    "terminal_cost": terminal_cost,
                    "residual_mass": residual_mass,
                }
            )
    return violations


def conditional_paths(
    paths: Iterable[CoupledPath], split_round: int, seed: frozenset[int]
) -> tuple[CoupledPath, ...]:
    selected = tuple(
        path
        for path in paths
        if path.first_divergence_round == split_round and path.first_seed == seed
    )
    mass = float(sum(path.probability for path in selected))
    if mass <= 0.0:
        raise ValueError("conditioning event has zero probability")
    return tuple(
        CoupledPath(
            probability=path.probability / mass,
            x_history=path.x_history,
            y_history=path.y_history,
            first_divergence_round=path.first_divergence_round,
            first_seed=path.first_seed,
        )
        for path in selected
    )


def probability_sum(paths: Iterable[CoupledPath]) -> float:
    return float(sum(path.probability for path in paths))
