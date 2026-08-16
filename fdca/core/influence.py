"""Exact direct-innovation and time-ordered influence enumeration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fdca.core.coupling import total_variation
from fdca.synthetic.models import SyntheticDecoder
from fdca.synthetic.state_space import State, enumerate_admissible_states, state_key


@dataclass(frozen=True)
class InfluenceResult:
    direct: np.ndarray
    matrix: np.ndarray
    direct_table: tuple[dict[str, object], ...]
    direct_witnesses: tuple[str | None, ...]
    influence_witnesses: dict[str, dict[str, object]]
    state_cases: int
    neighbor_pair_cases: int


def exact_influence(model: SyntheticDecoder) -> InfluenceResult:
    """Compute global s (direct b supremum) and M by finite enumeration."""

    n = model.n_positions
    direct = np.zeros(n, dtype=np.float64)
    matrix = np.zeros((n, n), dtype=np.float64)
    direct_table: list[dict[str, object]] = []
    direct_witnesses: list[str | None] = [None] * n
    influence_witnesses: dict[str, dict[str, object]] = {}
    state_cases = 0
    neighbor_pair_cases = 0

    for position in model.topological_positions:
        round_index = model.position_round[position]
        states = enumerate_admissible_states(model, round_index)
        for state in states:
            b_value = total_variation(
                model.law("p", position, state), model.law("q", position, state)
            )
            state_cases += 1
            direct_table.append(
                {
                    "position": position,
                    "round": round_index,
                    "state": state_key(state),
                    "b": b_value,
                }
            )
            if b_value > direct[position] + 1e-15 or direct_witnesses[position] is None:
                direct[position] = b_value
                direct_witnesses[position] = state_key(state)

        earlier_positions = tuple(
            earlier
            for batch in model.schedule[:round_index]
            for earlier in batch
        )
        for earlier in earlier_positions:
            best = 0.0
            best_witness: dict[str, object] | None = None
            for state in states:
                for alternate in model.vocabulary:
                    if alternate == state[earlier]:
                        continue
                    neighbor = list(state)
                    neighbor[earlier] = alternate
                    neighbor_state = tuple(neighbor)
                    value = total_variation(
                        model.law("q", position, state),
                        model.law("q", position, neighbor_state),
                    )
                    neighbor_pair_cases += 1
                    if value > best + 1e-15 or best_witness is None:
                        best = value
                        best_witness = {
                            "x": state_key(state),
                            "y": state_key(neighbor_state),
                            "value": value,
                        }
            matrix[position, earlier] = best
            if best_witness is not None:
                influence_witnesses[f"{position}<-{earlier}"] = best_witness

    return InfluenceResult(
        direct=direct,
        matrix=matrix,
        direct_table=tuple(direct_table),
        direct_witnesses=tuple(direct_witnesses),
        influence_witnesses=influence_witnesses,
        state_cases=state_cases,
        neighbor_pair_cases=neighbor_pair_cases,
    )


def time_order_violations(model: SyntheticDecoder, matrix: np.ndarray, tolerance: float = 1e-12) -> list[tuple[int, int, float]]:
    violations: list[tuple[int, int, float]] = []
    for target in range(model.n_positions):
        for source in range(model.n_positions):
            allowed = (
                target in model.position_round
                and source in model.position_round
                and model.position_round[source] < model.position_round[target]
            )
            if not allowed and abs(float(matrix[target, source])) > tolerance:
                violations.append((target, source, float(matrix[target, source])))
    return violations


def nilpotence_residual(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    return float(np.max(np.abs(np.linalg.matrix_power(matrix, matrix.shape[0]))))


def hybrid_inequality_violations(
    model: SyntheticDecoder,
    influence: InfluenceResult,
    tolerance: float = 1e-12,
) -> list[dict[str, object]]:
    """Exhaustively check Lemma 4.4 over every admissible state pair."""

    violations: list[dict[str, object]] = []
    for position in model.topological_positions:
        round_index = model.position_round[position]
        states = enumerate_admissible_states(model, round_index)
        earlier = tuple(position_ for batch in model.schedule[:round_index] for position_ in batch)
        for x_state in states:
            for y_state in states:
                mismatch = tuple(v for v in earlier if x_state[v] != y_state[v])
                lhs = total_variation(
                    model.law("p", position, x_state), model.law("q", position, y_state)
                )
                rhs = float(
                    influence.direct[position]
                    + sum(influence.matrix[position, v] for v in mismatch)
                )
                if lhs > rhs + tolerance:
                    violations.append(
                        {
                            "position": position,
                            "x": state_key(x_state),
                            "y": state_key(y_state),
                            "lhs": lhs,
                            "rhs": rhs,
                        }
                    )
    return violations
