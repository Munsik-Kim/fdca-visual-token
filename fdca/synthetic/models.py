"""Named and seeded-random tiny masked-decoder families."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Mapping, Sequence

import numpy as np

from fdca.synthetic.state_space import MASK, State, initial_state


LawFunction = Callable[[int, State], Sequence[float]]


def _as_probabilities(values: Sequence[float], size: int) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.shape != (size,):
        raise ValueError(f"expected probability vector of shape {(size,)}, got {probabilities.shape}")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probability vector contains non-finite values")
    if np.min(probabilities) < -1e-15:
        raise ValueError(f"negative probability: {probabilities}")
    probabilities = np.maximum(probabilities, 0.0)
    total = float(probabilities.sum())
    if not np.isclose(total, 1.0, atol=1e-12, rtol=0.0):
        raise ValueError(f"probabilities sum to {total}, not 1")
    return probabilities / total


def bernoulli(probability_one: float) -> np.ndarray:
    if not 0.0 <= probability_one <= 1.0:
        raise ValueError(f"invalid Bernoulli parameter {probability_one}")
    return np.asarray([1.0 - probability_one, probability_one], dtype=np.float64)


@dataclass(frozen=True)
class SyntheticDecoder:
    """A finite fixed-schedule product decoder with reference and approximate laws."""

    name: str
    description: str
    n_positions: int
    vocabulary: tuple[int, ...]
    schedule: tuple[tuple[int, ...], ...]
    p_law: LawFunction = field(repr=False, compare=False)
    q_law: LawFunction = field(repr=False, compare=False)
    context: tuple[tuple[int, int], ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not 1 <= self.n_positions <= 8:
            raise ValueError("T0 requires 1 <= N <= 8")
        if not 1 <= len(self.vocabulary) <= 4:
            raise ValueError("T0 requires 1 <= |V| <= 4")
        if len(set(self.vocabulary)) != len(self.vocabulary) or MASK in self.vocabulary:
            raise ValueError("vocabulary must contain unique non-MASK tokens")
        context_positions = [position for position, _ in self.context]
        scheduled = [position for batch in self.schedule for position in batch]
        expected = set(range(self.n_positions))
        if set(context_positions).intersection(scheduled):
            raise ValueError("context and scheduled positions overlap")
        if set(context_positions + scheduled) != expected:
            raise ValueError("context plus schedule must partition all positions")
        if len(context_positions + scheduled) != self.n_positions:
            raise ValueError("positions must appear exactly once")
        if any(not batch for batch in self.schedule):
            raise ValueError("empty commit batch")
        for _, token in self.context:
            if token not in self.vocabulary:
                raise ValueError("context token outside vocabulary")

    @property
    def position_round(self) -> dict[int, int]:
        return {
            position: round_index
            for round_index, batch in enumerate(self.schedule)
            for position in batch
        }

    @property
    def topological_positions(self) -> tuple[int, ...]:
        return tuple(position for batch in self.schedule for position in batch)

    @property
    def initial_state(self) -> State:
        return initial_state(self)

    def law(self, branch: str, position: int, state: State) -> np.ndarray:
        if branch not in {"p", "q"}:
            raise ValueError(f"unknown branch {branch!r}")
        if position not in self.position_round:
            raise ValueError(f"position {position} is not scheduled")
        law_function = self.p_law if branch == "p" else self.q_law
        return _as_probabilities(law_function(position, state), len(self.vocabulary))


def family_a_independent() -> SyntheticDecoder:
    """Family A: state-independent coordinates and a two-position product batch."""

    p_table = {
        0: np.asarray([0.50, 0.30, 0.20]),
        1: np.asarray([0.20, 0.50, 0.30]),
        2: np.asarray([0.60, 0.20, 0.20]),
    }
    q_table = {
        0: np.asarray([0.40, 0.35, 0.25]),
        1: np.asarray([0.25, 0.45, 0.30]),
        2: np.asarray([0.45, 0.30, 0.25]),
    }
    return SyntheticDecoder(
        name="family_a_independent",
        description="Three independent ternary coordinates; M is exactly zero.",
        n_positions=3,
        vocabulary=(0, 1, 2),
        schedule=((0, 1), (2,)),
        p_law=lambda position, state: p_table[position],
        q_law=lambda position, state: q_table[position],
        metadata={"family": "A", "expected": "M=0; direct innovation only"},
    )


def family_b_local_propagation() -> SyntheticDecoder:
    """Family B: sparse one/two-parent dependence with nonzero innovation."""

    def q_probability_one(position: int, state: State) -> float:
        if position == 0:
            return 0.50
        if position == 1:
            return 0.20 + 0.60 * state[0]
        if position == 2:
            return 0.15 + 0.55 * state[1] + 0.15 * state[0]
        if position == 3:
            return 0.15 + 0.55 * state[2] + 0.20 * state[1]
        raise ValueError(position)

    def p_probability_one(position: int, state: State) -> float:
        q_probability = q_probability_one(position, state)
        if position == 0:
            return 0.65
        return q_probability + (0.04 if q_probability <= 0.50 else -0.04)

    return SyntheticDecoder(
        name="family_b_local_propagation",
        description="Binary chain with sparse local parents and finite propagation.",
        n_positions=4,
        vocabulary=(0, 1),
        schedule=((0,), (1,), (2,), (3,)),
        p_law=lambda position, state: bernoulli(p_probability_one(position, state)),
        q_law=lambda position, state: bernoulli(q_probability_one(position, state)),
        metadata={
            "family": "B",
            "expected_nonzero_influences": {
                "1<-0": 0.60,
                "2<-0": 0.15,
                "2<-1": 0.55,
                "3<-1": 0.20,
                "3<-2": 0.55,
            },
        },
    )


def family_c_strong_global() -> SyntheticDecoder:
    """Family C: dense parity dependence with large global influence."""

    def q_probability_one(position: int, state: State) -> float:
        if position == 0:
            return 0.50
        parity = sum(state[index] for index in range(position)) % 2
        return 0.05 if parity == 0 else 0.95

    def p_probability_one(position: int, state: State) -> float:
        q_probability = q_probability_one(position, state)
        if position == 0:
            return 0.58
        return 0.13 if q_probability < 0.50 else 0.87

    return SyntheticDecoder(
        name="family_c_strong_global",
        description="Dense parity influence; the clipped envelope saturates under stress.",
        n_positions=5,
        vocabulary=(0, 1),
        schedule=((0,), (1,), (2,), (3,), (4,)),
        p_law=lambda position, state: bernoulli(p_probability_one(position, state)),
        q_law=lambda position, state: bernoulli(q_probability_one(position, state)),
        metadata={"family": "C", "expected_dense_influence": 0.90},
    )


def family_d_persistent_fresh_innovation() -> SyntheticDecoder:
    """Family D: independent fresh discrepancies after an earlier split."""

    p_one = {0: 0.50, 1: 0.50, 2: 0.50}
    q_one = {0: 0.25, 1: 0.00, 2: 0.90}
    return SyntheticDecoder(
        name="family_d_persistent_fresh_innovation",
        description=(
            "State-independent persistent approximation with fresh innovations at "
            "all three rounds; a first-split seed-only bound is false."
        ),
        n_positions=3,
        vocabulary=(0, 1),
        schedule=((0,), (1,), (2,)),
        p_law=lambda position, state: bernoulli(p_one[position]),
        q_law=lambda position, state: bernoulli(q_one[position]),
        metadata={
            "family": "D",
            "counterexample": "conditional on first split at round 0, M=0 but future mismatch remains",
        },
    )


def random_table_family(seed: int, n_positions: int = 4, vocab_size: int = 3) -> SyntheticDecoder:
    """Deterministic random conditional tables for exhaustive property tests."""

    if not 1 <= n_positions <= 8 or not 2 <= vocab_size <= 4:
        raise ValueError("random family outside T0 limits")
    rng = np.random.default_rng(seed)
    vocabulary = tuple(range(vocab_size))
    p_tables: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}
    q_tables: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}
    for position in range(n_positions):
        for history in product(vocabulary, repeat=position):
            q = rng.dirichlet(np.full(vocab_size, 2.0))
            alternate = rng.dirichlet(np.full(vocab_size, 2.0))
            epsilon = float(rng.uniform(0.02, 0.18))
            p = (1.0 - epsilon) * q + epsilon * alternate
            p_tables[(position, history)] = p
            q_tables[(position, history)] = q

    def history_key(position: int, state: State) -> tuple[int, tuple[int, ...]]:
        history = tuple(state[index] for index in range(position))
        if any(value == MASK for value in history):
            raise ValueError("law requested on a non-admissible history")
        return position, history

    return SyntheticDecoder(
        name=f"random_table_seed_{seed}",
        description="Seeded random ternary conditional tables for exhaustive regression.",
        n_positions=n_positions,
        vocabulary=vocabulary,
        schedule=tuple((position,) for position in range(n_positions)),
        p_law=lambda position, state: p_tables[history_key(position, state)],
        q_law=lambda position, state: q_tables[history_key(position, state)],
        metadata={"family": "random", "seed": seed},
    )


def canonical_families() -> tuple[SyntheticDecoder, ...]:
    return (
        family_a_independent(),
        family_b_local_propagation(),
        family_c_strong_global(),
        family_d_persistent_fresh_innovation(),
    )
