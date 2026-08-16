"""Exact finite-state branch-law enumeration."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Iterable

import numpy as np

from fdca.synthetic.models import SyntheticDecoder
from fdca.synthetic.state_space import State, enumerate_admissible_states, state_key


Law = dict[State, float]


def _batch_outcomes(
    model: SyntheticDecoder,
    branch: str,
    state: State,
    batch: tuple[int, ...],
) -> Iterable[tuple[tuple[int, ...], float]]:
    laws = [model.law(branch, position, state) for position in batch]
    for token_indices in product(range(len(model.vocabulary)), repeat=len(batch)):
        probability = float(
            np.prod([law[token_index] for law, token_index in zip(laws, token_indices, strict=True)])
        )
        if probability == 0.0:
            continue
        tokens = tuple(model.vocabulary[index] for index in token_indices)
        yield tokens, probability


def enumerate_stage_laws(model: SyntheticDecoder, branch: str) -> tuple[Law, ...]:
    """Return the exact state law after every stage, including stage zero."""

    stages: list[Law] = [{model.initial_state: 1.0}]
    for batch in model.schedule:
        next_law: defaultdict[State, float] = defaultdict(float)
        for state, state_probability in stages[-1].items():
            for tokens, transition_probability in _batch_outcomes(model, branch, state, batch):
                updated = list(state)
                for position, token in zip(batch, tokens, strict=True):
                    updated[position] = token
                next_law[tuple(updated)] += state_probability * transition_probability
        stages.append(dict(next_law))
    return tuple(stages)


def enumerate_stage_laws_by_round(
    model: SyntheticDecoder, branches: tuple[str, ...]
) -> tuple[Law, ...]:
    """Enumerate a marginal whose branch kernel can change between rounds."""

    if len(branches) != len(model.schedule) or any(branch not in {"p", "q"} for branch in branches):
        raise ValueError("branches must contain one 'p' or 'q' entry per round")
    stages: list[Law] = [{model.initial_state: 1.0}]
    for batch, branch in zip(model.schedule, branches, strict=True):
        next_law: defaultdict[State, float] = defaultdict(float)
        for state, state_probability in stages[-1].items():
            for tokens, transition_probability in _batch_outcomes(model, branch, state, batch):
                updated = list(state)
                for position, token in zip(batch, tokens, strict=True):
                    updated[position] = token
                next_law[tuple(updated)] += state_probability * transition_probability
        stages.append(dict(next_law))
    return tuple(stages)


def enumerate_terminal_law(model: SyntheticDecoder, branch: str) -> Law:
    return enumerate_stage_laws(model, branch)[-1]


def enumerate_terminal_law_by_round(model: SyntheticDecoder, branches: tuple[str, ...]) -> Law:
    return enumerate_stage_laws_by_round(model, branches)[-1]


def transition_table(model: SyntheticDecoder) -> list[dict[str, object]]:
    """Enumerate every declared same-state p/q categorical law and direct TV."""

    rows: list[dict[str, object]] = []
    for round_index, batch in enumerate(model.schedule):
        for state in enumerate_admissible_states(model, round_index):
            for position in batch:
                p = model.law("p", position, state)
                q = model.law("q", position, state)
                rows.append(
                    {
                        "round": round_index,
                        "position": position,
                        "state": state_key(state),
                        "p": p.tolist(),
                        "q": q.tolist(),
                        "b": float(0.5 * np.abs(p - q).sum()),
                    }
                )
    return rows


def law_normalization_error(law: Law) -> float:
    return abs(float(sum(law.values())) - 1.0)


def expected_state_count(model: SyntheticDecoder, stage: int) -> int:
    committed = sum(len(batch) for batch in model.schedule[:stage])
    return len(model.vocabulary) ** committed
