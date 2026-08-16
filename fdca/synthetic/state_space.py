"""Finite admissible state spaces for fixed commit-once schedules."""

from __future__ import annotations

from itertools import product
from typing import Protocol


MASK = -1
State = tuple[int, ...]


class ScheduledModel(Protocol):
    n_positions: int
    vocabulary: tuple[int, ...]
    schedule: tuple[tuple[int, ...], ...]
    context: tuple[tuple[int, int], ...]


def initial_state(model: ScheduledModel) -> State:
    """Return the common partially masked initial state."""

    state = [MASK] * model.n_positions
    for position, token in model.context:
        state[position] = token
    return tuple(state)


def committed_positions(model: ScheduledModel, stage: int) -> tuple[int, ...]:
    """Positions decided after ``stage`` commit batches have completed."""

    if not 0 <= stage <= len(model.schedule):
        raise ValueError(f"stage {stage} outside [0, {len(model.schedule)}]")
    context_positions = tuple(position for position, _ in model.context)
    scheduled = tuple(position for batch in model.schedule[:stage] for position in batch)
    return context_positions + scheduled


def unresolved_positions(model: ScheduledModel, stage: int) -> tuple[int, ...]:
    decided = set(committed_positions(model, stage))
    return tuple(position for position in range(model.n_positions) if position not in decided)


def enumerate_admissible_states(model: ScheduledModel, stage: int) -> tuple[State, ...]:
    """Exhaustively enumerate states after ``stage`` fixed batches.

    Context positions remain fixed. Every scheduled position in an earlier batch
    ranges over the declared vocabulary, and every future position is MASK.
    """

    initial = list(initial_state(model))
    variable_positions = tuple(
        position for batch in model.schedule[:stage] for position in batch
    )
    states: list[State] = []
    for values in product(model.vocabulary, repeat=len(variable_positions)):
        state = initial.copy()
        for position, value in zip(variable_positions, values, strict=True):
            state[position] = value
        states.append(tuple(state))
    return tuple(states)


def is_admissible_state(model: ScheduledModel, state: State, stage: int) -> bool:
    if len(state) != model.n_positions:
        return False
    context = dict(model.context)
    decided = set(committed_positions(model, stage))
    for position, value in enumerate(state):
        if position in context and value != context[position]:
            return False
        if position in decided:
            if value not in model.vocabulary:
                return False
        elif value != MASK:
            return False
    return True


def state_key(state: State) -> str:
    return "[" + ",".join("M" if value == MASK else str(value) for value in state) + "]"
