from __future__ import annotations

import numpy as np
import pytest

from fdca.synthetic.coupled_paths import (
    enumerate_coupled_paths,
    joint_marginals,
    marginal_errors,
    persistent_plan,
    probability_sum,
    terminal_joint_law,
)
from fdca.synthetic.enumerate_laws import (
    enumerate_stage_laws,
    enumerate_terminal_law,
    expected_state_count,
    law_normalization_error,
)
from fdca.synthetic.models import canonical_families, family_a_independent, random_table_family
from fdca.synthetic.state_space import enumerate_admissible_states, is_admissible_state


NAMED = canonical_families()
RANDOM = tuple(random_table_family(seed) for seed in (1729, 20260812, 314159))
ALL = NAMED + RANDOM


@pytest.mark.parametrize("model", ALL, ids=lambda model: model.name)
def test_state_enumeration_completeness(model):
    for stage in range(len(model.schedule) + 1):
        states = enumerate_admissible_states(model, stage)
        assert len(states) == expected_state_count(model, stage)
        assert len(set(states)) == len(states)
        assert all(is_admissible_state(model, state, stage) for state in states)


@pytest.mark.parametrize("model", ALL, ids=lambda model: model.name)
def test_transition_normalization(model):
    for round_index, batch in enumerate(model.schedule):
        for state in enumerate_admissible_states(model, round_index):
            for position in batch:
                for branch in ("p", "q"):
                    law = model.law(branch, position, state)
                    assert law.dtype == np.float64
                    assert np.min(law) >= 0.0
                    assert np.isclose(law.sum(), 1.0, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("model", ALL, ids=lambda model: model.name)
@pytest.mark.parametrize("branch", ("p", "q"))
def test_terminal_marginal_normalization(model, branch):
    stages = enumerate_stage_laws(model, branch)
    assert len(stages) == len(model.schedule) + 1
    for stage_index, law in enumerate(stages):
        assert law_normalization_error(law) <= 1e-12
        assert set(law).issubset(set(enumerate_admissible_states(model, stage_index)))


@pytest.mark.parametrize("model", ALL, ids=lambda model: model.name)
def test_trajectory_to_terminal_law_agreement_maximal(model):
    plan = persistent_plan(model)
    paths = enumerate_coupled_paths(model, "maximal", plan)
    assert abs(probability_sum(paths) - 1.0) <= 1e-12
    left_error, right_error = marginal_errors(model, paths, plan)
    assert left_error <= 1e-12
    assert right_error <= 1e-12


@pytest.mark.parametrize("model", NAMED, ids=lambda model: model.name)
def test_trajectory_to_terminal_law_agreement_crn(model):
    plan = persistent_plan(model)
    paths = enumerate_coupled_paths(model, "crn", plan)
    assert abs(probability_sum(paths) - 1.0) <= 1e-12
    left_error, right_error = marginal_errors(model, paths, plan)
    assert left_error <= 1e-12
    assert right_error <= 1e-12


def test_terminal_marginals_independently_recomputed_from_trajectories():
    model = random_table_family(1729)
    paths = enumerate_coupled_paths(model, "maximal")
    left, right = joint_marginals(terminal_joint_law(paths))
    assert left == pytest.approx(enumerate_terminal_law(model, "p"), abs=1e-12)
    assert right == pytest.approx(enumerate_terminal_law(model, "q"), abs=1e-12)


def test_product_batch_state_count_and_factorization():
    model = family_a_independent()
    stages = enumerate_stage_laws(model, "p")
    assert model.schedule[0] == (0, 1)
    assert len(stages[1]) == 3**2
    assert stages[1][(0, 0, -1)] == pytest.approx(0.50 * 0.20, abs=1e-15)


def test_deterministic_law_replay():
    model = random_table_family(314159)
    first = enumerate_stage_laws(model, "q")
    second = enumerate_stage_laws(model, "q")
    assert first == second
