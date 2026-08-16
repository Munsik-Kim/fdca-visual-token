from __future__ import annotations

import numpy as np
import pytest

from fdca.core.dag_envelope import (
    clipped_envelope,
    finite_resolvent_matrix,
    inverse_resolvent_matrix,
    linear_envelope,
    recursion_rhs,
)
from fdca.core.influence import (
    exact_influence,
    hybrid_inequality_violations,
    nilpotence_residual,
    time_order_violations,
)
from fdca.synthetic.coupled_paths import (
    coordinate_mismatch_profile,
    enumerate_coupled_paths,
)
from fdca.synthetic.models import (
    canonical_families,
    family_a_independent,
    family_b_local_propagation,
    family_c_strong_global,
    random_table_family,
)


NAMED = canonical_families()
RANDOM = tuple(random_table_family(seed) for seed in (1729, 20260812, 314159))
ALL = NAMED + RANDOM


def test_influence_matrix_zero_in_family_a():
    result = exact_influence(family_a_independent())
    assert np.array_equal(result.matrix, np.zeros((3, 3)))


def test_influence_matrix_exact_enumeration_in_family_b():
    result = exact_influence(family_b_local_propagation())
    expected = np.zeros((4, 4))
    expected[1, 0] = 0.60
    expected[2, 0] = 0.15
    expected[2, 1] = 0.55
    expected[3, 1] = 0.20
    expected[3, 2] = 0.55
    assert np.allclose(result.matrix, expected, atol=1e-12, rtol=0.0)


def test_family_c_dense_global_influence_is_exact():
    result = exact_influence(family_c_strong_global())
    for target in range(1, 5):
        assert np.allclose(result.matrix[target, :target], 0.90, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("model", ALL, ids=lambda model: model.name)
def test_m_is_time_ordered_and_nilpotent(model):
    result = exact_influence(model)
    assert time_order_violations(model, result.matrix) == []
    assert nilpotence_residual(result.matrix) <= 1e-12


@pytest.mark.parametrize("model", ALL, ids=lambda model: model.name)
def test_inverse_equals_finite_nilpotent_series(model):
    matrix = exact_influence(model).matrix
    inverse = inverse_resolvent_matrix(matrix)
    finite = finite_resolvent_matrix(matrix)
    assert np.allclose(inverse, finite, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("model", ALL, ids=lambda model: model.name)
def test_innovation_plus_propagation_hybrid_lemma_exhaustive(model):
    influence = exact_influence(model)
    assert hybrid_inequality_violations(model, influence) == []


@pytest.mark.parametrize("model", ALL, ids=lambda model: model.name)
def test_exact_pi_is_bounded_by_linear_recursion(model):
    influence = exact_influence(model)
    paths = enumerate_coupled_paths(model, "maximal")
    pi = coordinate_mismatch_profile(paths, model.n_positions)
    assert np.all(pi <= recursion_rhs(influence.direct, influence.matrix, pi) + 1e-10)


@pytest.mark.parametrize("model", ALL, ids=lambda model: model.name)
def test_exact_pi_is_bounded_by_clipped_envelope(model):
    influence = exact_influence(model)
    pi = coordinate_mismatch_profile(enumerate_coupled_paths(model, "maximal"), model.n_positions)
    clipped = clipped_envelope(model, influence.direct, influence.matrix)
    assert np.all(pi <= clipped + 1e-10)


@pytest.mark.parametrize("model", ALL, ids=lambda model: model.name)
def test_exact_pi_is_bounded_by_linear_resolvent(model):
    influence = exact_influence(model)
    pi = coordinate_mismatch_profile(enumerate_coupled_paths(model, "maximal"), model.n_positions)
    linear = linear_envelope(influence.matrix, influence.direct)
    assert np.all(pi <= linear + 1e-10)


@pytest.mark.parametrize("model", ALL, ids=lambda model: model.name)
def test_clipped_envelope_is_bounded_by_linear_resolvent(model):
    influence = exact_influence(model)
    clipped = clipped_envelope(model, influence.direct, influence.matrix)
    linear = linear_envelope(influence.matrix, influence.direct)
    assert np.all(clipped <= linear + 1e-10)


def test_family_a_terminal_mismatch_is_direct_innovation_only():
    model = family_a_independent()
    influence = exact_influence(model)
    pi = coordinate_mismatch_profile(enumerate_coupled_paths(model, "maximal"), model.n_positions)
    assert np.allclose(pi, influence.direct, atol=1e-12, rtol=0.0)


def test_family_c_clipping_stays_bounded_and_is_nontrivial():
    model = family_c_strong_global()
    influence = exact_influence(model)
    clipped = clipped_envelope(model, influence.direct, influence.matrix)
    linear = linear_envelope(influence.matrix, influence.direct)
    assert np.max(clipped) == pytest.approx(1.0, abs=1e-12)
    assert linear[-1] > 1.0
    assert np.all(clipped <= 1.0)


def test_crn_is_not_silently_promoted_to_maximal_theorem():
    model = family_a_independent()
    direct = exact_influence(model).direct
    crn_pi = coordinate_mismatch_profile(enumerate_coupled_paths(model, "crn"), model.n_positions)
    assert np.any(crn_pi > direct + 1e-12)
