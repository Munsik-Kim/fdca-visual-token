from __future__ import annotations

import numpy as np
import pytest

from fdca.core.consequence import evaluate_conditional_consequence
from fdca.core.first_divergence import first_divergence_distribution, first_seed_distribution
from fdca.core.influence import exact_influence
from fdca.synthetic.coupled_paths import (
    enumerate_coupled_paths,
    normalized_weights,
    persistent_plan,
    single_shock_plan,
    stable_set_pathwise_violations,
)
from fdca.synthetic.models import (
    canonical_families,
    family_b_local_propagation,
    family_d_persistent_fresh_innovation,
    random_table_family,
)


ALL = canonical_families() + (random_table_family(1729),)


@pytest.mark.parametrize("model", ALL, ids=lambda model: model.name)
@pytest.mark.parametrize("coupling", ("maximal", "crn"))
def test_residual_mask_stable_set_pathwise_violation_is_zero(model, coupling):
    paths = enumerate_coupled_paths(model, coupling)
    violations = stable_set_pathwise_violations(
        model, paths, normalized_weights(model.n_positions)
    )
    assert violations == []


def test_first_divergence_and_seed_distributions_normalize():
    model = family_b_local_propagation()
    paths = enumerate_coupled_paths(model, "maximal")
    assert sum(first_divergence_distribution(paths).values()) == pytest.approx(1.0, abs=1e-12)
    assert sum(first_seed_distribution(paths).values()) == pytest.approx(1.0, abs=1e-12)


def test_committed_mismatch_never_heals_pathwise():
    model = family_b_local_propagation()
    for path in enumerate_coupled_paths(model, "maximal"):
        for history_index in range(1, len(path.x_history)):
            mismatched = {
                position
                for position in range(model.n_positions)
                if path.x_history[history_index][position]
                != path.y_history[history_index][position]
            }
            terminal_mismatch = {
                position
                for position in range(model.n_positions)
                if path.x_terminal[position] != path.y_terminal[position]
            }
            assert mismatched.issubset(terminal_mismatch)


def test_single_shock_seed_propagation_envelope_passes():
    model = family_b_local_propagation()
    influence = exact_influence(model)
    paths = enumerate_coupled_paths(model, "maximal", single_shock_plan(model, 0))
    result = evaluate_conditional_consequence(
        model, paths, influence.direct, influence.matrix, 0, frozenset({0})
    )
    assert np.all(result.mismatch_profile <= result.seed_only_envelope + 1e-12)
    assert result.paired_cost <= result.seed_only_cost_bound + 1e-12
    assert result.seed_only_envelope == pytest.approx([1.0, 0.6, 0.48, 0.384], abs=1e-12)


def test_single_shock_has_no_fresh_suffix_source():
    model = family_b_local_propagation()
    influence = exact_influence(model)
    paths = enumerate_coupled_paths(model, "maximal", single_shock_plan(model, 0))
    result = evaluate_conditional_consequence(
        model, paths, influence.direct, influence.matrix, 0, frozenset({0})
    )
    assert np.allclose(result.mismatch_profile, result.seed_only_envelope, atol=1e-12, rtol=0.0)


def test_persistent_fresh_innovation_counterexample_exists():
    model = family_d_persistent_fresh_innovation()
    influence = exact_influence(model)
    paths = enumerate_coupled_paths(model, "maximal", persistent_plan(model))
    result = evaluate_conditional_consequence(
        model, paths, influence.direct, influence.matrix, 0, frozenset({0})
    )
    assert result.event_mass == pytest.approx(0.25, abs=1e-12)
    assert result.mismatch_profile == pytest.approx([1.0, 0.5, 0.4], abs=1e-12)
    assert result.paired_cost > result.seed_only_cost_bound + 0.29
    assert np.any(result.mismatch_profile > result.seed_only_envelope + 1e-12)


def test_persistent_envelope_with_future_innovation_passes():
    model = family_d_persistent_fresh_innovation()
    influence = exact_influence(model)
    paths = enumerate_coupled_paths(model, "maximal", persistent_plan(model))
    result = evaluate_conditional_consequence(
        model, paths, influence.direct, influence.matrix, 0, frozenset({0})
    )
    assert np.all(result.mismatch_profile <= result.persistent_envelope + 1e-12)
    assert result.paired_cost <= result.persistent_cost_bound + 1e-12
    assert result.persistent_envelope == pytest.approx([1.0, 0.5, 0.4], abs=1e-12)
