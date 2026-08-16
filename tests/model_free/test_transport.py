from __future__ import annotations

import pytest

from fdca.metrics.transport import exact_wasserstein
from fdca.synthetic.coupled_paths import (
    enumerate_coupled_paths,
    normalized_weights,
    paired_cost,
)
from fdca.synthetic.enumerate_laws import enumerate_terminal_law
from fdca.synthetic.models import canonical_families, random_table_family


@pytest.mark.parametrize("model", canonical_families(), ids=lambda model: model.name)
@pytest.mark.parametrize("coupling", ("maximal", "crn"))
def test_exact_terminal_wasserstein_is_bounded_by_constructed_coupling(model, coupling):
    reference = enumerate_terminal_law(model, "p")
    approximate = enumerate_terminal_law(model, "q")
    weights = normalized_weights(model.n_positions)
    transport = exact_wasserstein(reference, approximate, weights)
    cost = paired_cost(enumerate_coupled_paths(model, coupling), weights)
    assert transport.success
    assert transport.max_marginal_residual <= 1e-9
    assert transport.distance <= cost + 1e-9


def test_random_instance_exact_terminal_wasserstein_bound():
    model = random_table_family(1729)
    reference = enumerate_terminal_law(model, "p")
    approximate = enumerate_terminal_law(model, "q")
    weights = normalized_weights(model.n_positions)
    transport = exact_wasserstein(reference, approximate, weights)
    cost = paired_cost(enumerate_coupled_paths(model, "maximal"), weights)
    assert transport.distance <= cost + 1e-9


def test_wasserstein_identical_law_is_zero():
    model = canonical_families()[0]
    law = enumerate_terminal_law(model, "p")
    result = exact_wasserstein(law, law, normalized_weights(model.n_positions))
    assert result.distance == pytest.approx(0.0, abs=1e-12)
