from __future__ import annotations

import numpy as np
import pytest

from fdca.core.coupling import (
    audit_joint,
    conditional_maximal_joint,
    conditional_maximal_sample,
    inverse_cdf_crn_joint,
    inverse_cdf_sample,
    total_variation,
    uniform_from_key,
)


CASES = (
    (np.asarray([0.5, 0.5]), np.asarray([0.5, 0.5])),
    (np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])),
    (np.asarray([0.7, 0.3]), np.asarray([0.2, 0.8])),
    (np.asarray([0.2, 0.5, 0.3]), np.asarray([0.4, 0.1, 0.5])),
    (np.asarray([0.0, 0.4, 0.6]), np.asarray([0.5, 0.5, 0.0])),
)


@pytest.mark.parametrize("p,q", CASES)
def test_categorical_maximal_coupling_marginals(p, q):
    audit = audit_joint(conditional_maximal_joint(p, q), p, q)
    assert audit.x_marginal_error <= 1e-12
    assert audit.y_marginal_error <= 1e-12
    assert audit.normalization_error <= 1e-12


@pytest.mark.parametrize("p,q", CASES)
def test_categorical_maximal_mismatch_equals_tv(p, q):
    audit = audit_joint(conditional_maximal_joint(p, q), p, q)
    assert audit.mismatch_probability == pytest.approx(total_variation(p, q), abs=1e-12)


def test_tv_zero_boundary_has_no_nan_and_diagonal_joint():
    p = np.asarray([0.2, 0.3, 0.5])
    joint = conditional_maximal_joint(p, p)
    assert np.all(np.isfinite(joint))
    assert np.allclose(joint, np.diag(p), atol=1e-15, rtol=0.0)
    assert audit_joint(joint, p, p).mismatch_probability == pytest.approx(0.0, abs=1e-15)


def test_support_mismatch_case_is_valid():
    p = np.asarray([1.0, 0.0, 0.0])
    q = np.asarray([0.0, 0.25, 0.75])
    audit = audit_joint(conditional_maximal_joint(p, q), p, q)
    assert audit.mismatch_probability == pytest.approx(1.0, abs=1e-15)
    assert audit.x_marginal_error <= 1e-15
    assert audit.y_marginal_error <= 1e-15


@pytest.mark.parametrize("p,q", CASES)
def test_crn_marginal_validity(p, q):
    audit = audit_joint(inverse_cdf_crn_joint(p, q), p, q)
    assert audit.x_marginal_error <= 1e-12
    assert audit.y_marginal_error <= 1e-12
    assert audit.normalization_error <= 1e-12


@pytest.mark.parametrize("p,q", CASES)
def test_crn_mismatch_is_at_least_tv(p, q):
    audit = audit_joint(inverse_cdf_crn_joint(p, q), p, q)
    assert audit.mismatch_probability + 1e-12 >= audit.tv


def test_conditional_form_deterministic_replay():
    p = np.asarray([0.2, 0.5, 0.3])
    q = np.asarray([0.4, 0.1, 0.5])
    key = ("T0", "family", 17, 2)
    assert conditional_maximal_sample(p, q, key) == conditional_maximal_sample(p, q, key)


def test_reference_draw_is_shared_with_crn_baseline_key():
    p = np.asarray([0.2, 0.5, 0.3])
    q = np.asarray([0.4, 0.1, 0.5])
    key = ("T0", "reference-invariance", 9)
    x, _ = conditional_maximal_sample(p, q, key)
    expected_x = inverse_cdf_sample(p, uniform_from_key(*key, "proposal_x"))
    assert x == expected_x
