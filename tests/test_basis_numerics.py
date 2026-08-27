"""Tests for exact full-rank basis whitening and diagnostics."""

import cvxpy as cp
import numpy as np
import pytest

from signaldecomp.basis_numerics import (
    BasisRankError,
    basis_rank_diagnostics,
    whiten_basis,
)


def test_whitening_preserves_expression_penalty_and_coefficients():
    rng = np.random.default_rng(10)
    basis = rng.standard_normal((80, 6)) @ np.diag(np.logspace(0, 5, 6))
    fit_mask = np.ones(80, dtype=bool)
    fit_mask[::7] = False
    result = whiten_basis(basis, fit_mask)

    beta = rng.standard_normal(6)
    gamma = np.linalg.solve(result.transform, beta)
    recovered = result.recover_coefficients(gamma)

    assert np.allclose(basis @ beta, result.whitened_basis @ gamma)
    assert np.allclose(recovered, beta)
    assert np.isclose(np.sum(beta**2), np.sum(recovered**2))
    assert result.training_gram_error < 1e-10


def test_basis_diagnostics_report_column_scaling_separately():
    rng = np.random.default_rng(11)
    basis = rng.standard_normal((100, 3)) * np.array([1.0, 1e3, 1e6])
    diagnostics = basis_rank_diagnostics(basis, np.ones(100, dtype=bool))
    assert diagnostics.numerical_rank == 3
    assert diagnostics.condition_number > 1e5
    assert diagnostics.column_normalized_condition_number < 2.0


@pytest.mark.parametrize(
    "basis",
    [
        np.column_stack([np.arange(8.0), np.arange(8.0)]),
        np.ones((2, 3)),
    ],
)
def test_whitening_rejects_rank_deficient_or_underdetermined_basis(basis):
    with pytest.raises(BasisRankError) as excinfo:
        whiten_basis(basis, np.ones(basis.shape[0], dtype=bool), context="test")
    assert excinfo.value.diagnostics is not None
    assert "shape=" in str(excinfo.value)


def test_whitening_rejects_nonfinite_basis():
    basis = np.eye(4)
    basis[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        whiten_basis(basis, np.ones(4, dtype=bool))


def test_whitening_requires_boolean_nonempty_matching_mask():
    basis = np.eye(4)
    with pytest.raises(TypeError, match="boolean"):
        whiten_basis(basis, np.ones(4))
    with pytest.raises(ValueError, match="shape"):
        whiten_basis(basis, np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="no rows"):
        whiten_basis(basis, np.zeros(4, dtype=bool))


def test_explicit_rank_tolerance_fails_closed_without_truncation():
    basis = np.diag([1.0, 1e-9])
    fit_mask = np.ones(2, dtype=bool)
    assert whiten_basis(basis, fit_mask, rank_tolerance=1e-10)
    with pytest.raises(BasisRankError):
        whiten_basis(basis, fit_mask, rank_tolerance=1e-8)


@pytest.mark.skipif(
    not {"CLARABEL", "SCS"}.issubset(cp.installed_solvers()),
    reason="cross-solver conditioning control requires CLARABEL and SCS",
)
def test_whitening_improves_cross_solver_agreement_on_full_rank_case():
    rng = np.random.default_rng(33)
    n, p = 240, 8
    left, _ = np.linalg.qr(rng.standard_normal((n, p)))
    right, _ = np.linalg.qr(rng.standard_normal((p, p)))
    basis = left @ np.diag(np.logspace(0, -8, p)) @ right.T
    fit_mask = np.ones(n, dtype=bool)
    whitening = whiten_basis(basis, fit_mask)
    y = basis @ rng.standard_normal(p) + 1e-9 * rng.standard_normal(n)
    ridge_weight = 1e-12

    results = {}
    for formulation in ("raw", "white"):
        for solver in ("CLARABEL", "SCS"):
            if formulation == "raw":
                original_coef = cp.Variable(p)
                component = basis @ original_coef
            else:
                numerical_coef = cp.Variable(p)
                original_coef = whitening.transform @ numerical_coef
                component = whitening.whitened_basis @ numerical_coef
            objective = (
                cp.sum_squares(y - component) / n
                + ridge_weight * cp.sum_squares(original_coef)
            )
            problem = cp.Problem(cp.Minimize(objective))
            solve_kwargs = (
                {"eps": 1e-8, "max_iters": 200_000} if solver == "SCS" else {}
            )
            problem.solve(solver=solver, **solve_kwargs)
            results[(formulation, solver)] = (
                float(problem.value),
                np.asarray(component.value),
            )

    raw_component_gap = np.linalg.norm(
        results[("raw", "CLARABEL")][1] - results[("raw", "SCS")][1]
    )
    white_component_gap = np.linalg.norm(
        results[("white", "CLARABEL")][1] - results[("white", "SCS")][1]
    )
    raw_objective_gap = abs(
        results[("raw", "CLARABEL")][0] - results[("raw", "SCS")][0]
    )
    white_objective_gap = abs(
        results[("white", "CLARABEL")][0] - results[("white", "SCS")][0]
    )
    assert white_component_gap < raw_component_gap / 100
    assert white_objective_gap < raw_objective_gap / 1000
