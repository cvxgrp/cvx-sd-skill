"""Tests for exact full-rank basis whitening and diagnostics."""

import cvxpy as cp
import numpy as np
import pytest

from signaldecomp.basis_numerics import (
    BasisRankError,
    basis_rank_diagnostics,
    whiten_basis,
    whiten_basis_by_blocks,
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


def test_block_whitening_preserves_expression_coefficients_and_general_penalty():
    rng = np.random.default_rng(20)
    blocks = (
        rng.standard_normal((120, 2)) * np.array([1.0, 1e4]),
        rng.standard_normal((120, 3)) * np.array([1e-3, 1.0, 1e2]),
    )
    fit_mask = np.ones(120, dtype=bool)
    fit_mask[::8] = False
    result = whiten_basis_by_blocks(
        blocks,
        fit_mask,
        block_names=("main", "interaction"),
    )
    beta = rng.standard_normal(5)
    gamma = np.linalg.solve(result.transform, beta)
    penalty_operator = rng.standard_normal((4, 5))

    assert np.allclose(result.raw_basis @ beta, result.whitened_basis @ gamma)
    assert np.allclose(result.recover_coefficients(gamma), beta)
    assert np.allclose(
        penalty_operator @ beta,
        penalty_operator @ result.transform @ gamma,
    )
    assert result.training_gram_error < 1e-10
    assert result.block_names == ("main", "interaction")
    assert result.block_slices == (slice(0, 2), slice(2, 5))
    assert not result.raw_basis.flags.writeable
    assert not result.whitened_basis.flags.writeable
    assert not result.transform.flags.writeable


def test_block_whitening_complete_transform_matches_both_stages():
    rng = np.random.default_rng(21)
    blocks = (rng.standard_normal((90, 2)), rng.standard_normal((90, 3)))
    fit_mask = np.ones(90, dtype=bool)
    result = whiten_basis_by_blocks(blocks, fit_mask)
    block_transform = np.zeros((5, 5))
    for block_slice, whitening in zip(
        result.block_slices, result.block_whitenings, strict=True
    ):
        block_transform[block_slice, block_slice] = whitening.transform

    assert result.block_names == ("block_0", "block_1")
    assert np.allclose(
        result.transform,
        block_transform @ result.joint_whitening.transform,
    )
    assert np.allclose(
        result.whitened_basis,
        result.raw_basis @ result.transform,
    )
    assert result.diagnostics is result.joint_whitening.diagnostics


def test_block_whitening_recomputes_tolerance_for_joint_coordinates():
    # Each raw block has singular values 10, so an absolute raw-coordinate
    # threshold of 1.1 is admissible. First-stage whitening makes the joint
    # singular values 1, where reusing 1.1 would incorrectly report rank zero.
    block_1 = np.vstack([10.0 * np.eye(2), np.zeros((2, 2))])
    block_2 = np.vstack([np.zeros((2, 2)), 10.0 * np.eye(2)])
    fit_mask = np.ones(4, dtype=bool)

    result = whiten_basis_by_blocks(
        (block_1, block_2),
        fit_mask,
        rank_tolerance=1.1,
    )
    assert result.diagnostics.numerical_rank == 4
    assert result.diagnostics.rank_tolerance < 1.0

    with pytest.raises(BasisRankError, match="joint concatenation"):
        whiten_basis_by_blocks(
            (block_1, block_2),
            fit_mask,
            rank_tolerance=1.1,
            joint_rank_tolerance=1.1,
        )


def test_block_whitening_fails_with_named_block_or_joint_context():
    rng = np.random.default_rng(22)
    good = rng.standard_normal((60, 2))
    bad = np.column_stack([good[:, 0], good[:, 0]])
    fit_mask = np.ones(60, dtype=bool)
    with pytest.raises(BasisRankError, match="block 'duplicated'") as excinfo:
        whiten_basis_by_blocks(
            (good, bad),
            fit_mask,
            block_names=("good", "duplicated"),
            context="weather response",
        )
    assert excinfo.value.diagnostics is not None

    with pytest.raises(BasisRankError, match="joint concatenation") as excinfo:
        whiten_basis_by_blocks(
            (good, good.copy()),
            fit_mask,
            block_names=("first", "second"),
        )
    assert excinfo.value.diagnostics is not None


@pytest.mark.parametrize(
    ("blocks", "names", "match"),
    [
        ((), None, "at least one"),
        ((np.ones(4),), None, "2-D"),
        ((np.ones((4, 0)),), None, "at least one column"),
        ((np.ones((4, 1)), np.ones((5, 1))), None, "same number of rows"),
        ((np.eye(4),), ("first", "extra"), "length"),
        ((np.eye(4), np.eye(4)), ("same", "same"), "unique"),
        ((np.eye(4),), ("",), "non-empty strings"),
        ((np.eye(4),), "a", "not a string"),
    ],
)
def test_block_whitening_argument_validation(blocks, names, match):
    with pytest.raises(ValueError, match=match):
        whiten_basis_by_blocks(
            blocks,
            np.ones(4, dtype=bool),
            block_names=names,
        )


def test_block_whitening_rejects_nonfinite_input_and_bad_exact_mask():
    basis = np.eye(5)
    basis[0, 0] = np.nan
    with pytest.raises(ValueError, match="block 'driver'.*finite"):
        whiten_basis_by_blocks(
            (basis,),
            np.ones(5, dtype=bool),
            block_names=("driver",),
        )
    with pytest.raises(TypeError, match="boolean"):
        whiten_basis_by_blocks((np.eye(5),), np.ones(5))


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
