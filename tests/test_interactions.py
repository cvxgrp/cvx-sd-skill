"""Tests for tensor-product exogenous interaction designs and components."""

import cvxpy as cp
import numpy as np
import pytest

from signaldecomp import (
    BasisRankError,
    Component,
    exog_interaction,
    exog_linear,
    make_interaction_basis,
    make_problem,
    solve,
)
from signaldecomp.spline import default_knots, make_spline_basis


def test_interaction_basis_matches_tsgam_order_and_matrix_contraction():
    left = np.array(
        [[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5], [2.0, -2.0]]
    )
    right = np.array(
        [
            [5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0],
            [2.0, -3.0, 4.0],
            [4.0, 1.0, -2.0],
        ]
    )
    result = make_interaction_basis(left, right)
    expected = (left[:, :, None] * right[:, None, :]).reshape(left.shape[0], -1)
    coefficient_matrix = np.array([[0.5, -1.0, 2.0], [3.0, 0.25, -0.75]])

    assert result.design.shape == (4, 6)
    assert result.left_width == 2
    assert result.right_width == 3
    assert np.array_equal(result.design, expected)
    assert np.allclose(
        result.design @ coefficient_matrix.reshape(-1, order="C"),
        np.sum((left @ coefficient_matrix) * right, axis=1),
    )
    assert not result.design.flags.writeable
    assert not result.valid_mask.flags.writeable


def test_interaction_basis_accepts_vectors_and_zero_fills_unavailable_rows():
    left = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    right = np.array([2.0, 3.0, np.inf, 5.0, 1.0, 4.0, 7.0, 3.0])
    result = make_interaction_basis(left, right)

    assert result.design.shape == (8, 1)
    assert np.array_equal(
        result.valid_mask,
        [True, False, False, True, True, True, True, True],
    )
    assert np.array_equal(
        result.design[:, 0],
        [2.0, 0.0, 0.0, 20.0, 5.0, 24.0, 49.0, 24.0],
    )

    built = make_problem(np.arange(8.0), [exog_interaction(left, right)])
    assert np.array_equal(built["fit_mask"], result.valid_mask)


def test_interaction_component_retains_an_immutable_design_snapshot():
    t = np.arange(1.0, 7.0)
    left = np.column_stack([t, t**2])
    right = np.linspace(1.0, 2.0, 6)
    component = exog_interaction(left, right)
    interaction_basis = component.metadata["interaction_basis"]
    stored_design = interaction_basis.design.copy()
    stored_left = interaction_basis.left_basis.copy()
    stored_right = interaction_basis.right_basis.copy()
    left[:] = -100.0
    right[:] = np.nan

    assert np.array_equal(interaction_basis.design, stored_design)
    assert np.array_equal(interaction_basis.left_basis, stored_left)
    assert np.array_equal(interaction_basis.right_basis, stored_right)
    assert not interaction_basis.left_basis.flags.writeable
    assert not interaction_basis.right_basis.flags.writeable


@pytest.mark.parametrize(
    "offset_form",
    ["explicit_column", "reparameterized_span"],
)
@pytest.mark.parametrize("constant_side", ["left", "right"])
def test_interaction_rejects_a_constant_direction_in_either_factor(
    constant_side,
    offset_form,
):
    x = np.linspace(-1.0, 1.0, 20)
    factor_with_offset = (
        np.column_stack([np.ones(x.size), x])
        if offset_form == "explicit_column"
        else np.column_stack([1.0 + x, 1.0 - x])
    )
    other = np.column_stack([x**2, x**3])
    left, right = (
        (factor_with_offset, other)
        if constant_side == "left"
        else (other, factor_with_offset)
    )

    with pytest.raises(ValueError, match=f"{constant_side}_basis.*offset-free"):
        make_interaction_basis(left, right)


def test_interaction_rejects_overflowing_products_before_component_build():
    left = np.array([1e308])
    right = np.array([2.0])
    with pytest.raises(ValueError, match="overflowed.*rescale"):
        exog_interaction(left, right)


def test_interaction_rejects_underflowing_nonzero_products():
    left = np.array([1e-308])
    right = np.array([1e-308])
    with pytest.raises(ValueError, match="underflowed.*rescale"):
        exog_interaction(left, right)


def test_interaction_only_audit_uses_the_final_fitted_support():
    rng = np.random.default_rng(106)
    n_fit = 50
    left = np.concatenate([np.ones(n_fit), np.linspace(2.0, 3.0, n_fit)])
    right = rng.standard_normal(2 * n_fit)
    y = rng.standard_normal(2 * n_fit)
    y[n_fit:] = np.nan

    component = exog_interaction(left, right, role="cross")
    with pytest.raises(ValueError, match="not interaction-only.*final fit_mask"):
        make_problem(y, [component])


def test_interaction_only_audit_includes_other_component_availability():
    rng = np.random.default_rng(107)
    n_fit = 50
    left = np.concatenate([np.ones(n_fit), np.linspace(2.0, 3.0, n_fit)])
    right = rng.standard_normal(2 * n_fit)
    valid_mask = np.zeros(2 * n_fit, dtype=bool)
    valid_mask[:n_fit] = True

    def build_gate(T):
        return cp.Constant(np.zeros(T)), 0, []

    gate = Component("availability_gate", build_gate, valid_mask=valid_mask)
    component = exog_interaction(left, right, role="cross")
    with pytest.raises(ValueError, match="not interaction-only.*final fit_mask"):
        make_problem(rng.standard_normal(2 * n_fit), [component, gate])


def test_interaction_only_audit_rejects_reciprocal_factor_intercept():
    left = np.linspace(1.0, 2.0, 100)
    right = 1.0 / left
    component = exog_interaction(left, right, role="cross")

    with pytest.raises(ValueError, match="overlaps.*intercept"):
        make_problem(np.zeros(100), [component])


def test_offset_free_spline_factor_bases_are_accepted():
    left_driver = np.linspace(-2.0, 2.0, 100)
    right_driver = np.sin(np.linspace(0.0, 3.0, 100))
    left = make_spline_basis(
        left_driver,
        default_knots(left_driver, 5),
        include_offset=False,
    )
    right = make_spline_basis(
        right_driver,
        default_knots(right_driver, 5),
        include_offset=False,
    )

    result = make_interaction_basis(left, right)
    assert result.design.shape == (100, 16)


def test_interaction_component_recovers_coefficients_and_signal():
    rng = np.random.default_rng(100)
    left = rng.standard_normal((300, 2))
    right = rng.standard_normal((300, 3))
    coefficient_matrix = np.array([[0.8, -0.3, 0.5], [-1.1, 0.2, 0.7]])
    y = np.sum((left @ coefficient_matrix) * right, axis=1)

    out = solve(
        make_problem(
            y,
            [
                exog_interaction(
                    left,
                    right,
                    role="weather_interaction",
                    factor_names=("irradiance", "temperature"),
                )
            ],
        )
    )

    assert np.allclose(out["values"]["weather_interaction"], y, atol=1e-6)
    assert np.allclose(
        out["values"]["weather_interaction_coef"],
        coefficient_matrix,
        atol=1e-6,
    )
    assert out["component_metadata"]["weather_interaction"]["factor_names"] == (
        "irradiance",
        "temperature",
    )


def test_interaction_is_separate_from_both_main_effect_roles():
    rng = np.random.default_rng(101)
    left = rng.standard_normal(400)
    right = rng.standard_normal(400)
    y = 1.2 * left - 0.8 * right + 0.7 * left * right
    out = solve(
        make_problem(
            y,
            [
                exog_linear(left, role="left_main"),
                exog_linear(right, role="right_main"),
                exog_interaction(left, right, role="cross"),
            ],
        )
    )

    assert abs(out["values"]["left_main_beta"] - 1.2) < 1e-6
    assert abs(out["values"]["right_main_beta"] + 0.8) < 1e-6
    assert np.allclose(out["values"]["cross_coef"], [[0.7]], atol=1e-6)
    assert set(("left_main", "right_main", "cross")) <= out["values"].keys()


def test_whitened_interaction_preserves_solution_penalty_and_coefficients():
    rng = np.random.default_rng(102)
    left = rng.standard_normal((260, 2)) * np.array([1.0, 1e3])
    right = rng.standard_normal((260, 2)) * np.array([1.0, 1e-2])
    coefficient_matrix = np.array([[0.6, -0.2], [0.001, 0.03]])
    y = np.sum((left @ coefficient_matrix) * right, axis=1)
    fit_mask = np.ones(y.size, dtype=bool)
    kwargs = dict(weight=3e-3, role="cross")

    raw = solve(make_problem(y, [exog_interaction(left, right, **kwargs)]))
    white = solve(
        make_problem(
            y,
            [
                exog_interaction(
                    left,
                    right,
                    **kwargs,
                    whiten=True,
                    fit_mask=fit_mask,
                )
            ],
        )
    )

    assert np.allclose(raw["values"]["cross"], white["values"]["cross"], atol=1e-5)
    assert np.allclose(
        raw["values"]["cross_coef"], white["values"]["cross_coef"], atol=2e-5
    )
    whitening = white["component_metadata"]["cross"]["whitening"]
    recovered = whitening.recover_coefficients(
        white["values"]["cross_numerical_coef"]
    ).reshape(2, 2, order="C")
    assert np.allclose(recovered, white["values"]["cross_coef"])
    assert np.isclose(
        np.sum(recovered**2), np.sum(white["values"]["cross_coef"] ** 2)
    )
    assert np.isclose(raw["problem"].value, white["problem"].value, rtol=1e-7)


def test_whitened_interaction_requires_exact_immutable_fit_mask():
    rng = np.random.default_rng(103)
    left = rng.standard_normal((100, 2))
    right = rng.standard_normal((100, 2))
    caller_mask = np.ones(100, dtype=bool)
    component = exog_interaction(
        left,
        right,
        whiten=True,
        fit_mask=caller_mask,
    )
    stored_mask = component.parameterization_mask.copy()
    caller_mask[:4] = False
    assert np.array_equal(component.parameterization_mask, stored_mask)
    with pytest.raises(ValueError, match="read-only"):
        component.parameterization_mask[0] = False

    y = rng.standard_normal(100)
    y[-1] = np.nan
    with pytest.raises(ValueError, match="exactly match"):
        make_problem(y, [component])


def test_interaction_whitening_rejects_unavailable_and_deficient_fit_rows():
    rng = np.random.default_rng(104)
    left = rng.standard_normal((80, 2))
    right = rng.standard_normal((80, 2))
    left[5, 0] = np.nan
    with pytest.raises(ValueError, match="unavailable"):
        exog_interaction(
            left,
            right,
            whiten=True,
            fit_mask=np.ones(80, dtype=bool),
        )

    duplicated = np.column_stack([right[:, 0], right[:, 0]])
    fit_mask = np.ones(80, dtype=bool)
    fit_mask[5] = False
    with pytest.raises(BasisRankError, match="interaction component"):
        exog_interaction(
            left,
            duplicated,
            whiten=True,
            fit_mask=fit_mask,
        )


def test_interaction_component_and_complete_problem_are_dcp():
    rng = np.random.default_rng(105)
    left = rng.standard_normal((60, 2))
    right = rng.standard_normal((60, 2))
    component = exog_interaction(left, right, weight=0.2)
    expression, loss, constraints = component.build(60)
    built = make_problem(rng.standard_normal(60), [component])

    assert expression.is_affine()
    assert loss.is_convex() and loss.is_dcp()
    assert all(constraint.is_dcp() for constraint in constraints)
    assert built["problem"].is_dcp()


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (
            lambda: make_interaction_basis(np.ones((4, 2, 1)), np.ones(4)),
            ValueError,
            "1-D or 2-D",
        ),
        (
            lambda: make_interaction_basis(np.ones(4), np.ones(5)),
            ValueError,
            "same number",
        ),
        (
            lambda: make_interaction_basis(np.ones((4, 0)), np.ones(4)),
            ValueError,
            "at least one",
        ),
        (
            lambda: exog_interaction(np.ones(4), np.ones(4), weight=np.nan),
            ValueError,
            "weight",
        ),
        (
            lambda: exog_interaction(
                np.ones(4), np.ones(4), factor_names=("one",)
            ),
            ValueError,
            "factor_names",
        ),
        (
            lambda: exog_interaction(
                np.ones(4), np.ones(4), factor_names="lr"
            ),
            ValueError,
            "factor_names",
        ),
        (
            lambda: exog_interaction(
                np.arange(4.0), np.arange(4.0) ** 2, whiten=True
            ),
            ValueError,
            "fit_mask",
        ),
    ],
)
def test_interaction_argument_validation(call, error, match):
    with pytest.raises(error, match=match):
        call()
