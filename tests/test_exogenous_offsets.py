"""Tests for offset exogenous design matrices and components."""

import cvxpy as cp
import numpy as np
import pytest

from signaldecomp import Component, exog_linear, exog_spline, make_problem, solve
from signaldecomp.exogenous import make_offset_basis, offset_source_mask
from signaldecomp.spline import default_knots, make_spline_basis


def test_offset_basis_uses_explicit_conventional_orientation():
    values = np.arange(5.0)
    result = make_offset_basis(values, offsets=(1, 0, -1))
    expected = np.column_stack(
        [
            [0.0, 0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0, 0.0],
        ]
    )
    assert np.array_equal(result.design, expected)
    assert np.array_equal(result.valid_mask, [False, True, True, True, False])


def test_negated_tsgam_lags_reproduce_implemented_shift():
    values = np.arange(7.0)[:, None]
    tsgam_lags = (-2, 0, 1)
    tsgam_blocks = []
    for lag in tsgam_lags:
        block = np.roll(values.copy(), -lag, axis=0)
        if lag > 0:
            block[-lag:] = np.nan
        elif lag < 0:
            block[:-lag] = np.nan
        tsgam_blocks.append(block)
    expected = np.hstack(tsgam_blocks)
    actual = make_offset_basis(values, offsets=tuple(-lag for lag in tsgam_lags))
    assert np.allclose(actual.design[actual.valid_mask], expected[actual.valid_mask])


def test_offset_source_mask_maps_target_fit_rows_back_to_driver_rows():
    fit_mask = np.array([False, False, True, True, False, False])
    # Targets 2 and 3 consume sources 1,2 for offset +1 and 3,4 for offset -1.
    assert np.array_equal(
        offset_source_mask(fit_mask, offsets=(1, -1)),
        [False, True, True, True, True, False],
    )


def test_zero_offset_linear_matches_legacy_scalar_formulation():
    rng = np.random.default_rng(70)
    z = rng.standard_normal(180)
    y = 1.8 * z + 0.05 * rng.standard_normal(z.size)

    legacy = Component(role="driver", build=None)

    def build(T):
        beta = cp.Variable(name="driver_beta")
        legacy.aux["driver_beta"] = beta
        return beta * z, 0.03 * cp.square(beta), []

    legacy.build = build
    old = solve(make_problem(y, [legacy]))
    new = solve(
        make_problem(
            y,
            [exog_linear(z, weight=0.03, offsets=(0,), role="driver")],
        )
    )
    assert np.allclose(old["values"]["driver"], new["values"]["driver"])
    assert np.isclose(old["values"]["driver_beta"], new["values"]["driver_beta"])


def test_zero_offset_spline_matches_legacy_formulation():
    rng = np.random.default_rng(71)
    z = rng.uniform(-2.0, 2.0, 220)
    y = np.sin(z) + 0.03 * rng.standard_normal(z.size)
    knots = default_knots(z, 6)
    basis = make_spline_basis(z, knots, include_offset=False)

    legacy = Component(role="driver", build=None)

    def build(T):
        coef = cp.Variable(basis.shape[1], name="driver_coef")
        legacy.aux["driver_coef"] = coef
        return basis @ coef, 2e-3 * cp.sum_squares(coef), []

    legacy.build = build
    old = solve(make_problem(y, [legacy]))
    new = solve(
        make_problem(
            y,
            [
                exog_spline(
                    z,
                    knots=knots,
                    weight=2e-3,
                    offsets=(0,),
                    role="driver",
                )
            ],
        )
    )
    assert np.allclose(old["values"]["driver"], new["values"]["driver"])
    assert np.allclose(old["values"]["driver_coef"], new["values"]["driver_coef"])


def test_offset_linear_recovers_past_response_and_masks_boundary():
    rng = np.random.default_rng(12)
    z = rng.standard_normal(300)
    y = np.zeros(300)
    y[1:] = 2.4 * z[:-1]
    built = make_problem(y, [exog_linear(z, offsets=(1,), role="driver")])
    assert not built["fit_mask"][0]
    out = solve(built)
    assert abs(out["values"]["driver_beta"] - 2.4) < 1e-6


def test_solved_linear_coefficients_follow_requested_offset_order():
    rng = np.random.default_rng(72)
    z = rng.standard_normal(300)
    offsets = (2, 0, 1)
    expected_beta = np.array([1.4, -0.6, 0.25])
    offset_basis = make_offset_basis(z, offsets)
    y = offset_basis.design @ expected_beta
    out = solve(make_problem(y, [exog_linear(z, offsets=offsets, role="driver")]))
    assert np.allclose(out["values"]["driver_beta"], expected_beta, atol=1e-6)


def test_missing_exogenous_value_restricts_fit_mask():
    z = np.arange(10.0)
    z[4] = np.nan
    built = make_problem(np.arange(10.0), [exog_linear(z)])
    assert built["mask"].all()
    assert not built["fit_mask"][4]
    assert built["fit_mask"].sum() == 9


def test_lag_smoothing_reduces_offset_coefficient_variation():
    rng = np.random.default_rng(13)
    z = rng.standard_normal(500)
    y = 2.0 * z + 0.02 * rng.standard_normal(500)
    loose = solve(
        make_problem(
            y,
            [exog_linear(z, offsets=(0, 1, 2), role="driver")],
        )
    )
    stiff = solve(
        make_problem(
            y,
            [
                exog_linear(
                    z,
                    offsets=(0, 1, 2),
                    lag_smooth_weight=1e3,
                    role="driver",
                )
            ],
        )
    )
    assert np.linalg.norm(np.diff(stiff["values"]["driver_beta"])) < np.linalg.norm(
        np.diff(loose["values"]["driver_beta"])
    )


def test_whitened_spline_preserves_unwhitened_solution_and_coefficients():
    rng = np.random.default_rng(14)
    z = np.linspace(-2.0, 2.0, 240)
    y = np.sin(z) + 0.1 * z + 0.01 * rng.standard_normal(z.size)
    fit_mask = np.ones(z.size, dtype=bool)
    kwargs = dict(n_knots=7, weight=2e-3, role="driver")
    raw = solve(make_problem(y, [exog_spline(z, **kwargs)]))
    white = solve(
        make_problem(
            y,
            [exog_spline(z, **kwargs, whiten=True, fit_mask=fit_mask)],
        )
    )
    assert np.allclose(raw["values"]["driver"], white["values"]["driver"], atol=2e-5)
    assert np.allclose(
        raw["values"]["driver_coef"],
        white["values"]["driver_coef"],
        atol=2e-4,
    )
    whitening = white["component_metadata"]["driver"]["whitening"]
    recovered = whitening.recover_coefficients(
        white["values"]["driver_numerical_coef"]
    )
    assert np.allclose(recovered, white["values"]["driver_coef"])


def test_whitened_spline_rejects_fit_mask_mismatch():
    z = np.linspace(-1.0, 1.0, 80)
    y = np.cos(z)
    fit_mask = np.ones(80, dtype=bool)
    component = exog_spline(z, n_knots=5, whiten=True, fit_mask=fit_mask)
    y[-1] = np.nan
    with pytest.raises(ValueError, match="exactly match"):
        make_problem(y, [component])


def test_spline_parameterization_mask_is_an_immutable_snapshot():
    z = np.linspace(-1.0, 1.0, 100)
    caller_mask = np.ones(100, dtype=bool)
    component = exog_spline(
        z,
        n_knots=5,
        whiten=True,
        fit_mask=caller_mask,
    )
    stored_mask = component.parameterization_mask.copy()
    caller_mask[:10] = False

    assert np.array_equal(component.parameterization_mask, stored_mask)
    with pytest.raises(ValueError, match="read-only"):
        component.parameterization_mask[0] = False


def test_whitened_spline_composes_with_another_component_validity_mask():
    rng = np.random.default_rng(73)
    T = 260
    spline_driver = rng.uniform(-2.0, 2.0, T)
    linear_driver = rng.standard_normal(T)
    linear_driver[100] = np.nan
    y = np.sin(spline_driver) + 0.3 * np.nan_to_num(linear_driver)
    y_train = y.copy()
    y_train[180:195] = np.nan
    spline_valid = make_offset_basis(spline_driver, offsets=(0, 1)).valid_mask
    linear_valid = make_offset_basis(linear_driver, offsets=(0,)).valid_mask
    fit_mask = np.isfinite(y_train) & spline_valid & linear_valid

    built = make_problem(
        y_train,
        [
            exog_spline(
                spline_driver,
                n_knots=5,
                offsets=(0, 1),
                whiten=True,
                fit_mask=fit_mask,
                role="spline_driver",
            ),
            exog_linear(linear_driver, role="linear_driver"),
        ],
    )
    assert np.array_equal(built["fit_mask"], fit_mask)
    out = solve(built)
    assert np.all(np.isfinite(out["values"]["spline_driver_coef"]))


def test_spline_knots_use_offset_source_rows_from_fit_mask():
    z = np.arange(10.0)
    z[-1] = 1000.0
    y = np.arange(10.0)
    fit_mask = np.ones(10, dtype=bool)
    fit_mask[[0, -1]] = False
    component = exog_spline(z, n_knots=4, offsets=(1,), fit_mask=fit_mask)
    # Fitted targets 1..8 consume source rows 0..7, never the extreme row 9.
    assert component.metadata["knots"][-1] == 7.0
    y[[0, -1]] = np.nan
    make_problem(y, [component])


def test_knot_policy_requires_and_receives_training_source_support():
    z = np.arange(12.0)
    with pytest.raises(ValueError, match="fit_mask is required"):
        exog_spline(z, knot_policy=lambda values, n: np.linspace(0, 1, n))

    fit_mask = np.ones(12, dtype=bool)
    fit_mask[[0, -1]] = False
    seen = {}

    def policy(values, n_knots):
        seen["values"] = values.copy()
        return np.linspace(values.min(), values.max(), n_knots)

    component = exog_spline(
        z,
        n_knots=4,
        offsets=(1,),
        fit_mask=fit_mask,
        knot_policy=policy,
    )
    assert np.array_equal(seen["values"], np.arange(10.0))
    assert component.metadata["knots"][-1] == 9.0


def test_multi_offset_whitening_preserves_ridge_and_lag_smoothing():
    rng = np.random.default_rng(15)
    z = rng.uniform(-2.0, 2.0, 320)
    y = np.sin(z) + 0.3 * np.roll(z, 1) + 0.02 * rng.standard_normal(z.size)
    fit_mask = np.ones(z.size, dtype=bool)
    fit_mask[0] = False
    kwargs = dict(
        n_knots=5,
        weight=1e-3,
        role="driver",
        offsets=(0, 1),
        lag_smooth_weight=0.2,
    )
    raw = solve(make_problem(y, [exog_spline(z, **kwargs)]))
    white = solve(
        make_problem(
            y,
            [exog_spline(z, **kwargs, whiten=True, fit_mask=fit_mask)],
        )
    )
    assert np.allclose(raw["values"]["driver"], white["values"]["driver"], atol=2e-5)
    assert np.allclose(
        raw["values"]["driver_coef"],
        white["values"]["driver_coef"],
        atol=2e-4,
    )


def test_spline_lag_smoothing_reduces_offset_coefficient_variation():
    rng = np.random.default_rng(81)
    z = rng.uniform(-2.0, 2.0, 500)
    y = np.sin(z) + 0.02 * rng.standard_normal(z.size)
    kwargs = dict(
        n_knots=5,
        weight=1e-4,
        offsets=(0, 1, 2),
        role="driver",
    )
    loose = solve(make_problem(y, [exog_spline(z, **kwargs)]))
    stiff = solve(
        make_problem(
            y,
            [exog_spline(z, **kwargs, lag_smooth_weight=1e3)],
        )
    )
    loose_variation = np.linalg.norm(
        np.diff(loose["values"]["driver_coef"], axis=1)
    )
    stiff_variation = np.linalg.norm(
        np.diff(stiff["values"]["driver_coef"], axis=1)
    )
    assert stiff_variation < loose_variation / 100
