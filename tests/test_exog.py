"""Tests for exogenous (covariate) components."""

import numpy as np
import pytest

from signaldecomp import (
    exog_linear,
    exog_spline,
    make_problem,
    smooth_trend,
    solve,
)
from signaldecomp.spline import default_knots, spline_support_diagnostics

_OPTIMAL = ("optimal", "optimal_inaccurate")


def test_exog_linear_recovers_coefficient():
    rng = np.random.default_rng(0)
    T = 400
    z = rng.standard_normal(T)
    beta_true = 1.7
    y = beta_true * z + 0.02 * rng.standard_normal(T)
    out = solve(make_problem(y, components=[exog_linear(z, role="temp")]))
    assert out["status"] in _OPTIMAL
    assert abs(out["values"]["temp_beta"] - beta_true) < 0.05


def test_exog_spline_recovers_nonlinear_response():
    # The spline coefficient ridge (weight) trades off against the residual:
    # too heavy and real nonlinear structure leaks into the residual. A light
    # ridge with enough knots recovers sin(z)+0.3z over ~2 periods well.
    rng = np.random.default_rng(1)
    T = 600
    z = np.sort(rng.uniform(-3, 3, T))
    f_true = np.sin(z) + 0.3 * z
    y = f_true + 0.03 * rng.standard_normal(T)
    out = solve(
        make_problem(y, components=[exog_spline(z, n_knots=16, weight=1e-5, role="temp")])
    )
    assert out["status"] in _OPTIMAL
    fit = out["values"]["temp"]
    rmse = np.sqrt(np.mean((fit - f_true) ** 2))
    assert rmse < 0.07


def test_exog_spline_no_constant_column():
    # The spline response drops the DC column: coef length == n_knots - 1.
    rng = np.random.default_rng(2)
    T = 300
    z = rng.uniform(0, 10, T)
    y = rng.standard_normal(T)
    out = solve(make_problem(y, components=[exog_spline(z, n_knots=10, role="temp")]))
    assert out["values"]["temp_coef"].shape[0] == 10 - 1


def test_exog_separates_from_trend():
    # An exogenous linear response and a time trend are distinct: recover both.
    rng = np.random.default_rng(3)
    T = 500
    t = np.arange(T)
    z = rng.standard_normal(T)
    trend_true = 0.004 * t
    y = 1.2 * z + trend_true + 0.03 * rng.standard_normal(T)
    out = solve(
        make_problem(
            y,
            components=[exog_linear(z, role="temp"), smooth_trend(weight=1e2, role="trend")],
        )
    )
    assert out["status"] in _OPTIMAL
    assert abs(out["values"]["temp_beta"] - 1.2) < 0.1
    trend_rmse = np.sqrt(np.mean((out["values"]["trend"] - trend_true) ** 2))
    assert trend_rmse < 0.05


def test_exog_length_mismatch_raises():
    rng = np.random.default_rng(4)
    y = rng.standard_normal(100)
    z = rng.standard_normal(50)  # wrong length
    with pytest.raises(ValueError, match="expected"):
        make_problem(y, components=[exog_linear(z, role="temp")])


def test_default_knots_use_only_explicit_fitting_support():
    z = np.linspace(0.0, 1.0, 100)
    z[-1] = 1000.0
    fit_mask = np.ones(100, dtype=bool)
    fit_mask[-1] = False
    knots = default_knots(z, 5, fit_mask=fit_mask)
    assert knots[-1] < 2.0


def test_spline_support_diagnostics_report_counts_and_rank():
    z = np.linspace(-2.0, 2.0, 200)
    fit_mask = np.ones(200, dtype=bool)
    fit_mask[::9] = False
    knots = np.linspace(-1.5, 1.5, 6)
    diagnostics = spline_support_diagnostics(z, knots, fit_mask)
    assert diagnostics.n_fit == fit_mask.sum()
    assert diagnostics.below_boundary_count > 0
    assert diagnostics.above_boundary_count > 0
    assert diagnostics.interval_counts.sum() < diagnostics.n_fit
    assert diagnostics.basis.numerical_rank == len(knots) - 1
