"""Tests for the convex component catalog.

Ports the components.py smoke test (four-way decomposition: residual +
periodic + trend + sparse) and adds checks for the trend/sparse/bounds
factories.
"""

import numpy as np

from signaldecomp import (
    bounded,
    linear_trend,
    make_problem,
    multiperiodic,
    nonneg,
    smooth_trend,
    solve,
    sparse,
)

_OPTIMAL = ("optimal", "optimal_inaccurate")


def _four_component_signal(seed=2):
    rng = np.random.default_rng(seed)
    T, P = 500, 50.0
    t = np.arange(T)
    trend = 1.0 + 0.003 * t
    periodic = 0.6 * np.sin(2 * np.pi * t / P)
    spikes = np.zeros(T)
    idx = rng.choice(T, size=6, replace=False)
    spikes[idx] = rng.uniform(1.5, 2.5, size=6) * rng.choice([-1, 1], 6)
    y = trend + periodic + spikes + 0.05 * rng.standard_normal(T)
    y[150:180] = np.nan
    return y, {"trend": trend, "periodic": periodic, "idx": idx, "P": P}


def test_four_way_decomposition_recovers_components():
    y, truth = _four_component_signal()
    out = solve(
        make_problem(
            y,
            components=[
                multiperiodic(periods=truth["P"], num_harmonics=6, weight=1e-2,
                              role="seasonal"),
                smooth_trend(weight=1e2, order=2, role="trend"),
                sparse(weight=1e-3, role="spikes"),
            ],
        )
    )
    assert out["status"] in _OPTIMAL
    trend_rmse = np.sqrt(np.mean((out["values"]["trend"] - truth["trend"]) ** 2))
    seas_rmse = np.sqrt(np.mean((out["values"]["seasonal"] - truth["periodic"]) ** 2))
    recovered = np.sum(np.abs(out["values"]["spikes"][truth["idx"]]) > 0.5)
    assert trend_rmse < 0.1
    assert seas_rmse < 0.1
    assert recovered >= 5


def test_linear_trend_exposes_slope_and_intercept():
    rng = np.random.default_rng(3)
    T = 300
    t = np.arange(T)
    a, b = 2.0, 0.01
    y = a + b * t + 0.02 * rng.standard_normal(T)
    out = solve(make_problem(y, components=[linear_trend(role="trend")]))
    assert out["status"] in _OPTIMAL
    assert abs(out["values"]["trend_a"] - a) < 0.05
    assert abs(out["values"]["trend_b"] - b) < 0.005


def test_nonneg_wrapper_enforces_lower_bound():
    rng = np.random.default_rng(4)
    T = 200
    y = np.abs(rng.standard_normal(T)) + 0.5
    out = solve(make_problem(y, components=[nonneg(smooth_trend(weight=1e1))]))
    assert out["status"] in _OPTIMAL
    assert np.min(out["values"]["trend"]) > -1e-6


def test_bounded_wrapper_enforces_interval():
    rng = np.random.default_rng(5)
    T = 200
    y = 5.0 + rng.standard_normal(T)
    out = solve(
        make_problem(y, components=[bounded(smooth_trend(weight=1e1), lower=0.0, upper=3.0)])
    )
    assert out["status"] in _OPTIMAL
    vals = out["values"]["trend"]
    assert np.min(vals) > -1e-6
    assert np.max(vals) < 3.0 + 1e-6


def test_all_builders_satisfy_dcp_contract():
    """Every catalog component's loss is convex (or 0) and its expr affine,
    checkable in isolation -- the catalog's central promise (see
    reference/formulation.md, reference/component-catalog.md)."""
    import cvxpy as cp
    from signaldecomp import (
        linear_trend, smooth_trend, pwl_trend, monotone_trend, sparse,
        multiperiodic, exog_linear, exog_spline,
    )
    T = 120
    z = np.linspace(0.0, 1.0, T)
    builders = {
        "linear_trend": linear_trend(role="c"),
        "smooth_trend": smooth_trend(weight=1e1, role="c"),
        "pwl_trend": pwl_trend(weight=1e0, role="c"),
        "monotone_trend": monotone_trend(weight=0.0, role="c"),
        "sparse": sparse(weight=1e0, role="c"),
        "multiperiodic": multiperiodic(periods=24.0, num_harmonics=3, role="c"),
        "exog_linear": exog_linear(z, role="c"),
        "exog_spline": exog_spline(z, role="c"),
    }
    for name, comp in builders.items():
        expr, loss, cons = comp.build(T)
        assert expr.is_affine(), f"{name}: expr not affine"
        if isinstance(loss, cp.Expression):
            assert loss.is_convex() and loss.is_dcp(), f"{name}: loss not convex/DCP"
        for c in cons:
            assert c.is_dcp(), f"{name}: constraint not DCP"


def test_builders_expose_documented_aux_keys():
    """Aux keys promised by the catalog appear in solved values."""
    from signaldecomp import (
        linear_trend, multiperiodic, exog_linear, exog_spline,
    )
    rng = np.random.default_rng(7)
    T = 200
    z = np.linspace(0.0, 1.0, T)
    y = 1.0 + 0.01 * np.arange(T) + 0.3 * z + 0.05 * rng.standard_normal(T)
    cases = [
        (linear_trend(role="tr"), ["tr_a", "tr_b"]),
        (multiperiodic(periods=24.0, num_harmonics=3, role="se"), ["se_theta"]),
        (exog_linear(z, role="ex"), ["ex_beta"]),
        (exog_spline(z, role="ex"), ["ex_coef"]),
    ]
    for comp, keys in cases:
        out = solve(make_problem(y, components=[comp]))
        assert out["status"] in _OPTIMAL
        for k in keys:
            assert k in out["values"], f"missing aux key {k}"
