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
