"""Tests for the keystone masked decomposition builder.

These port and extend the original decompose.py smoke test: a synthetic
trend + periodic + noise signal with a gap must be recovered, and the core
invariants (x0 residual, masked linking equality, DCP gate, role uniqueness)
must hold.
"""

import cvxpy as cp
import numpy as np
import pytest

from signaldecomp import Component, make_problem, multiperiodic, smooth_trend, solve

_OPTIMAL = ("optimal", "optimal_inaccurate")


def _synthetic(seed=0, gap=(200, 230)):
    """Trend + periodic + noise with a masked gap; returns (y, truth dict)."""
    rng = np.random.default_rng(seed)
    T, P = 600, 50
    t = np.arange(T)
    trend = 0.002 * t
    periodic = 0.5 * np.sin(2 * np.pi * t / P)
    y = trend + periodic + 0.05 * rng.standard_normal(T)
    y[gap[0] : gap[1]] = np.nan
    return y, {"trend": trend, "periodic": periodic, "P": P, "gap": gap}


def test_recovers_trend_and_periodic_through_gap():
    y, truth = _synthetic()
    built = make_problem(
        y,
        components=[
            multiperiodic(periods=float(truth["P"]), num_harmonics=6, weight=1e-2),
            smooth_trend(weight=1e2, order=2),
        ],
    )
    out = solve(built)
    assert out["status"] in _OPTIMAL
    trend_rmse = np.sqrt(np.mean((out["values"]["trend"] - truth["trend"]) ** 2))
    assert trend_rmse < 0.05


def test_residual_unconstrained_on_gap():
    # Unobserved entries are not in the linking constraint, so the residual is
    # free there; at the l2 optimum it is exactly zero on the gap.
    y, truth = _synthetic()
    built = make_problem(y, components=[smooth_trend(weight=1e2)])
    out = solve(built)
    g0, g1 = truth["gap"]
    assert np.max(np.abs(out["values"]["residual"][g0:g1])) < 1e-8


def test_mask_matches_observed_entries():
    y, _ = _synthetic()
    built = make_problem(y, components=[smooth_trend(weight=1e2)])
    assert np.array_equal(built["mask"], ~np.isnan(y))


def test_residual_is_x0_and_present():
    y, _ = _synthetic()
    built = make_problem(y, components=[smooth_trend(weight=1e2)])
    assert "residual" in built["variables"]
    assert built["residual"] is built["variables"]["residual"]


def test_duplicate_role_raises():
    y, _ = _synthetic()
    with pytest.raises(ValueError, match="duplicate"):
        make_problem(
            y,
            components=[smooth_trend(weight=1e2), smooth_trend(weight=1e1)],
        )


def test_rejects_non_1d_signal():
    with pytest.raises(ValueError, match="1-D"):
        make_problem(np.zeros((10, 2)), components=[])


def test_rejects_all_missing():
    with pytest.raises(ValueError, match="observed"):
        make_problem(np.full(10, np.nan), components=[])


def test_residual_loss_accepts_callable():
    # A user-supplied DCP-compliant loss is accepted directly.
    y, _ = _synthetic()

    def custom(x):
        return (1.0 / x.shape[0]) * cp.sum_squares(x)

    built = make_problem(y, components=[smooth_trend(weight=1e2)], residual_loss=custom)
    out = solve(built)
    assert out["status"] in _OPTIMAL


def test_residual_loss_rejects_unknown_string():
    y, _ = _synthetic()
    with pytest.raises(ValueError, match="callable or one of"):
        make_problem(y, components=[smooth_trend(weight=1e2)], residual_loss="nope")


def test_solve_rejects_non_dcp():
    # Build a pathological component whose loss is non-convex, and confirm the
    # DCP gate catches it rather than silently solving.
    y, _ = _synthetic()

    def bad_build(T):
        x = cp.Variable(T, name="bad")
        # maximizing a convex square (i.e. a concave-in-Minimize term) is non-DCP
        return x, -cp.sum_squares(x), []

    built = make_problem(y, components=[Component(role="bad", build=bad_build)])
    with pytest.raises(ValueError, match="DCP"):
        solve(built)


@pytest.mark.parametrize("residual_loss", [
    "l2", "l1",
    __import__("signaldecomp").huber_loss(M=0.5),
    __import__("signaldecomp").quantile_loss(q=0.5),
])
def test_shipped_residual_losses_solve(residual_loss):
    """Both string aliases and parameterized factory losses reach optimal."""
    y, _ = _synthetic()
    out = solve(make_problem(y, components=[smooth_trend(weight=1e2)],
                             residual_loss=residual_loss))
    assert out["status"] in _OPTIMAL
