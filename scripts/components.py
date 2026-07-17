"""Convex component catalog for signal decomposition.

Each factory returns a decompose.Component -- a build(T) callable yielding
(expr, loss, constraints) -- so components compose uniformly in
decompose.make_problem. All components here are convex (V1 scope).

A note on what is not here: the residual losses (mean-square-small, l1, Huber,
quantile) are properties of the residual x1 and are selected via
make_problem(..., residual_loss=...); they are not structural components. A
structural component captures a kind of signal (a trend, a seasonal shape,
sparse spikes), and its loss expresses how plausible that shape is; the residual
absorbs whatever is left.

The truncated-Fourier periodic component (``multiperiodic``) lives in
periodic.py and is re-exported here for convenience.

Non-convex classes from the framework (Boolean/finite-set, cardinality/r-sparse,
single-jump, Markov, log-Huber) are intentionally omitted; where a convex
relaxation exists (e.g. l1 for cardinality) use the corresponding component here.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from decompose import Component
from periodic import multiperiodic  # noqa: F401  (re-exported)


def linear_trend(role: str = "trend", slope_weight: float = 0.0) -> Component:
    """Affine trend a + b * t.

    Exposes the intercept and slope as scalar aux quantities (<role>_a,
    <role>_b); the slope is the natural per-sample rate of change.

    Parameters
    ----------
    role : str
        Component role name.
    slope_weight : float
        Optional ridge penalty on the slope magnitude (0 = unpenalized).
    """
    comp = Component(role=role, build=None)

    def build(T):
        t = np.arange(T, dtype=float)
        coef = cp.Variable(2, name=f"{role}_coef")
        expr = coef[0] + coef[1] * t
        loss = slope_weight * cp.square(coef[1])
        comp.aux[f"{role}_a"] = coef[0]
        comp.aux[f"{role}_b"] = coef[1]
        return expr, loss, []

    comp.build = build
    return comp


def smooth_trend(weight, order=2, role="trend"):
    """Mean-square-smooth trend penalizing the order-th difference.

    order=2 (default) penalizes curvature, giving a smooth trend; order=1
    penalizes slope, damping level changes.
    """
    comp = Component(role=role, build=None)

    def build(T):
        x = cp.Variable(T, name=role)
        loss = weight * cp.sum_squares(cp.diff(x, k=order))
        return x, loss, []

    comp.build = build
    return comp


def pwl_trend(weight, role="trend"):
    """Piecewise-linear trend via l1 penalty on the second difference.

    The l1 second difference (l1 trend filtering) yields a trend that is
    piecewise linear with a small number of knots -- an interpretable,
    breakpoint-style trend.
    """
    comp = Component(role=role, build=None)

    def build(T):
        x = cp.Variable(T, name=role)
        loss = weight * cp.norm1(cp.diff(x, k=2))
        return x, loss, []

    comp.build = build
    return comp


def monotone_trend(weight=0.0, increasing=False, role="trend"):
    """Monotone (isotonic) trend: non-increasing by default.

    Used for quantities like cumulative degradation or wear that cannot reverse
    direction. An optional second-difference smoothness penalty (weight)
    regularizes the shape.

    Parameters
    ----------
    weight : float
        Second-difference smoothness weight (0 = unregularized monotone fit).
    increasing : bool
        If True, constrain non-decreasing; else non-increasing (default).
    role : str
        Component role name.
    """
    comp = Component(role=role, build=None)

    def build(T):
        x = cp.Variable(T, name=role)
        d = cp.diff(x)
        cons = [d >= 0] if increasing else [d <= 0]
        loss = weight * cp.sum_squares(cp.diff(x, k=2)) if weight else 0
        return x, loss, cons

    comp.build = build
    return comp


def sparse(weight, role="sparse"):
    """Component-wise sparse component via l1 penalty.

    Yields a signal that is zero at most entries and nonzero at a few -- the
    convex surrogate for cardinality. Use for sparse outliers, spikes, or rare
    events.
    """
    comp = Component(role=role, build=None)

    def build(T):
        x = cp.Variable(T, name=role)
        loss = weight * cp.norm1(x)
        return x, loss, []

    comp.build = build
    return comp


def bounded(inner, lower=None, upper=None):
    """Wrap a component with box constraints lower <= x <= upper.

    Adds elementwise bounds to an existing component's expression without
    changing its loss. Either bound may be None (one-sided). For example,
    bounded(smooth_trend(...), lower=0.0) gives a nonnegative smooth trend.
    """
    comp = Component(role=inner.role, build=None, aux=inner.aux)

    def build(T):
        expr, loss, cons = inner.build(T)
        cons = list(cons)
        if lower is not None:
            cons.append(expr >= lower)
        if upper is not None:
            cons.append(expr <= upper)
        return expr, loss, cons

    comp.build = build
    return comp


def nonneg(inner):
    """Constrain a component to be nonnegative (shorthand for bounded(.., 0))."""
    return bounded(inner, lower=0.0)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "scripts")
    from decompose import make_problem, solve

    rng = np.random.default_rng(2)
    T = 500
    P = 50.0
    t = np.arange(T)
    true_trend = 1.0 + 0.003 * t
    true_seasonal = 0.6 * np.sin(2 * np.pi * t / P)
    true_sparse = np.zeros(T)
    spike_idx = rng.choice(T, size=6, replace=False)
    true_sparse[spike_idx] = rng.uniform(1.5, 2.5, size=6) * rng.choice([-1, 1], 6)
    noise = 0.05 * rng.standard_normal(T)
    y = true_trend + true_seasonal + true_sparse + noise
    y[150:180] = np.nan

    # NOTE: the sparse weight trades off against the residual. Too high and the
    # sparse component collapses to zero (spikes leak into the residual); this
    # is the sparse-vs-residual identifiability tension parameter search exists
    # to resolve. 1e-3 recovers the spikes with residual ~= the true noise.
    built = make_problem(
        y,
        components=[
            multiperiodic(periods=P, num_harmonics=6, weight=1e-2, role="seasonal"),
            smooth_trend(weight=1e2, order=2, role="trend"),
            sparse(weight=1e-3, role="spikes"),
        ],
        residual_loss="l2",
    )
    out = solve(built)

    trend_hat = out["values"]["trend"]
    seasonal_hat = out["values"]["seasonal"]
    spikes_hat = out["values"]["spikes"]

    trend_rmse = np.sqrt(np.mean((trend_hat - true_trend) ** 2))
    seas_rmse = np.sqrt(np.mean((seasonal_hat - true_seasonal) ** 2))
    recovered = np.sum(np.abs(spikes_hat[spike_idx]) > 0.5)

    print(f"status:           {out['status']}")
    print(f"trend RMSE:       {trend_rmse:.4f}")
    print(f"seasonal RMSE:    {seas_rmse:.4f}")
    print(f"spikes recovered: {recovered}/6")
    assert out["status"] in ("optimal", "optimal_inaccurate")
    assert trend_rmse < 0.1, trend_rmse
    assert seas_rmse < 0.1, seas_rmse
    assert recovered >= 5, recovered
    print("OK")
