"""Convex component catalog for signal decomposition.

Each factory returns a decompose.Component -- a build(T) callable yielding
(expr, loss, constraints) -- so components compose uniformly in
decompose.make_problem. All components here are convex (V1 scope).

A note on what is not here: the residual losses (mean-square-small, l1, Huber,
quantile) are properties of the residual x0 and are selected via
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

from signaldecomp.decompose import Component
from signaldecomp.periodic import multiperiodic  # noqa: F401  (re-exported)
from signaldecomp.spline import default_knots, make_spline_basis


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
        # Analysis + GLOBAL (l2^2 energy): an aggregate 'overall smoothness'
        # claim, NOT a per-entry one -- so it is NOT averaged by length.
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
        d = cp.diff(x, k=2)
        # Per-element normalization (see smooth_trend).
        loss = weight / d.shape[0] * cp.norm1(d)
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
        # Analysis + GLOBAL smoothness (see smooth_trend): not length-averaged.
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
        # Per-element normalization (see smooth_trend).
        loss = weight / x.shape[0] * cp.norm1(x)
        return x, loss, []

    comp.build = build
    return comp


def exog_linear(z, weight=0.0, role="exog"):
    """Linear response to an exogenous covariate: ``beta * z``.

    Models a signal contribution proportional to an external, time-aligned
    covariate ``z`` (e.g. temperature, irradiance). Unlike the time-based
    components, this is a function of covariate values, not of the time index.
    The covariate is captured at construction (it is a property of this
    component instance); ``build(T)`` checks that ``len(z) == T``.

    Parameters
    ----------
    z : numpy.ndarray, shape (T,)
        Exogenous covariate, aligned to the signal's time axis.
    weight : float
        Optional ridge penalty on the coefficient magnitude (0 = unpenalized).
    role : str
        Component role name.

    Returns
    -------
    Component
        With ``aux`` exposing the scalar coefficient under ``"<role>_beta"``.
    """
    z = np.asarray(z, dtype=float)
    comp = Component(role=role, build=None)

    def build(T):
        if z.shape != (T,):
            raise ValueError(
                f"exogenous covariate for role {role!r} has shape {z.shape}, "
                f"expected ({T},) to match the signal length."
            )
        beta = cp.Variable(name=f"{role}_beta")
        comp.aux[f"{role}_beta"] = beta
        expr = beta * z
        loss = weight * cp.square(beta)
        return expr, loss, []

    comp.build = build
    return comp


def exog_spline(z, n_knots=10, knots=None, weight=1e-2, role="exog"):
    """Nonlinear response to an exogenous covariate via a natural cubic spline.

    Models a smooth, possibly nonlinear dependence of the signal on an external,
    time-aligned covariate ``z`` as ``H(z) @ coef``, where ``H`` is a natural
    cubic spline basis (linear beyond the boundary knots). The constant column
    is dropped, so the covariate response carries no offset (that belongs to the
    trend intercept). The covariate is captured at construction; ``build(T)``
    checks that ``len(z) == T``.

    Parameters
    ----------
    z : numpy.ndarray, shape (T,)
        Exogenous covariate, aligned to the signal's time axis.
    n_knots : int
        Number of evenly spaced knots to place over the covariate range, used
        when ``knots`` is None. Must be >= 3.
    knots : numpy.ndarray, optional
        Explicit knot locations. If given, ``n_knots`` is ignored.
    weight : float
        Ridge penalty on the spline coefficients (controls smoothness).
    role : str
        Component role name.

    Returns
    -------
    Component
        With ``aux`` exposing the coefficient vector under ``"<role>_coef"``.
    """
    z = np.asarray(z, dtype=float)
    resolved_knots = (
        np.asarray(knots, dtype=float) if knots is not None else default_knots(z, n_knots)
    )
    comp = Component(role=role, build=None)

    def build(T):
        if z.shape != (T,):
            raise ValueError(
                f"exogenous covariate for role {role!r} has shape {z.shape}, "
                f"expected ({T},) to match the signal length."
            )
        H = make_spline_basis(z, resolved_knots, include_offset=False)
        coef = cp.Variable(H.shape[1], name=f"{role}_coef")
        comp.aux[f"{role}_coef"] = coef
        expr = H @ coef
        # SYNTHESIS: penalty is in coefficient space (fixed basis dim, no
        # signal-length dependence) -- never averaged by length.
        loss = weight * cp.sum_squares(coef)
        return expr, loss, []

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
