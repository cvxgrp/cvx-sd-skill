"""Data-fidelity loss factories for the residual component.

In the signal-decomposition framework every component k has a loss phi_k --
including components whose loss is a pure convex indicator function (0 on the
feasible set, +infinity off it), which is how a "constraint" is expressed on the
theory side. This module is specifically about the *data-fidelity* loss: phi_1,
the loss on the residual x0, which measures how well the components together
reproduce the observed data. (Component losses live with their components in
``components.py``.)

A data-fidelity loss is any DCP-compliant convex function of the residual
variable: a callable ``loss_fn(x) -> scalar cvxpy expression``.
``make_problem`` accepts any such callable, so users are not limited to a fixed
menu -- write your own, compose them, or start from one of the presets below and
iterate.

The presets here are convenience factories returning the common data-fidelity
losses from the framework. They double as worked patterns for defining new ones.
Each is normalized by the series length so weights are comparable across signals
of different length.

Example of a custom loss (the pattern these presets follow)::

    def my_loss(x):
        return (1.0 / x.shape[0]) * cp.sum(cp.huber(x, 0.5)) + 1e-3 * cp.norm1(x)

    make_problem(y, components, residual_loss=my_loss)
"""

from __future__ import annotations

from collections.abc import Callable

import cvxpy as cp


def l2_loss() -> Callable:
    """Mean-square-small residual (the framework default).

    Returns
    -------
    callable
        ``x -> (1/T) * ||x||_2^2``.
    """

    def loss(x):
        return (1.0 / x.shape[0]) * cp.sum_squares(x)

    return loss


def l1_loss() -> Callable:
    """Sum-absolute (robust) residual; less sensitive to outliers than l2."""

    def loss(x):
        return (1.0 / x.shape[0]) * cp.norm1(x)

    return loss


def huber_loss(M: float = 1.0) -> Callable:
    """Huber (robust) residual: quadratic within +/-M, linear beyond.

    Parameters
    ----------
    M : float
        Threshold between the quadratic and linear regimes.
    """

    def loss(x):
        return (1.0 / x.shape[0]) * cp.sum(cp.huber(x, M))

    return loss


def quantile_loss(q: float = 0.5) -> Callable:
    """Pinball / quantile (asymmetric) residual at quantile level ``q``.

    ``q=0.5`` reduces to (scaled) sum-absolute. Values above/below zero are
    penalized asymmetrically, useful when over- and under-estimation differ in
    cost.

    Parameters
    ----------
    q : float
        Quantile level in (0, 1).
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1); got {q}")

    def loss(x):
        n = x.shape[0]
        return (2.0 / n) * (q * cp.sum(cp.pos(x)) + (1 - q) * cp.sum(cp.pos(-x)))

    return loss
