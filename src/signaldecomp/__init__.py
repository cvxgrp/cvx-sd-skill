"""signaldecomp: convex signal decomposition for scalar time series.

Decompose a 1-D signal ``y`` into interpretable components -- a residual plus
structural terms (trend, periodic, sparse, exogenous, ...) -- by solving a
convex problem in CVXPY. Missing data is native: the consistency constraint is
imposed only on observed entries.

Primary entry points::

    from signaldecomp import make_problem, solve, Component
    from signaldecomp import multiperiodic, smooth_trend, sparse  # components
    from signaldecomp import huber_loss, quantile_loss            # data-fidelity

    built = make_problem(y, components=[multiperiodic(365.2425), smooth_trend(1e2)])
    out = solve(built)
    trend = out["values"]["trend"]
"""

from __future__ import annotations

from signaldecomp.components import (
    bounded,
    linear_trend,
    monotone_trend,
    multiperiodic,
    nonneg,
    pwl_trend,
    smooth_trend,
    sparse,
)
from signaldecomp.data_fidelity import (
    huber_loss,
    l1_loss,
    l2_loss,
    quantile_loss,
)
from signaldecomp.decompose import Component, make_problem, solve
from signaldecomp.periodic import period_samples

__all__ = [
    # core
    "Component",
    "make_problem",
    "solve",
    # components
    "multiperiodic",
    "linear_trend",
    "smooth_trend",
    "pwl_trend",
    "monotone_trend",
    "sparse",
    "bounded",
    "nonneg",
    # periodic helpers
    "period_samples",
    # data-fidelity losses
    "l2_loss",
    "l1_loss",
    "huber_loss",
    "quantile_loss",
]
