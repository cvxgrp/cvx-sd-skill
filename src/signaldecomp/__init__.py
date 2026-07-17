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
    exog_linear,
    exog_spline,
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
from signaldecomp.heatmap import (
    fold_from_standardized,
    fold_to_2d,
    plot_heatmap,
    steps_per_day,
)
from signaldecomp.periodic import period_samples
from signaldecomp.reporting import (
    components_to_frame,
    plot_decomposition,
    plot_stability,
)
from signaldecomp.time_axis import (
    derive_delta,
    nearest_standard_freq,
    scan_rates,
    standardize_time_axis,
)
from signaldecomp.validation import (
    bootstrap_ci,
    expanding_window_stability,
    holdout_select,
    valid_endpoints,
)

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
    "exog_linear",
    "exog_spline",
    "bounded",
    "nonneg",
    # periodic helpers
    "period_samples",
    # time-axis standardization
    "standardize_time_axis",
    "derive_delta",
    "scan_rates",
    "nearest_standard_freq",
    # heat-map diagnostic
    "fold_to_2d",
    "fold_from_standardized",
    "steps_per_day",
    "plot_heatmap",
    # data-fidelity losses
    "l2_loss",
    "l1_loss",
    "huber_loss",
    "quantile_loss",
    # validation
    "bootstrap_ci",
    "expanding_window_stability",
    "holdout_select",
    "valid_endpoints",
    # reporting (pandas round-trip + plots)
    "components_to_frame",
    "plot_decomposition",
    "plot_stability",
]
