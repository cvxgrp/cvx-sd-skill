"""signaldecomp: convex signal decomposition for scalar time series.

Decompose a 1-D signal ``y`` into interpretable components -- a residual plus
structural terms (trend, periodic, sparse, exogenous, ...) -- by solving a
convex problem or a specified deterministic sequence of convex problems modeled
in CVXPY. Unavailable data is native: the consistency constraint is imposed on
rows where the signal is observed and every component input is valid.

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
    exog_interaction,
    exog_linear,
    exog_spline,
    grouped_sparse,
    grouped_trend,
    linear_trend,
    monotone_trend,
    multiperiodic,
    nonneg,
    pwc_trend,
    pwl_trend,
    smooth_trend,
    sparse,
)
from signaldecomp.basis_numerics import (
    BasisRankDiagnostics,
    BasisRankError,
    BasisWhitening,
    ComposedBasisWhitening,
    basis_rank_diagnostics,
    whiten_basis,
    whiten_basis_by_blocks,
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
from signaldecomp.exogenous import OffsetBasis, make_offset_basis, offset_source_mask
from signaldecomp.grouped import GroupBasis, make_group_basis
from signaldecomp.interactions import InteractionBasis, make_interaction_basis
from signaldecomp.periodic import (
    SECONDS_PER_DAY,
    SECONDS_PER_WEEK,
    SECONDS_PER_YEAR,
    period_samples,
)
from signaldecomp.reporting import (
    components_to_frame,
    format_report,
    plot_decomposition,
    plot_stability,
)
from signaldecomp.spline import SplineSupportDiagnostics, spline_support_diagnostics
from signaldecomp.transform import prepare_input, recover_components, recover_frame
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
    # basis numerics and design utilities
    "BasisRankDiagnostics",
    "BasisRankError",
    "BasisWhitening",
    "ComposedBasisWhitening",
    "basis_rank_diagnostics",
    "whiten_basis",
    "whiten_basis_by_blocks",
    "OffsetBasis",
    "make_offset_basis",
    "offset_source_mask",
    "GroupBasis",
    "make_group_basis",
    "InteractionBasis",
    "make_interaction_basis",
    "SplineSupportDiagnostics",
    "spline_support_diagnostics",
    # components
    "multiperiodic",
    "linear_trend",
    "smooth_trend",
    "pwl_trend",
    "pwc_trend",
    "monotone_trend",
    "sparse",
    "exog_linear",
    "exog_spline",
    "exog_interaction",
    "grouped_trend",
    "grouped_sparse",
    "bounded",
    "nonneg",
    # periodic helpers
    "period_samples",
    "SECONDS_PER_DAY",
    "SECONDS_PER_WEEK",
    "SECONDS_PER_YEAR",
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
    "format_report",
    "plot_decomposition",
    "plot_stability",
    # transforms (log / multiplicative pre-post-processing)
    "prepare_input",
    "recover_components",
    "recover_frame",
]
