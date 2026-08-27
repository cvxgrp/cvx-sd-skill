# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
"""Natural cubic spline basis for exogenous (covariate) components.

The basis-construction function :func:`make_spline_basis` is adapted from the
TSGAM estimator by the Alliance for Sustainable Energy, LLC and Nimish Telang
(BSD-3-Clause). It builds a natural cubic spline basis: a smooth, flexible
response that is linear beyond the boundary knots (the "natural" constraint),
suitable for modeling a nonlinear dependence of the signal on an exogenous
covariate as a convex ``H @ coef`` term.

The basis is shared by the importable exogenous component builders. Numerical
rank and whitening live in :mod:`signaldecomp.basis_numerics`; offset expansion
lives in :mod:`signaldecomp.exogenous`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from signaldecomp.basis_numerics import (
    BasisRankDiagnostics,
    basis_rank_diagnostics,
)


@dataclass(frozen=True)
class SplineSupportDiagnostics:
    """Training-support and singular-geometry diagnostics for a spline basis."""

    knots: np.ndarray
    n_fit: int
    interval_counts: np.ndarray
    below_boundary_count: int
    above_boundary_count: int
    basis: BasisRankDiagnostics


def make_spline_basis(x, knots, include_offset=False):
    """Build a natural cubic spline basis matrix evaluated at ``x``.

    Parameters
    ----------
    x : numpy.ndarray, shape (n,)
        Covariate values at which to evaluate the basis.
    knots : numpy.ndarray, shape (n_knots,)
        Knot locations, sorted ascending, spanning the covariate range.
    include_offset : bool
        If True, keep the leading constant column. For signal decomposition the
        constant belongs to the trend intercept, so this is normally False (the
        constant column is dropped, mirroring the DC-drop in the periodic basis).

    Returns
    -------
    numpy.ndarray
        The spline basis of shape (n, n_knots) if include_offset else
        (n, n_knots - 1). Column 0 is the constant, column 1 the linear term,
        and the rest the natural cubic terms; the constant is dropped unless
        include_offset is True.

    Notes
    -----
    Adapted from the TSGAM estimator (Alliance for Sustainable Energy, LLC and
    Nimish Telang; BSD-3-Clause).
    """

    def d_func(xx, k, k_max):
        n1 = np.clip(np.power(xx - k, 3), 0, np.inf)
        n2 = np.clip(np.power(xx - k_max, 3), 0, np.inf)
        return (n1 - n2) / (k_max - k)

    knots = np.asarray(knots, dtype=float)
    if knots.ndim != 1:
        raise ValueError(f"knots must be 1-D; got shape {knots.shape}.")
    n_knots = len(knots)
    if n_knots < 3:
        raise ValueError(f"need at least 3 knots for a cubic spline; got {n_knots}.")
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D; got shape {x.shape}.")
    if not np.all(np.isfinite(knots)):
        raise ValueError("knots must contain only finite values.")
    if np.any(np.diff(knots) <= 0):
        raise ValueError("knots must be strictly increasing.")
    H = np.ones((len(x), n_knots), dtype=float)
    H[:, 1] = x
    for _i in range(n_knots - 2):
        _j = _i + 2
        H[:, _j] = d_func(x, knots[_i], knots[-1]) - d_func(x, knots[-2], knots[-1])
    return H if include_offset else H[:, 1:]


def default_knots(x, n_knots, fit_mask=None):
    """Evenly spaced knots spanning the finite range of ``x``.

    Parameters
    ----------
    x : numpy.ndarray
        Covariate values.
    n_knots : int
        Number of knots (>= 3).

    Returns
    -------
    numpy.ndarray
        Knot locations from min(x) to max(x), shape (n_knots,).
    """
    if n_knots < 3:
        raise ValueError(f"n_knots must be >= 3; got {n_knots}.")
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D; got shape {x.shape}.")
    if fit_mask is None:
        mask = np.ones(x.shape[0], dtype=bool)
    else:
        mask = np.asarray(fit_mask)
        if mask.dtype != np.bool_:
            raise TypeError("fit_mask must have boolean dtype.")
        if mask.shape != x.shape:
            raise ValueError(
                f"fit_mask has shape {mask.shape}, expected {x.shape}."
            )
    finite = x[mask & np.isfinite(x)]
    if finite.size == 0:
        raise ValueError("x has no finite values to place knots over.")
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if lower == upper:
        raise ValueError("x has no finite range over which to place knots.")
    return np.linspace(lower, upper, n_knots)


def spline_support_diagnostics(
    x,
    knots,
    fit_mask,
    *,
    rank_tolerance=None,
):
    """Audit training support and numerical geometry for explicit spline knots.

    The caller supplies the exact training mask. This helper never chooses a
    boundary rule, clips observations, or changes the requested basis.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D; got shape {x.shape}.")
    mask = np.asarray(fit_mask)
    if mask.dtype != np.bool_:
        raise TypeError("fit_mask must have boolean dtype.")
    if mask.shape != x.shape:
        raise ValueError(f"fit_mask has shape {mask.shape}, expected {x.shape}.")
    if not mask.any():
        raise ValueError("fit_mask selects no rows.")
    if not np.all(np.isfinite(x[mask])):
        raise ValueError("x must be finite on every row selected by fit_mask.")

    knots = np.asarray(knots, dtype=float)
    fitted_x = x[mask]
    fitted_basis = make_spline_basis(fitted_x, knots, include_offset=False)
    fitted_basis_mask = np.ones(fitted_x.shape[0], dtype=bool)
    basis_diagnostics = basis_rank_diagnostics(
        fitted_basis,
        fitted_basis_mask,
        rank_tolerance=rank_tolerance,
    )
    interval_counts, _ = np.histogram(fitted_x, bins=knots)
    return SplineSupportDiagnostics(
        knots=knots.copy(),
        n_fit=int(mask.sum()),
        interval_counts=interval_counts,
        below_boundary_count=int(np.count_nonzero(fitted_x < knots[0])),
        above_boundary_count=int(np.count_nonzero(fitted_x > knots[-1])),
        basis=basis_diagnostics,
    )
