"""Natural cubic spline basis for exogenous (covariate) components.

The basis-construction function :func:`make_spline_basis` is adapted from the
TSGAM estimator by the Alliance for Sustainable Energy, LLC and Nimish Telang
(BSD-3-Clause). It builds a natural cubic spline basis: a smooth, flexible
response that is linear beyond the boundary knots (the "natural" constraint),
suitable for modeling a nonlinear dependence of the signal on an exogenous
covariate as a convex ``H @ coef`` term.

Only the pure-numpy basis construction is lifted here; the estimator, lead/lag
expansion, and interaction machinery from the original are out of scope.
"""

from __future__ import annotations

import numpy as np


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
    n_knots = len(knots)
    if n_knots < 3:
        raise ValueError(f"need at least 3 knots for a cubic spline; got {n_knots}.")
    x = np.asarray(x, dtype=float)
    H = np.ones((len(x), n_knots), dtype=float)
    H[:, 1] = x
    for _i in range(n_knots - 2):
        _j = _i + 2
        H[:, _j] = d_func(x, knots[_i], knots[-1]) - d_func(x, knots[-2], knots[-1])
    return H if include_offset else H[:, 1:]


def default_knots(x, n_knots):
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
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        raise ValueError("x has no finite values to place knots over.")
    return np.linspace(np.min(finite), np.max(finite), n_knots)
