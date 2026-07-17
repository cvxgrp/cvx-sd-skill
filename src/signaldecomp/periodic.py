"""Periodic components via truncated Fourier bases.

Periodic structure is represented with a truncated Fourier basis built by the
vendored :mod:`signaldecomp.basis` module. The key design choices, both
grounded in the framework:

1. **Real-valued float periods in physical time.** A period is a real number
   (e.g. 365.2425 days), *not* an integer count of samples. It is converted to
   a sample count only via the sampling interval ``delta`` (seconds), and that
   conversion stays floating-point -- the basis is evaluated at the physical
   sample times, so the phase carries correctly across cycles and leap days
   need no special handling. See :func:`period_samples`.

2. **The DC (offset) column is dropped.** :func:`signaldecomp.basis.make_basis_matrix`
   returns a basis whose first column is the constant term. We slice it off from
   both the basis and its regularization matrix (``[:, 1:]``), so a periodic
   component cannot represent a constant offset -- that degree of freedom
   belongs to the trend's intercept. This resolves the DC non-uniqueness
   between periodic and trend *by construction*, with no extra constraint.
   (See ``plans/notes-offset-identifiability.md``.)

Multi-scale periodicity (e.g. daily + weekly + yearly) is expressed by passing
several periods at once; the basis module builds a joint (wider) basis with
cross-terms.
"""

from __future__ import annotations

from collections.abc import Sequence

import cvxpy as cp
import numpy as np

from signaldecomp import basis
from signaldecomp.decompose import Component

# Physical period constants, in seconds. The yearly period uses the mean
# Gregorian year (365.2425 days) so that leap years require no special casing.
SECONDS_PER_DAY = 86_400.0
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY
SECONDS_PER_YEAR = 365.2425 * SECONDS_PER_DAY


def period_samples(physical_period_seconds: float, delta_seconds: float) -> float:
    """Convert a physical period to a (float) number of samples.

    Parameters
    ----------
    physical_period_seconds : float
        Period in seconds (e.g. ``SECONDS_PER_YEAR``).
    delta_seconds : float
        Sampling interval in seconds (the canonical internal unit for delta).

    Returns
    -------
    float
        Period in samples. Deliberately *not* rounded: the Fourier basis is
        valid for non-integer periods, and rounding here would reintroduce the
        leap-year / phase-drift error the float representation avoids.
    """
    if delta_seconds <= 0:
        raise ValueError("delta_seconds must be positive.")
    return physical_period_seconds / delta_seconds


def multiperiodic(
    periods: float | Sequence[float],
    num_harmonics: int = 6,
    weight: float = 1e-1,
    role: str = "periodic",
) -> Component:
    """Truncated-Fourier component over one or more periods (DC column removed).

    A generic periodic component: a truncated Fourier basis over the given
    period(s). "Seasonal" structure is one common use, but daily, weekly, or
    any other cyclic pattern is expressed the same way -- the component makes no
    domain assumption.


    Parameters
    ----------
    periods : float or sequence of float
        Period length(s) in **samples** (use :func:`period_samples` to convert
        from physical time). Pass several for multi-scale seasonality.
    num_harmonics : int
        Number of harmonic pairs per period.
    weight : float
        Regularization weight on the Fourier coefficients (baked into the
        regularization matrix from :mod:`signaldecomp.basis`).
    role : str
        Semantic component name.

    Returns
    -------
    Component
        With ``aux`` exposing the coefficient vector under ``"<role>_theta"``.
    """
    period_list = [float(periods)] if np.isscalar(periods) else [float(p) for p in periods]
    comp = Component(role=role, build=None)

    def build(T: int):
        # Drop the DC column (index 0) from both basis and regularizer so the
        # seasonal component cannot absorb a constant offset.
        basis_mat = basis.make_basis_matrix(num_harmonics, T, period_list)[:, 1:]
        reg = basis.make_regularization_matrix(
            num_harmonics, weight, period_list
        ).tocsr()[:, 1:]
        theta = cp.Variable(basis_mat.shape[1], name=f"{role}_theta")
        # Expose the coefficient vector; make_problem reads comp.aux after
        # calling build(), so populating it here is sufficient.
        comp.aux[f"{role}_theta"] = theta
        expr = basis_mat @ theta
        loss = cp.sum_squares(reg @ theta)
        return expr, loss, []

    comp.build = build
    return comp
