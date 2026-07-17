"""Periodic (seasonal) components via truncated Fourier bases.

Periodic structure is represented with a truncated Fourier basis built by
``spcqe``. The key design choices, both grounded in the framework:

1. **Real-valued float periods in physical time.** A period is a real number
   (e.g. 365.2425 days), *not* an integer count of samples. It is converted to
   a sample count only via the sampling interval ``delta`` (seconds), and that
   conversion stays floating-point -- the basis is evaluated at the physical
   sample times, so the phase carries correctly across cycles and leap days
   need no special handling. See :func:`period_samples`.

2. **The DC (offset) column is dropped.** ``spcqe.make_basis_matrix`` returns a
   basis whose first column is the constant term. We slice it off from both the
   basis and its regularization matrix (``[:, 1:]``), so a seasonal component
   cannot represent a constant offset -- that degree of freedom belongs to the
   trend's intercept. This resolves the DC non-uniqueness between seasonal and
   trend *by construction*, with no extra constraint. (See
   ``plans/notes-offset-identifiability.md``.)

Multi-scale seasonality (e.g. daily + weekly + yearly) is expressed by passing
several periods at once; ``spcqe`` builds a joint basis.
"""

from __future__ import annotations

from collections.abc import Sequence

import cvxpy as cp
import numpy as np

import basis
from decompose import Component

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
        ``spcqe`` regularization matrix).
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


if __name__ == "__main__":
    # Smoke test: recover a two-scale (fast + slow) seasonal signal, no trend,
    # through a gap. Verifies the spcqe basis path, DC-drop, sparse-W matmul,
    # and multi-period support.
    import sys

    sys.path.insert(0, "scripts")
    from decompose import make_problem, solve

    rng = np.random.default_rng(1)
    T = 730  # two years of "daily" samples
    t = np.arange(T)
    P_year = 365.2425
    P_week = 7.0
    true = (
        1.0 * np.sin(2 * np.pi * t / P_year)
        + 0.3 * np.cos(2 * np.pi * t / P_week)
    )
    y = true + 0.05 * rng.standard_normal(T)
    y[100:140] = np.nan

    comp = multiperiodic(
        periods=[P_year, P_week], num_harmonics=6, weight=1e-2, role="periodic"
    )
    built = make_problem(y, components=[comp], residual_loss="l2")
    out = solve(built)

    seasonal_hat = out["values"]["periodic"]
    rmse = np.sqrt(np.mean((seasonal_hat - true) ** 2))
    dc = np.mean(seasonal_hat)
    theta = out["values"]["periodic_theta"]

    # NOTE: spcqe builds a TENSOR (outer-product) basis across periods, not
    # independent additive seasonals. For n periods each with H harmonics the
    # full basis has prod(2H+1) columns; after dropping the single DC column
    # that is prod(2H+1) - 1. For 2 periods, H=6: (2*6+1)**2 - 1 = 168.
    expected_coefs = (2 * 6 + 1) ** 2 - 1  # == 168

    print(f"status:            {out['status']}")
    print(f"seasonal RMSE:     {rmse:.4f}")
    print(f"seasonal DC mean:  {dc:.4e}  (small, not forced to 0: see note)")
    print(f"n coefficients:    {theta.shape[0]}  (expected {expected_coefs})")
    print(f"period_samples yr: {period_samples(SECONDS_PER_YEAR, SECONDS_PER_DAY):.4f}")
    assert out["status"] in ("optimal", "optimal_inaccurate")
    assert rmse < 0.05, rmse
    # The DC *column* is dropped by construction (theta has no constant term),
    # so there is no offset degree of freedom -- this resolves the
    # seasonal/trend non-uniqueness. The empirical sample mean is small but not
    # machine-zero: a finite sum of harmonics over a non-integer number of
    # periods need not average to exactly zero. Assert the structural fact.
    assert theta.shape[0] == expected_coefs, theta.shape[0]
    assert abs(dc) < 1e-2, dc
    print("OK")
