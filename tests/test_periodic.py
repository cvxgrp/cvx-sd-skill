"""Tests for the multiperiodic (truncated-Fourier) component.

Ports the periodic.py smoke test (recover a two-scale periodic signal through a
gap) and locks in the DC-column-dropped invariant and the physical-time period
scaling.
"""

import numpy as np

from signaldecomp import make_problem, multiperiodic, solve
from signaldecomp.periodic import (
    SECONDS_PER_DAY,
    SECONDS_PER_YEAR,
    period_samples,
)

_OPTIMAL = ("optimal", "optimal_inaccurate")


def test_recovers_two_scale_periodic_through_gap():
    rng = np.random.default_rng(1)
    T = 730
    t = np.arange(T)
    P_year, P_week = 365.2425, 7.0
    true = 1.0 * np.sin(2 * np.pi * t / P_year) + 0.3 * np.cos(2 * np.pi * t / P_week)
    y = true + 0.05 * rng.standard_normal(T)
    y[100:140] = np.nan
    comp = multiperiodic(periods=[P_year, P_week], num_harmonics=6, weight=1e-2)
    out = solve(make_problem(y, components=[comp]))
    assert out["status"] in _OPTIMAL
    rmse = np.sqrt(np.mean((out["values"]["periodic"] - true) ** 2))
    assert rmse < 0.05


def test_dc_column_dropped_coefficient_count():
    # Two periods, H=6: full basis has (2H+1)**2 columns; dropping the single
    # DC column leaves (2H+1)**2 - 1 == 168 coefficients. No constant term.
    rng = np.random.default_rng(1)
    T = 730
    y = rng.standard_normal(T)
    comp = multiperiodic(periods=[365.2425, 7.0], num_harmonics=6, weight=1e-2)
    out = solve(make_problem(y, components=[comp]))
    theta = out["values"]["periodic_theta"]
    assert theta.shape[0] == (2 * 6 + 1) ** 2 - 1  # == 168


def test_default_role_is_periodic():
    comp = multiperiodic(periods=50.0)
    assert comp.role == "periodic"


def test_period_samples_scales_by_delta():
    # A yearly physical period, sampled daily, is 365.2425 samples.
    assert period_samples(SECONDS_PER_YEAR, SECONDS_PER_DAY) == 365.2425


def test_period_samples_rejects_nonpositive_delta():
    import pytest

    with pytest.raises(ValueError, match="positive"):
        period_samples(SECONDS_PER_YEAR, 0.0)
