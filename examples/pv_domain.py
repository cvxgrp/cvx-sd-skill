"""PV DOMAIN LAYER -- NOT part of signaldecomp.

This module is the *domain* layer for the PV worked example: it converts a
solved decomposition's trend curve into a photovoltaic **degradation rate**
(percent per year). That percent-per-year conversion is domain knowledge, not
part of the domain-agnostic signaldecomp substrate -- so it lives here, in the
example, consuming only the PUBLIC ``out["values"]`` contract. Pure numpy; no
rdtools, no signaldecomp internals.

Boundary: signaldecomp gives you the trend *curve* (and everything else);
turning that curve into "the degradation rate" is the user's / domain's call.
Here we make one simple, transparent choice for all trend types.
"""

from __future__ import annotations

import numpy as np

SECONDS_PER_DAY = 86400.0
DAYS_PER_YEAR = 365.2425


def samples_per_year_from_delta(delta):
    """Samples per physical year for a grid spacing of ``delta`` seconds.

    Encodes the Delta-scaling invariant: physical time -> sample counts. For
    daily data (delta = 86400) this is 365.2425.

    Parameters
    ----------
    delta : float
        Grid spacing in seconds (from ``standardize_time_axis``).

    Returns
    -------
    float
        Number of samples in one year.
    """
    return DAYS_PER_YEAR * SECONDS_PER_DAY / float(delta)


def overall_degradation_rate(out, samples_per_year, log_space=False, role="trend"):
    """Overall degradation rate (%/yr) from a solved trend curve.

    One simple, transparent measure for ALL trend types: the normalized slope
    of the chord from the first trend point to the last, annualized. Negative
    means the signal is declining (degrading).

    - linear domain: ``100 * chord_slope_per_sample * samples_per_year /
      trend[0]`` (percent of the initial level, per year).
    - log domain (``log_space=True``, i.e. the decomposition was solved on
      ``log(y)``): the compound annual rate ``(exp(chord_slope * samples_per_year)
      - 1) * 100``, which is level-independent.

    Parameters
    ----------
    out : dict
        A solved output from ``signaldecomp.solve``.
    samples_per_year : float
        Samples in one year (see :func:`samples_per_year_from_delta`).
    log_space : bool
        Whether the decomposition was solved in log space.
    role : str
        Role name of the trend component (default ``"trend"``).

    Returns
    -------
    float
        Overall degradation rate in percent per year.
    """
    trend = np.asarray(out["values"][role], dtype=float)
    n = trend.shape[0]
    if n < 2:
        raise ValueError("trend must have at least two points.")
    chord_slope = (trend[-1] - trend[0]) / (n - 1)  # per sample
    if log_space:
        return float((np.exp(chord_slope * samples_per_year) - 1.0) * 100.0)
    return float(chord_slope * samples_per_year / trend[0] * 100.0)


def format_degradation_report(rate_pct_yr, meta):
    """Simplified single-rate PV degradation report (markdown).

    Parameters
    ----------
    rate_pct_yr : float
        Overall degradation rate from :func:`overall_degradation_rate`.
    meta : dict
        Model configuration to echo (e.g. ``trend_type``, ``loss``,
        ``num_harmonics``, ``lam_seasonal``, ``lam_trend``, ``log_transform``).

    Returns
    -------
    str
        Markdown-formatted report.
    """
    cfg = (
        f"**Trend:** `{meta.get('trend_type', '?')}` | "
        f"**Loss:** `{meta.get('loss', '?')}` | "
        f"**Harmonics:** {meta.get('num_harmonics', '?')} | "
        f"**log:** {meta.get('log_transform', False)}"
    )
    return (
        "## Degradation report (PV domain)\n\n"
        f"{cfg}\n\n"
        "| Quantity | Value |\n"
        "|---|---|\n"
        f"| Overall degradation rate | {rate_pct_yr:+.3f} %/yr |\n\n"
        "> Chord-slope rate: normalized slope of the trend from the first to "
        "the last sample, annualized. Negative = degrading.\n"
        ">\n"
        "> **DOMAIN LAYER:** this %/yr conversion is PV-specific and lives in "
        "the example, NOT in signaldecomp. The substrate gives you the trend "
        "curve; interpreting it as a rate is the domain's choice.\n"
    )
