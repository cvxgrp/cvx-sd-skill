"""Standardize a raw time series onto a regular grid for decomposition.

The decomposition core assumes ``y`` is a 1-D vector on a **regular grid**, with
a scalar sampling interval ``delta`` (seconds) tying index space to physical
time, and missing entries marked NaN (which become the mask). Real data rarely
arrives that way: timestamps may be irregular, unsorted, duplicated, or contain
gaps from skipped acquisitions. This module bridges raw input to that canonical
form.

Main entry point: :func:`standardize_time_axis`, which rebuilds the index at
regular intervals starting at **midnight of the first day**, snapping existing
samples to the nearest grid point and leaving genuine gaps as NaN. It emits
``(y, delta, index)`` plus scan-rate diagnostics.

The sampling interval is the **modal** (most common) inter-sample gap, found by
clustering gaps by *relative* closeness (so sub-rate jitter does not fragment a
single rate) and measured with ``Timedelta.total_seconds()``. Using
``.total_seconds()`` -- not ``.seconds``, which wraps at 24h -- keeps multi-day
and sub-second gaps correct.

Adapted from a PV data-handling routine; the solar-noon / timezone-correction
heuristics (which assumed a power signal) have been removed -- this core is
domain-agnostic.

WARNING: provide timestamps in **local standard time with no DST shifts**.
This module deliberately does not handle daylight-saving transitions: a regular
fixed-second grid cannot represent the 23h/25h DST days, and "D" is treated as
exactly 86400s. DST-shifted timestamps will produce spurious gaps/duplicates at
the transitions. Convert to local standard time (or UTC) before standardizing.
Data coarser than daily (weekly, monthly, ...) is out of scope -- prepare it
yourself with a calendar-aware cadence.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


def _gap_seconds(index):
    """Positive inter-sample gaps of a (sorted) index, in seconds (unrounded).

    Uses ``Timedelta.total_seconds()`` so multi-day gaps do not wrap.
    """
    ordered = index.sort_values()
    deltas = ordered[1:] - ordered[:-1]
    secs = deltas.total_seconds().to_numpy(dtype=float)
    secs = secs[secs > 0]
    if secs.size == 0:
        raise ValueError("all timestamps are identical; cannot derive spacing.")
    return secs


def _cluster_gaps(secs, rel_tol):
    """Cluster gap sizes by *relative* closeness, weighted by frequency.

    Sampling jitter scales with the sampling rate, so "same rate" is a
    proportional notion, not an absolute one: gaps within ``rel_tol`` of a
    cluster's running mean join it. Genuinely distinct rates (e.g. 60s vs 300s)
    form separate clusters. Clusters are returned most-frequent first.

    Parameters
    ----------
    secs : numpy.ndarray
        Positive gap sizes in seconds (unrounded).
    rel_tol : float
        Relative tolerance; a gap joins a cluster if it is within
        ``rel_tol`` (fractional) of that cluster's representative value.

    Returns
    -------
    list of tuple
        ``(representative_seconds, count)`` per cluster, sorted by count desc.
        The representative is the frequency-weighted median of cluster members.
    """
    order = np.argsort(secs)
    clusters = []  # each: list of member gap values
    for g in secs[order]:
        placed = False
        for members in clusters:
            rep = np.median(members)
            if abs(g - rep) <= rel_tol * rep:
                members.append(g)
                placed = True
                break
        if not placed:
            clusters.append([g])
    summary = [(float(np.median(m)), len(m)) for m in clusters]
    summary.sort(key=lambda t: t[1], reverse=True)
    return summary


# Standard fixed-duration sampling rates, in seconds, that the empirical delta
# may be snapped to. Capped at daily (86400): coarser cadences (weekly,
# monthly, ...) are not fixed numbers of seconds and are out of scope for this
# helper -- see :func:`nearest_standard_freq`. "D" is treated as exactly 86400s
# (local standard time, no DST -- see the module WARNING).
_STANDARD_FREQS = {
    "s": 1,
    "5s": 5,
    "15s": 15,
    "30s": 30,
    "min": 60,
    "5min": 300,
    "10min": 600,
    "15min": 900,
    "30min": 1800,
    "h": 3600,
    "2h": 7200,
    "3h": 10800,
    "6h": 21600,
    "12h": 43200,
    "D": 86400,
}
_MAX_STANDARD_SECONDS = max(_STANDARD_FREQS.values())  # daily


def nearest_standard_freq(delta_seconds, snap_tol=0.01):
    """Snap an empirical spacing to the nearest standard pandas frequency.

    Finds the standard fixed-duration frequency (see ``_STANDARD_FREQS``, up to
    daily) whose canonical length is closest, in *relative* terms, to
    ``delta_seconds``. If the closest is within ``snap_tol`` (fractional), its
    clean name and canonical seconds are returned; otherwise the empirical value
    is kept and the frequency name is None.

    This gives jittered/rounded timestamps a clean rate (e.g. an empirical 3599s
    becomes ``("h", 3600.0)``) without relocating genuinely non-standard data
    (which stays at its empirical spacing).

    Parameters
    ----------
    delta_seconds : float
        The empirically derived sampling interval, in seconds.
    snap_tol : float
        Maximum relative distance to accept a snap (default 1%).

    Returns
    -------
    tuple
        ``(freq_str_or_None, seconds)``. On a successful snap, the canonical
        pandas freq string and its exact seconds; otherwise ``(None,
        delta_seconds)``.

    Raises
    ------
    ValueError
        If ``delta_seconds`` is coarser than daily beyond ``snap_tol``. At that
        scale seconds is the wrong unit and a fixed-seconds grid is
        inappropriate; prepare coarser (e.g. monthly) data yourself with a
        calendar-aware cadence.
    """
    if delta_seconds <= 0:
        raise ValueError(f"delta_seconds must be positive; got {delta_seconds}")
    if delta_seconds > _MAX_STANDARD_SECONDS * (1 + snap_tol):
        raise ValueError(
            f"empirical spacing {delta_seconds:.0f}s is coarser than daily; this "
            f"helper targets fixed-rate sub-daily-to-daily data. Prepare coarser "
            f"(e.g. monthly) data yourself with a calendar-aware cadence."
        )
    best_name, best_secs, best_rel = None, delta_seconds, np.inf
    for name, secs in _STANDARD_FREQS.items():
        rel = abs(delta_seconds - secs) / secs
        if rel < best_rel:
            best_name, best_secs, best_rel = name, secs, rel
    if best_rel <= snap_tol:
        return best_name, float(best_secs)
    return None, float(delta_seconds)


def derive_delta(index, rel_tol=0.02):
    """Derive the sampling interval (seconds) as the modal inter-sample gap.

    Gaps are clustered by *relative* closeness (see :func:`_cluster_gaps`) so
    sub-rate jitter collapses to one rate; the interval is the representative
    of the most frequent cluster.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Timestamps (need not be regular or sorted); at least two entries.
    rel_tol : float
        Relative tolerance for treating gaps as the same rate (default 2%).

    Returns
    -------
    float
        The modal gap in seconds.
    """
    if len(index) < 2:
        raise ValueError("need at least two timestamps to derive delta.")
    clusters = _cluster_gaps(_gap_seconds(index), rel_tol)
    return float(clusters[0][0])


def scan_rates(index, rel_tol=0.02, min_fraction=0.05):
    """Detect and describe the inter-sample sampling rate(s).

    A record that switches sampling rate partway (e.g. 5-minute data that
    becomes 1-minute) looks regular to a naive delta but is not. Gaps are
    clustered by *relative* closeness (see :func:`_cluster_gaps`) so jitter does
    not fragment a single rate; each cluster holding at least ``min_fraction``
    of the intervals is reported as a distinct rate. When several are present,
    the dates between which the daily-median rate transitions are also returned,
    so the caller can decide whether to split the record.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Timestamps.
    rel_tol : float
        Relative tolerance for treating gaps as the same rate (default 2%).
    min_fraction : float
        A cluster must hold at least this fraction of all intervals to count as
        a distinct rate.

    Returns
    -------
    dict
        - ``"modal_seconds"`` : the representative gap of the largest cluster
          (the chosen delta).
        - ``"rates_seconds"`` : list of distinct dominant rates (seconds),
          most frequent first.
        - ``"multiple_rates"`` : True if more than one dominant rate is present.
        - ``"transitions"`` : list of ``(last_date_before, first_date_after)``
          pairs where the daily-median scan rate changes (empty unless
          multiple rates).
    """
    if len(index) < 2:
        raise ValueError("need at least two timestamps to report scan rates.")
    ordered = index.sort_values()
    secs = _gap_seconds(ordered)
    n = secs.size
    clusters = _cluster_gaps(secs, rel_tol)
    modal = float(clusters[0][0])
    rates = [rep for rep, count in clusters if count > min_fraction * n]

    transitions = []
    if len(rates) > 1:
        # Assign each gap to its nearest dominant rate, take the daily-median
        # assigned rate, then find the day-boundaries where it changes.
        rate_arr = np.array(rates, dtype=float)
        assigned = rate_arr[np.argmin(np.abs(secs[:, None] - rate_arr[None, :]), axis=1)]
        gap_series = pd.Series(np.r_[assigned, [np.nan]], index=ordered)
        daily = gap_series.groupby(gap_series.index.date).median().dropna()
        if len(daily) >= 2:
            changed = np.diff(daily.to_numpy()) != 0
            leading = daily.index[np.r_[changed, [False]]]
            trailing = daily.index[np.r_[[False], changed]]
            transitions = list(zip(leading, trailing))

    return {
        "modal_seconds": modal,
        "rates_seconds": rates,
        "multiple_rates": bool(len(rates) > 1),
        "transitions": transitions,
    }


def standardize_time_axis(
    data,
    value_column=None,
    datetime_column=None,
    freq=None,
    rel_tol=0.02,
    snap_tol=0.01,
):
    """Coerce raw time-indexed data onto a regular grid for decomposition.

    Rebuilds the index at regular ``delta``-second intervals from midnight of
    the first day through the last day, snaps each existing sample to its
    nearest grid point (at most one per point), and leaves unfilled grid points
    as NaN -- exactly the missing-data representation the decomposition mask
    consumes.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Time-indexed input. Either a DatetimeIndex, or a datetime column named
        by ``datetime_column`` (if None, a column containing "time"/"Time" is
        used when the index is not already datetimes).
    value_column : str, optional
        For a DataFrame, the column holding the signal. Required for a
        multi-column DataFrame; a single-column DataFrame or a Series is
        unambiguous.
    datetime_column : str, optional
        Column to parse as the datetime index (for data whose index is not
        already datetimes).
    freq : str, optional
        A standard pandas frequency string (e.g. ``"h"``, ``"15min"``, ``"D"``)
        asserting the nominal sampling rate. If given it is authoritative: the
        grid uses it and ``delta`` is its canonical seconds. If None, ``delta``
        is derived empirically (modal gap) and snapped to the nearest standard
        rate within ``snap_tol``.
    rel_tol : float
        Relative tolerance for clustering gaps into rates (default 2%).
    snap_tol : float
        Relative tolerance for snapping the empirical ``delta`` to a standard
        frequency (default 1%); see :func:`nearest_standard_freq`.

    Returns
    -------
    dict
        - ``"y"`` : numpy.ndarray, the signal on the regular grid (NaN in gaps).
        - ``"delta"`` : float, grid spacing in seconds.
        - ``"index"`` : pandas.DatetimeIndex, the regular grid.
        - ``"freq"`` : the standard pandas freq string (asserted or snapped), or
          None if the empirical rate matched no standard frequency.
        - ``"scan_rates"`` : dict from :func:`scan_rates` (diagnostics).
        - ``"series"`` : pandas.Series of ``y`` on ``index`` (convenience).
    """
    if isinstance(data, pd.Series):
        df = data.to_frame(name=value_column or (data.name if data.name is not None else "value"))
        value_column = df.columns[0]
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("data must be a pandas Series or DataFrame.")

    # Establish a DatetimeIndex.
    if not isinstance(df.index, pd.DatetimeIndex):
        key = datetime_column
        if key is None:
            candidates = [c for c in df.columns if "time" in c.lower()]
            if not candidates:
                raise ValueError(
                    "no DatetimeIndex and no datetime column found; pass "
                    "datetime_column."
                )
            key = candidates[0]
        df[key] = pd.to_datetime(df[key])
        df = df.set_index(key)
    df.index = pd.to_datetime(df.index)

    # Choose the value column.
    if value_column is None:
        non_dt = [c for c in df.columns]
        if len(non_dt) != 1:
            raise ValueError(
                "value_column is required for a multi-column DataFrame; "
                f"columns are {list(df.columns)}."
            )
        value_column = non_dt[0]
    elif value_column not in df.columns:
        raise ValueError(f"value_column {value_column!r} not in {list(df.columns)}.")

    # Drop tz so grid construction/reindex is unambiguous (domain-agnostic:
    # we standardize the axis, we do not attempt tz correction).
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Clean the index before deriving spacing.
    df = df.loc[df.index.notnull()]
    df = df.loc[~df.index.duplicated()]
    df = df.sort_index()
    if len(df) < 2:
        raise ValueError("need at least two valid timestamps to standardize.")

    rates = scan_rates(df.index, rel_tol=rel_tol)
    if freq is not None:
        # User-asserted nominal rate is authoritative. Prefer the fixed-seconds
        # table (it defines "D"=86400, which pd.Timedelta refuses since <Day> is
        # a calendar offset); fall back to pd.Timedelta for other fixed offsets.
        if freq in _STANDARD_FREQS:
            delta = float(_STANDARD_FREQS[freq])
        else:
            try:
                delta = float(
                    pd.Timedelta(pd.tseries.frequencies.to_offset(freq)).total_seconds()
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"freq={freq!r} is not a fixed-duration rate this helper can "
                    f"use (calendar offsets like weekly/monthly are out of scope). "
                    f"Use a sub-daily-to-daily fixed rate."
                ) from exc
        grid_freq = freq
    else:
        # Empirical modal gap, snapped to the nearest standard rate (or kept).
        grid_freq, delta = nearest_standard_freq(rates["modal_seconds"], snap_tol)
    if delta <= 0:
        raise ValueError(f"derived non-positive delta ({delta}); check timestamps.")

    # Regular grid: midnight of first day through end of last day. Build on the
    # standard freq string when we have one (exact, no drift), else on seconds.
    start = df.index[0]
    end = df.index[-1]
    grid = pd.date_range(
        start=start.normalize(),
        end=(end.normalize() + timedelta(days=1)),
        freq=grid_freq if grid_freq is not None else f"{int(delta)}s",
    )[:-1]

    # Snap existing samples to nearest grid point (at most one each); gaps NaN.
    snapped = df[[value_column]].reindex(index=grid, method="nearest", limit=1)
    series = snapped[value_column]
    series.name = value_column

    return {
        "y": series.to_numpy(dtype=float),
        "delta": delta,
        "index": grid,
        "freq": grid_freq,
        "scan_rates": rates,
        "series": series,
    }
