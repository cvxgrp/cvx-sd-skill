"""Sub-daily heat-map diagnostic: fold a 1-D signal into a day x time-of-day matrix.

For sub-daily data, folding the vector so each column is one day and each row is
a time-of-day reveals daily structure, seasonal drift, and gaps at a glance. It
works on the raw signal ``y`` *or* on any solved component/residual (pass its
value array), so the same view can compare input and decomposition.

The fold is a column-major reshape into ``(n_steps, n_days)`` where
``n_steps = round(86400 / delta)`` is the number of samples per day. This
requires the signal to be on a **standardized, midnight-anchored regular grid**
spanning whole days (see :mod:`signaldecomp.time_axis`); the fold validates that
its length is a whole number of days.

Adapted from a PV data-handling routine; the solar-power labels and clear-day
markers have been removed -- this core is domain-agnostic.
"""

from __future__ import annotations

import numpy as np

_SECONDS_PER_DAY = 86400


def _trim_empty_days(D):
    """Drop leading/trailing all-NaN columns (days) from a folded matrix.

    Returns the trimmed matrix and the ``(start, stop)`` column indices kept, so
    a caller can trim a parallel day axis identically.
    """
    empty = np.all(np.isnan(D), axis=0)
    n = empty.size
    start = 0
    while start < n and empty[start]:
        start += 1
    stop = n
    while stop > start and empty[stop - 1]:
        stop -= 1
    return D[:, start:stop], (start, stop)


def steps_per_day(delta):
    """Number of samples per day for a sampling interval ``delta`` (seconds).

    Parameters
    ----------
    delta : float
        Sampling interval in seconds. Must divide a day (sub-daily).

    Returns
    -------
    int
        ``round(86400 / delta)``.

    Raises
    ------
    ValueError
        If ``delta`` is not sub-daily (``> 86400``), or does not evenly divide a
        day (so days would not have a consistent number of samples).
    """
    if delta <= 0:
        raise ValueError(f"delta must be positive; got {delta}")
    if delta > _SECONDS_PER_DAY:
        raise ValueError(
            f"heat map is a sub-daily diagnostic; delta={delta}s is coarser than "
            f"one day. Nothing to fold."
        )
    ratio = _SECONDS_PER_DAY / delta
    n = int(round(ratio))
    if abs(ratio - n) > 1e-6:
        raise ValueError(
            f"delta={delta}s does not evenly divide a day (86400s); the fold "
            f"needs a whole number of samples per day."
        )
    return n


def fold_to_2d(y, delta, trim_empty=True):
    """Fold a 1-D signal into a ``(n_steps, n_days)`` time-of-day x day matrix.

    Column ``j`` is day ``j``; row ``i`` is the ``i``-th sample within a day.
    The reshape is column-major (``order='F'``) so consecutive samples fill down
    a day-column before moving to the next day.

    Parameters
    ----------
    y : numpy.ndarray, shape (T,)
        The signal on a standardized, midnight-anchored regular grid spanning
        whole days (raw ``y`` or any solved component/residual value). NaNs are
        preserved (they render as gaps).
    delta : float
        Sampling interval in seconds (from :func:`signaldecomp.time_axis`).
    trim_empty : bool
        If True (default), drop leading/trailing all-NaN days.

    Returns
    -------
    numpy.ndarray, shape (n_steps, n_days)
        The folded matrix.

    Raises
    ------
    ValueError
        If ``delta`` is not sub-daily / does not divide a day, or if ``len(y)``
        is not a whole number of days.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D; got ndim={y.ndim}")
    n_steps = steps_per_day(delta)
    T = y.shape[0]
    if T % n_steps != 0:
        raise ValueError(
            f"len(y)={T} is not a whole number of days at {n_steps} samples/day "
            f"({T / n_steps:.3f} days). Standardize the axis first so the record "
            f"spans midnight-to-midnight whole days."
        )
    D = y.reshape(n_steps, -1, order="F").copy()
    if trim_empty:
        D, _ = _trim_empty_days(D)
    return D


def fold_from_standardized(std_out, key="y", trim_empty=True):
    """Fold the output of :func:`signaldecomp.time_axis.standardize_time_axis`.

    Convenience wrapper: reads ``delta`` from the standardized dict and folds
    the array under ``key`` (default the signal ``"y"``; pass a solved
    component's array explicitly via :func:`fold_to_2d` instead).

    Parameters
    ----------
    std_out : dict
        The dict returned by ``standardize_time_axis`` (needs ``"delta"`` and
        the array under ``key``).
    key : str
        Which array in ``std_out`` to fold (default ``"y"``).
    trim_empty : bool
        Passed to :func:`fold_to_2d`.

    Returns
    -------
    numpy.ndarray
        The folded matrix.
    """
    return fold_to_2d(std_out[key], std_out["delta"], trim_empty=trim_empty)


def plot_heatmap(
    D,
    signed=False,
    ax=None,
    figsize=(12, 6),
    title=None,
    cbar_label="value",
    day_axis=None,
    year_lines=False,
):
    """Render a folded matrix as a heat map (days on x, time-of-day on y).

    Parameters
    ----------
    D : numpy.ndarray, shape (n_steps, n_days)
        A folded matrix from :func:`fold_to_2d`. NaNs render as gaps.
    signed : bool
        If False (default), use the sequential ``plasma`` colormap -- suited to
        non-negative signals. If True, use the diverging ``seismic`` colormap
        centered at zero (``TwoSlopeNorm(vcenter=0)``) -- suited to signed data
        such as a residual or a detrended component, where zero should read as
        the neutral (white) midpoint.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; a new figure/axis is created if None.
    figsize : tuple
        Figure size when creating a new figure.
    title : str, optional
        Axis title.
    cbar_label : str
        Colorbar label.
    day_axis : pandas.DatetimeIndex, optional
        Per-column dates; if given, x-ticks are labeled by month (or by year for
        records spanning more than ~1.5 years).
    year_lines : bool
        If True and ``day_axis`` is given, draw vertical lines at tick dates.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the heat map.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import TwoSlopeNorm

    if ax is None:
        fig, ax = plt.subplots(nrows=1, figsize=figsize)
    else:
        fig = ax.get_figure()

    if signed:
        base = colormaps["seismic"]
        finite = D[np.isfinite(D)]
        vmax = np.max(np.abs(finite)) if finite.size else 1.0
        vmax = vmax if vmax > 0 else 1.0
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    else:
        base = colormaps["plasma"]
        norm = None
    # NaN gaps render as a visible light gray (with_extremes returns a new cmap;
    # set_bad is pending deprecation).
    cmap = base.with_extremes(bad="0.85")

    im = ax.imshow(
        D, cmap=cmap, norm=norm, interpolation="none", aspect="auto", origin="upper"
    )
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel("Day number")
    ax.set_yticks([])
    ax.set_ylabel("Time of day")

    if day_axis is not None:
        day_axis = day_axis[: D.shape[1]]
        if D.shape[1] >= 356 * 1.5:
            mask = (day_axis.month == 1) & (day_axis.day == 1)
            ticks = np.arange(D.shape[1])[mask]
            ax.set_xticks(ticks)
            ax.set_xticklabels(day_axis[ticks].year)
            ax.set_xlabel("Year")
        else:
            mask = day_axis.day == 1
            ticks = np.arange(D.shape[1])[mask]
            ax.set_xticks(ticks)
            ax.set_xticklabels(day_axis[ticks].month)
            ax.set_xlabel("Month")
        if year_lines:
            for d in ticks:
                ax.axvline(d, ls="--", color="gray", linewidth=1)
    return fig
