"""Domain-agnostic validation for signal decompositions.

Three operations, each parameterized by:

- a ``build_fn(y) -> built`` closure that rebuilds the problem on new/modified
  data (wrapping the user's ``make_problem(...)`` call), and
- an ``extractor(out) -> scalar | dict[str, scalar]`` that pulls the quantity
  of interest from a solved output.

Both data-varying operations (bootstrap, expanding window) REBUILD and re-solve
-- they change ``y``/``T``, so a fixed ``cp.Parameter`` cannot be reused.

Nothing here is domain-specific: the extractor decides what quantity matters
(a trend slope, a seasonal amplitude, a ratio), so the same machinery serves
any decomposition.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from signaldecomp.decompose import solve

_OPTIMAL = ("optimal", "optimal_inaccurate")


def _as_dict(value):
    """Normalize an extractor return (scalar or dict) to a dict of scalars.

    A bare scalar is wrapped under the key "value"; a dict passes through, so
    every downstream routine can treat the multi-quantity case uniformly.
    """
    if isinstance(value, dict):
        return value
    return {"value": value}


def _reconstruction(out):
    """Structural reconstruction: the sum of all components except the residual.

    Equivalently y - residual on observed entries. Built from the solved
    component values (the full-length role arrays), so it works for any model.
    """
    resid = out["values"]["residual"]
    T = resid.shape[0]
    recon = np.zeros(T)
    for role, val in out["values"].items():
        if role == "residual":
            continue
        arr = np.asarray(val)
        if arr.shape == (T,):
            recon = recon + arr
    return recon


def bootstrap_ci(
    y,
    build_fn,
    extractor,
    block_size,
    n_resamples=500,
    confidence_level=68.2,
    min_success_fraction=0.5,
    random_state=None,
    solver=None,
):
    """Moving-block bootstrap confidence intervals for extracted quantities.

    Resamples the observed residuals in contiguous blocks, adds them back to the
    structural reconstruction, refits, and re-extracts -- building an empirical
    distribution for each quantity the extractor returns.

    Parameters
    ----------
    y : numpy.ndarray, shape (T,)
        The observed signal (NaN where missing). The mask is preserved across
        resamples: masked entries stay NaN so each refit sees the same missing
        pattern.
    build_fn : callable
        build_fn(y) -> built; rebuilds the problem on a resampled signal.
    extractor : callable
        extractor(out) -> scalar or dict; the quantity(ies) to bootstrap.
    block_size : int
        Moving-block length, in samples. REQUIRED -- there is no safe default.
        Choose it to preserve the dependence still present in the residual:
        roughly the longest correlation length left unexplained by the model
        (often ~ one dominant period). The right value depends on the data and
        the components in the model.
    n_resamples : int
        Number of bootstrap replicates.
    confidence_level : float
        Central interval width in percent (default 68.2, i.e. ~1 sigma).
    min_success_fraction : float
        If fewer than this fraction of replicates solve successfully, the CI is
        considered unreliable and an empty dict is returned.
    random_state : int, optional
        Seed for reproducibility.
    solver : optional
        Passed through to solve() (default CLARABEL).

    Returns
    -------
    dict
        Maps each quantity key to a numpy array [lower, upper] at the requested
        confidence level. Empty dict if too few replicates succeeded. For a
        scalar extractor the single key is "value".
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1; got {block_size}")
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    valid_idx = np.where(mask)[0]
    M = valid_idx.size
    if M == 0:
        raise ValueError("y has no observed entries to bootstrap.")

    def _solve(sig):
        return solve(build_fn(sig)) if solver is None else solve(build_fn(sig), solver=solver)

    # Point solve: the residual and reconstruction we resample around.
    point_out = _solve(y)
    resid_full = point_out["values"]["residual"]
    recon_full = _reconstruction(point_out)
    res_valid = resid_full[valid_idx]

    rng = np.random.default_rng(random_state)
    L = min(block_size, M)
    n_blocks = int(np.ceil(M / L))

    collected = {}
    for _ in range(n_resamples):
        starts = rng.integers(0, M - L + 1, size=n_blocks)
        resampled = np.concatenate([res_valid[s : s + L] for s in starts])[:M]
        y_star = recon_full.copy()
        y_star[valid_idx] = recon_full[valid_idx] + resampled
        y_star[~mask] = np.nan
        try:
            out_b = _solve(y_star)
        except Exception:
            continue
        if out_b["status"] not in _OPTIMAL:
            continue
        for k, v in _as_dict(extractor(out_b)).items():
            collected.setdefault(k, []).append(v)

    n_success = len(next(iter(collected.values()), []))
    if n_success < min_success_fraction * n_resamples:
        return {}

    lower_pct = (100 - confidence_level) / 2
    upper_pct = 100 - lower_pct
    return {
        k: np.percentile(np.array(v), [lower_pct, upper_pct], axis=0)
        for k, v in collected.items()
    }


def valid_endpoints(y, min_window, step):
    """Window lengths for expanding-window analysis, snapped to observed samples.

    Nominal lengths ``range(min_window, T+1, step)`` (plus the full length) are
    each snapped *backward* to end on the nearest observed (non-NaN) sample,
    then deduplicated. This guarantees every window ends on real data.

    Parameters
    ----------
    y : numpy.ndarray, shape (T,)
        The signal (NaN where missing).
    min_window : int
        Minimum window length in samples. REQUIRED -- choose from the data's
        time scales (e.g. at least one dominant period).
    step : int
        Nominal spacing between successive window lengths, in samples.

    Returns
    -------
    numpy.ndarray of int
        Unique sorted window lengths, each ending on an observed sample.
    """
    y = np.asarray(y, dtype=float)
    T = y.shape[0]
    valid_ix = np.where(~np.isnan(y))[0]
    if valid_ix.size == 0:
        raise ValueError("y contains no observed (non-NaN) samples.")
    nominal = np.arange(min_window, T + 1, step)
    if nominal.size == 0 or nominal[-1] != T:
        nominal = np.append(nominal, T)
    snapped = []
    for n in nominal:
        candidates = valid_ix[valid_ix < n]
        if candidates.size == 0:
            continue
        snapped.append(int(candidates[-1]) + 1)
    snapped = np.unique(snapped)
    return snapped[snapped >= min_window]


def expanding_window_stability(
    y,
    build_fn,
    extractor,
    min_window,
    step,
    tol,
    solver=None,
):
    """Track how extracted quantities stabilize as the data record grows.

    Solves the decomposition on growing prefixes ``y[:n]`` (n from
    :func:`valid_endpoints`), records the extracted quantity(ies) per window,
    and reports between-window change and a convergence point.

    Parameters
    ----------
    y : numpy.ndarray, shape (T,)
        The full signal (NaN where missing).
    build_fn : callable
        ``build_fn(y_window) -> built``; rebuilt per window (T changes).
    extractor : callable
        ``extractor(out) -> scalar | dict``; the quantity(ies) to track.
    min_window : int
        Minimum (first) window length in samples. REQUIRED -- no default;
        choose from the data's time scales.
    step : int
        Nominal spacing between window lengths, in samples.
    tol : float
        Convergence tolerance (same units as the extracted quantity). A
        quantity is "converged" at the earliest window after which it stays
        within ``tol`` of its final-window value for all later windows.
    solver : optional
        Passed through to :func:`solve` (default CLARABEL).

    Returns
    -------
    dict
        - ``"windows"`` : ndarray of solved window lengths.
        - ``"history"`` : dict key -> ndarray over windows (NaN where the solve
          failed or the key was absent).
        - ``"delta"``   : dict key -> ndarray of |change| between successive
          windows.
        - ``"converged_at"`` : dict key -> window length of convergence (or
          None if it never settles within ``tol``).
        - ``"tol"`` : the tolerance used.
    """
    y = np.asarray(y, dtype=float)
    windows = valid_endpoints(y, min_window, step)
    F = windows.size

    def _solve(sig):
        return solve(build_fn(sig)) if solver is None else solve(build_fn(sig), solver=solver)

    per_window = []
    keys = set()
    for n in windows:
        try:
            out = _solve(y[:n])
            rec = _as_dict(extractor(out)) if out["status"] in _OPTIMAL else None
        except Exception:
            rec = None
        per_window.append(rec)
        if rec is not None:
            keys.update(rec.keys())

    history = {k: np.full(F, np.nan) for k in keys}
    for i, rec in enumerate(per_window):
        if rec is None:
            continue
        for k in keys:
            if k in rec:
                history[k][i] = rec[k]

    delta = {}
    for k, arr in history.items():
        d = np.full(max(F - 1, 0), np.nan)
        for i in range(F - 1):
            if not (np.isnan(arr[i]) or np.isnan(arr[i + 1])):
                d[i] = abs(arr[i + 1] - arr[i])
        delta[k] = d

    converged_at = {}
    for k, arr in history.items():
        finite = arr[~np.isnan(arr)]
        if finite.size == 0:
            converged_at[k] = None
            continue
        final = finite[-1]
        within = np.abs(arr - final) <= tol
        out_idx = np.where(~within)[0]
        if out_idx.size == 0:
            converged_at[k] = int(windows[0])
        elif out_idx[-1] == F - 1:
            converged_at[k] = None
        else:
            converged_at[k] = int(windows[out_idx[-1] + 1])

    return {
        "windows": windows,
        "history": history,
        "delta": delta,
        "converged_at": converged_at,
        "tol": tol,
    }


def holdout_select(
    y,
    candidates,
    holdout_fraction=0.2,
    holdout_slice=None,
    metric="rmse",
    solver=None,
):
    """Select among candidate models by held-out imputation error.

    Holds out a contiguous block of *observed* entries (masking them to NaN),
    refits each candidate on the reduced data, and scores how well each
    reconstructs the held-out truth. This reuses the decomposition's native
    missing-data mechanism: a held-out entry is just a masked entry whose true
    value we happen to know.

    Because scoring is by imputation of held-out data (not training fit), a more
    flexible model does not automatically win -- overfitting is penalized. The
    candidates may differ in components or in hyperparameters.

    Parameters
    ----------
    y : numpy.ndarray, shape (T,)
        The observed signal (NaN where already missing).
    candidates : dict[str, callable]
        Maps a model name to a ``build_fn(y) -> built``.
    holdout_fraction : float
        Size of the contiguous held-out block as a fraction of the observed
        length. Used only when ``holdout_slice`` is None; the block is placed in
        the centre of the record to avoid edge effects.
    holdout_slice : slice, optional
        Explicit held-out index range (into the full signal). Overrides
        ``holdout_fraction`` when given.
    metric : str
        Scoring metric on the held-out entries: ``"rmse"`` (default) or
        ``"mae"``.
    solver : optional
        Passed through to :func:`solve` (default CLARABEL).

    Returns
    -------
    dict
        - ``"scores"`` : dict name -> held-out error (NaN if that model failed).
        - ``"best"``   : name of the lowest-error model (None if all failed).
        - ``"holdout_index"`` : the integer indices held out and scored.
        - ``"metric"`` : the metric used.

    Notes
    -----
    V1 uses a single contiguous held-out block. Blocked (rather than random)
    hold-out respects temporal correlation. K-fold / multiple blocks are a
    natural extension not built here.
    """
    if metric not in ("rmse", "mae"):
        raise ValueError(f"metric must be 'rmse' or 'mae'; got {metric!r}")
    y = np.asarray(y, dtype=float)
    T = y.shape[0]
    observed = np.where(~np.isnan(y))[0]
    if observed.size == 0:
        raise ValueError("y has no observed entries to hold out.")

    if holdout_slice is not None:
        holdout_index = np.arange(T)[holdout_slice]
        # Only score entries that were actually observed.
        holdout_index = holdout_index[~np.isnan(y[holdout_index])]
    else:
        if not 0.0 < holdout_fraction < 1.0:
            raise ValueError(
                f"holdout_fraction must be in (0, 1); got {holdout_fraction}"
            )
        n_hold = max(1, int(round(holdout_fraction * observed.size)))
        start = (observed.size - n_hold) // 2  # centre block, in observed space
        holdout_index = observed[start : start + n_hold]
    if holdout_index.size == 0:
        raise ValueError("the held-out block covers no observed entries.")

    y_train = y.copy()
    y_train[holdout_index] = np.nan  # mask the held-out truth
    truth = y[holdout_index]

    def _solve(sig):
        return solve(build_fn(sig)) if solver is None else solve(build_fn(sig), solver=solver)

    scores = {}
    for name, build_fn in candidates.items():
        try:
            out = _solve(y_train)
            if out["status"] not in _OPTIMAL:
                scores[name] = float("nan")
                continue
            recon = _reconstruction(out)
            err = recon[holdout_index] - truth
            if metric == "rmse":
                scores[name] = float(np.sqrt(np.mean(err**2)))
            else:
                scores[name] = float(np.mean(np.abs(err)))
        except Exception:
            scores[name] = float("nan")

    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    best = min(finite, key=finite.get) if finite else None
    return {
        "scores": scores,
        "best": best,
        "holdout_index": holdout_index,
        "metric": metric,
    }
