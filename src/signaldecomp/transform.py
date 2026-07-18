"""Domain-agnostic pre/post-processing transforms for signal decomposition.

The log transform turns an ADDITIVE decomposition into a MULTIPLICATIVE one:
decomposing ``log(y) = x1 + x2 + ...`` and exponentiating gives
``y ~= exp(x1) * exp(x2) * ...`` in the original domain -- the structural
components *multiply* rather than add, and the residual becomes a multiplicative
factor. This is a common, domain-agnostic modeling choice (e.g. proportional
seasonality, compound growth/decay); the interpretation of any component remains
the caller's.

:func:`prepare_input` applies the transform (masking non-positive values, which
have no log) before :func:`signaldecomp.make_problem`. :func:`recover_components`
back-transforms a solved output to the original domain, choosing additive (sum)
or multiplicative (product) reconstruction to match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_input(y, log_transform=False, floor=1e-6):
    """Prepare a signal for :func:`signaldecomp.make_problem`.

    With ``log_transform=True``, non-positive values (which have no real log)
    are set to NaN -- i.e. treated as missing, the same mask mechanism the
    decomposition already uses -- and the natural log is taken. Otherwise the
    signal is returned unchanged (as a float array copy).

    Parameters
    ----------
    y : array-like
        Input signal.
    log_transform : bool
        Apply the natural-log transform (-> multiplicative decomposition).
    floor : float
        Values at or below this are treated as missing before the log.

    Returns
    -------
    numpy.ndarray
        Float array; log-transformed with NaN at non-positive entries when
        ``log_transform`` is set.
    """
    y = np.asarray(y, dtype=float).copy()
    if log_transform:
        y[y <= floor] = np.nan
        y = np.log(y)
    return y


def recover_components(out, log_transform=False, index=None):
    """Back-transform a solved decomposition to the original domain.

    General over any set of structural roles (no hard-coded component names).
    With ``log_transform=False`` this is the ordinary additive view: components
    as solved, reconstruction = their sum. With ``log_transform=True`` each
    structural component is exponentiated and the reconstruction is their
    *product* (the multiplicative model); the residual is returned as the
    multiplicative factor ``exp(residual)``.

    Must be called with the same ``log_transform`` used to build the problem
    (i.e. whether the solve ran on ``log(y)``).

    Parameters
    ----------
    out : dict
        A solved output from :func:`signaldecomp.solve`.
    log_transform : bool
        Whether the decomposition was solved in log space.
    index : pandas.Index, optional
        Index to align on; defaults to a RangeIndex.

    Returns
    -------
    pandas.DataFrame
        Columns: each structural role (original domain), ``residual``, and
        ``reconstruction``. For the multiplicative case, ``residual`` is the
        factor ``exp(residual)`` (multiply it into the reconstruction to recover
        the observed signal), and ``reconstruction`` is the product of the
        exponentiated structural components.

    Raises
    ------
    ValueError
        If ``out`` lacks a residual or ``index`` length is inconsistent.
    """
    values = out["values"]
    if "residual" not in values:
        raise ValueError("out['values'] has no 'residual'; not a solved output.")
    residual = np.asarray(values["residual"], dtype=float)
    T = residual.shape[0]

    structural = {}
    for role, val in values.items():
        if role == "residual":
            continue
        arr = np.asarray(val, dtype=float) if np.ndim(val) else None
        if arr is not None and arr.shape == (T,):
            structural[role] = arr

    if index is None:
        index = pd.RangeIndex(T)
    elif len(index) != T:
        raise ValueError(f"index length {len(index)} != component length {T}.")

    cols = {}
    if log_transform:
        recovered = {r: np.exp(a) for r, a in structural.items()}
        for r, a in recovered.items():
            cols[r] = a
        cols["residual"] = np.exp(residual)
        cols["reconstruction"] = (
            np.prod(np.stack(list(recovered.values()), axis=0), axis=0)
            if recovered
            else np.ones(T)
        )
    else:
        for r, a in structural.items():
            cols[r] = a
        cols["residual"] = residual
        cols["reconstruction"] = (
            np.sum(np.stack(list(structural.values()), axis=0), axis=0)
            if structural
            else np.zeros(T)
        )

    return pd.DataFrame(cols, index=index)


def recover_frame(out, log_transform=False, index=None, y=None):
    """Plot-ready original-domain frame for :func:`signaldecomp.plot_decomposition`.

    The transform-aware counterpart to
    :func:`signaldecomp.components_to_frame`: back-transforms a solved
    decomposition (via :func:`recover_components`) and, if given, attaches the
    observed signal ``y`` as a column for the top panel.

    The ``residual`` column is referenced to the model's "no deviation" value:
    **0** for an additive model, **1** for a multiplicative/log model (where the
    residual is the factor ``exp(residual)``). Pass the matching ``residual_ref``
    to :func:`plot_decomposition` (0 or 1) so the residual panel's fill and
    baseline are drawn correctly.

    Parameters
    ----------
    out : dict
        A solved output from :func:`signaldecomp.solve`.
    log_transform : bool
        Whether the decomposition was solved in log space.
    index : pandas.Index, optional
        Index to align on; defaults to a RangeIndex.
    y : numpy.ndarray, optional
        The observed signal in the ORIGINAL domain (not log-transformed); added
        as a ``y`` column when given.

    Returns
    -------
    pandas.DataFrame
        Columns: each original-domain role, ``residual`` (referenced to 0 or 1),
        ``reconstruction``, and ``y`` if supplied.
    """
    df = recover_components(out, log_transform=log_transform, index=index)
    if y is not None:
        y = np.asarray(y, dtype=float)
        if y.shape[0] != len(df):
            raise ValueError(f"y length {y.shape[0]} != frame length {len(df)}.")
        df["y"] = y
    return df
