"""Fourier basis and regularization matrices for periodic signal components.

Vendored and trimmed from ``spcqe`` (https://github.com/cvxgrp/spcqe).
Original authors: Bennet Meyers, Mehmet Giray, Aramis Dufour.

Adapted for this skill: the ``trend`` option is removed -- a linear trend is a
separate signal-decomposition component here, not something mixed into the
Fourier terms. The ``standing_wave`` and ``custom_basis`` hooks are retained.

Column layout of :func:`make_basis_matrix` (order is fixed by construction):

1. **Offset** -- a single column of ones (the DC / constant term), always at
   index 0.
2. **Per-period Fourier blocks** -- for each period, ``2 * num_harmonics``
   columns interleaved ``[cos, sin, cos, sin, ...]`` (or ``num_harmonics`` sine
   columns for a standing-wave period).
3. **Pairwise cross-term blocks** -- for every *pair* of periods, the outer
   product of their Fourier columns (the multi-periodic / quasiperiodic
   construction). Only pairwise; no triple-and-higher products.

Because the offset is always column 0, dropping the DC degree of freedom (so a
seasonal component cannot absorb a constant that belongs to the trend) is simply
``B[:, 1:]`` and ``W.tocsr()[:, 1:]``.

Note on multi-period use: passing several periods to one call builds a coupled
*tensor* basis (with cross terms), which grows multiplicatively. For independent
additive seasonals, build one component per period instead. See
``reference/periodic-and-time.md``.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.sparse import spdiags


def make_basis_matrix(
    num_harmonics,
    length,
    periods,
    standing_wave=False,
    max_cross_k=None,
    custom_basis=None,
):
    """Create a Fourier basis matrix for one or multiple periods.

    Builds a Fourier basis for each period, an offset column, and pairwise
    cross-terms between periods.

    Parameters
    ----------
    num_harmonics : int or iterable of int
        Number of harmonics per period (single value applies to all periods).
    length : int
        Length of the time series.
    periods : float or iterable of float
        Period length(s) in samples. Each period yields a Fourier block.
    standing_wave : bool or iterable of bool
        Per-period flag. A standing-wave basis uses only sine terms at half the
        angular frequency (zero at the boundaries). Defaults to False.
    max_cross_k : int, optional
        Cap on the number of harmonics used per side when forming cross-terms.
    custom_basis : dict, optional
        Map period-index -> basis matrix, substituted for that period's block.

    Returns
    -------
    numpy.ndarray
        Basis matrix; column 0 is the offset (all ones).
    """
    sort_idx, Ps, num_harmonics, standing_wave = initialize_arrays(
        num_harmonics, periods, standing_wave, custom_basis
    )
    t_values = np.arange(length)
    B_fourier = []
    for ix, P in enumerate(Ps):
        i_values = np.arange(1, num_harmonics[ix] + 1)[:, np.newaxis]
        if standing_wave[ix]:
            w = 2 * np.pi / (P * 2)
            B_sin = np.sin(i_values * w * np.mod(t_values, P))
            B_f = np.empty((length, num_harmonics[ix]), dtype=float)
            B_f[:] = B_sin.T
        else:
            w = 2 * np.pi / P
            B_cos = np.cos(i_values * w * t_values)
            B_sin = np.sin(i_values * w * t_values)
            B_f = np.empty((length, 2 * num_harmonics[ix]), dtype=float)
            B_f[:, ::2] = B_cos.T
            B_f[:, 1::2] = B_sin.T
        B_fourier.append(B_f)
    # Substitute any user-provided custom bases.
    if custom_basis is not None:
        for ix, val in custom_basis.items():
            if val.shape[0] != length:
                multiplier = max(1, length // val.shape[0] + 1)
                new_val = np.tile(val, (multiplier, 1))[:length]
            else:
                new_val = val[:length]
            ixt = np.where(sort_idx == ix)[0][0]
            B_fourier[ixt] = new_val
    # Offset column (DC), always first.
    B0 = [np.ones((length, 1))]
    # Pairwise cross-terms (empty list when there is only one period).
    C = [
        cross_bases(*base_tuple, max_k=max_cross_k)
        for base_tuple in combinations(B_fourier, 2)
    ]
    return np.hstack(B0 + B_fourier + C)


def make_regularization_matrix(
    num_harmonics,
    weight,
    periods,
    standing_wave=False,
    max_cross_k=None,
    custom_basis=None,
):
    """Create the diagonal regularization matrix matching :func:`make_basis_matrix`.

    Diagonal weights follow the Dirichlet energy of the basis functions:
    ``weight * 2*pi / sqrt(P) * i`` for harmonic ``i`` (repeated for the cos/sin
    pair when not a standing wave). The offset column is unregularized (weight
    zero).

    Parameters
    ----------
    num_harmonics : int or iterable of int
        Number of harmonics per period.
    weight : float
        Overall regularization weight.
    periods : float or iterable of float
        Period length(s) in samples.
    standing_wave : bool or iterable of bool
        Per-period standing-wave flag.
    max_cross_k : int, optional
        Cap on harmonics per side for cross-terms.
    custom_basis : dict, optional
        Custom bases (their columns are regularized by harmonic index).

    Returns
    -------
    scipy.sparse.dia_matrix
        Square diagonal regularization matrix; entry 0 (offset) is zero.
    """
    sort_idx, Ps, num_harmonics, standing_wave = initialize_arrays(
        num_harmonics, periods, standing_wave, custom_basis
    )
    ls_original = [weight * (2 * np.pi) / np.sqrt(P) for P in Ps]
    i_value_list = []
    for ix, nh in enumerate(num_harmonics):
        if standing_wave[ix]:
            i_value_list.append(np.arange(1, nh + 1))
        else:
            i_value_list.append(np.repeat(np.arange(1, nh + 1), 2))
    blocks_original = [iv * lx for iv, lx in zip(i_value_list, ls_original)]
    if custom_basis is not None:
        for ix, val in custom_basis.items():
            ixt = np.where(sort_idx == ix)[0][0]
            blocks_original[ixt] = ls_original[ixt] * np.arange(1, val.shape[1] + 1)
    if max_cross_k is not None:
        max_cross_k *= 2
    # Periods are sorted (by initialize_arrays), so cross blocks align with the
    # basis cross blocks.
    blocks_cross = [
        [l2 for l1 in c[0][:max_cross_k] for l2 in c[1][:max_cross_k]]
        for c in combinations(blocks_original, 2)
    ]
    # Offset weight is zero (unregularized), matching the offset column.
    coeff_i = np.concatenate([np.zeros(1)] + blocks_original + blocks_cross)
    return spdiags(coeff_i, 0, coeff_i.size, coeff_i.size)


def initialize_arrays(num_harmonics, periods, standing_wave, custom_basis):
    """Validate inputs and sort periods (and their harmonics) descending.

    Returns the sort indices along with period, harmonic-count, and
    standing-wave arrays sorted by descending period, so the basis and
    regularization matrices share a consistent column order.

    Raises
    ------
    TypeError
        If ``custom_basis`` is neither a dict nor None.
    ValueError
        If ``num_harmonics`` or ``standing_wave`` length mismatches ``periods``.
    """
    if not (isinstance(custom_basis, dict) or custom_basis is None):
        raise TypeError(
            "custom_basis should be a dictionary mapping the period index to "
            "its basis matrix."
        )
    Ps = np.atleast_1d(periods)
    num_harmonics = np.atleast_1d(num_harmonics)
    if len(num_harmonics) == 1:
        num_harmonics = np.tile(num_harmonics, len(Ps))
    elif len(num_harmonics) != len(Ps):
        raise ValueError(
            "Pass a single num_harmonics for all periods, or one per period."
        )
    standing_wave = np.atleast_1d(standing_wave)
    if len(standing_wave) == 1:
        standing_wave = np.tile(standing_wave, len(Ps))
    elif len(standing_wave) != len(Ps):
        raise ValueError(
            "Pass a single standing_wave for all periods, or one per period."
        )
    sort_idx = np.argsort(-Ps)
    Ps = -np.sort(-Ps)
    num_harmonics = num_harmonics[sort_idx]
    standing_wave = standing_wave[sort_idx]
    return sort_idx, Ps, num_harmonics, standing_wave


def cross_bases(B_P1, B_P2, max_k=None):
    """Outer-product cross-terms between two per-period basis blocks.

    Parameters
    ----------
    B_P1, B_P2 : numpy.ndarray
        Per-period Fourier basis blocks (same number of rows).
    max_k : int, optional
        If given, use only the first ``2 * max_k`` columns of each block.

    Returns
    -------
    numpy.ndarray
        Cross-term matrix with ``B_P1_cols * B_P2_cols`` columns.
    """
    if max_k is None:
        B_P1_new = B_P1[:, :, None]
        B_P2_new = B_P2[:, None, :]
    else:
        B_P1_new = B_P1[:, : 2 * max_k, None]
        B_P2_new = B_P2[:, None, : 2 * max_k]
    result = B_P1_new * B_P2_new
    return result.reshape(result.shape[0], -1)
