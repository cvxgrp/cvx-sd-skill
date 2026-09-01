"""Pure design-matrix utilities for exogenous signal components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OffsetBasis:
    """Concatenated offset basis and rows where every block is available."""

    design: np.ndarray
    valid_mask: np.ndarray
    offsets: tuple[int, ...]
    block_width: int


def normalize_offsets(offsets):
    """Validate ordered integer offsets while preserving caller order."""
    try:
        values = tuple(offsets)
    except TypeError as exc:
        raise TypeError("offsets must be a non-empty iterable of integers.") from exc
    if not values:
        raise ValueError("offsets must contain at least one value.")
    normalized = []
    for offset in values:
        if (
            not isinstance(offset, (int, np.integer))
            or isinstance(offset, (bool, np.bool_))
        ):
            raise TypeError(f"offsets must contain only integers; got {offset!r}.")
        normalized.append(int(offset))
    if len(set(normalized)) != len(normalized):
        raise ValueError("offsets must be unique.")
    return tuple(normalized)


def make_offset_basis(basis, offsets=(0,)):
    """Construct ordered offset blocks using ``out[t] = basis[t - offset]``.

    Positive offsets use past rows and pad the beginning; negative offsets use
    future rows and pad the end. Undefined or non-finite source rows become
    finite zeros in the returned design and are excluded by ``valid_mask``.
    """
    basis = np.asarray(basis, dtype=float)
    if basis.ndim == 1:
        basis = basis[:, np.newaxis]
    if basis.ndim != 2:
        raise ValueError(f"basis must be 1-D or 2-D; got shape {basis.shape}.")
    if basis.shape[1] == 0:
        raise ValueError("basis must contain at least one column.")
    offsets = normalize_offsets(offsets)
    T, width = basis.shape
    blocks = []
    valid_blocks = []
    source_finite = np.all(np.isfinite(basis), axis=1)

    for offset in offsets:
        block = np.zeros((T, width), dtype=float)
        valid = np.zeros(T, dtype=bool)
        if offset == 0:
            block[:] = np.nan_to_num(basis, nan=0.0, posinf=0.0, neginf=0.0)
            valid[:] = source_finite
        elif offset > 0:
            if offset < T:
                source = basis[:-offset]
                block[offset:] = np.nan_to_num(
                    source, nan=0.0, posinf=0.0, neginf=0.0
                )
                valid[offset:] = source_finite[:-offset]
        else:
            lead = -offset
            if lead < T:
                source = basis[lead:]
                block[:-lead] = np.nan_to_num(
                    source, nan=0.0, posinf=0.0, neginf=0.0
                )
                valid[:-lead] = source_finite[lead:]
        blocks.append(block)
        valid_blocks.append(valid)

    return OffsetBasis(
        design=np.hstack(blocks),
        valid_mask=np.logical_and.reduce(valid_blocks),
        offsets=offsets,
        block_width=width,
    )


def offset_source_mask(fit_mask, offsets=(0,)):
    """Rows of an original driver consumed by fitted target rows.

    This is the support mask to use for fitted knot/domain construction. It is
    the union of source rows ``t - offset`` for every target row ``t`` selected
    by ``fit_mask`` and every requested offset.
    """
    mask = np.asarray(fit_mask)
    if mask.dtype != np.bool_:
        raise TypeError("fit_mask must have boolean dtype.")
    if mask.ndim != 1:
        raise ValueError(f"fit_mask must be 1-D; got shape {mask.shape}.")
    offsets = normalize_offsets(offsets)
    T = mask.shape[0]
    source_mask = np.zeros(T, dtype=bool)
    for offset in offsets:
        if offset == 0:
            source_mask |= mask
        elif offset > 0:
            if offset < T:
                source_mask[:-offset] |= mask[offset:]
        else:
            lead = -offset
            if lead < T:
                source_mask[lead:] |= mask[:-lead]
    return source_mask
