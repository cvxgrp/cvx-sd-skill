"""Tensor-product design utilities for exogenous interactions.

The column-product ordering is adapted from TSGAM: the right-factor column
index varies fastest, so a flat coefficient vector corresponds to a
``(left_width, right_width)`` matrix in C order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
@dataclass(frozen=True)
class InteractionBasis:
    """Tensor-product basis with explicit factor widths and row availability."""

    design: np.ndarray
    valid_mask: np.ndarray
    left_basis: np.ndarray
    right_basis: np.ndarray
    left_width: int
    right_width: int

    def validate_interaction_only(self, fit_mask, *, context="interaction basis"):
        """Require no fitted-support overlap with intercept or main effects."""
        mask = np.asarray(fit_mask)
        if mask.dtype != np.bool_:
            raise TypeError(f"{context}: fit_mask must have boolean dtype.")
        if mask.shape != self.valid_mask.shape:
            raise ValueError(
                f"{context}: fit_mask has shape {mask.shape}, expected "
                f"{self.valid_mask.shape}."
            )
        if not mask.any():
            raise ValueError(f"{context}: fit_mask selects no rows.")
        if np.any(mask & ~self.valid_mask):
            raise ValueError(
                f"{context}: fit_mask includes rows where a factor is unavailable."
            )

        lower_order = np.column_stack(
            [
                np.ones(int(mask.sum())),
                self.left_basis[mask],
                self.right_basis[mask],
            ]
        )
        interaction = self.design[mask]
        lower_scaled = _column_normalized(lower_order)
        interaction_scaled = _column_normalized(interaction)
        combined = np.hstack([lower_scaled, interaction_scaled])
        singular_values = np.linalg.svd(combined, compute_uv=False)
        tolerance = (
            max(combined.shape)
            * np.finfo(float).eps
            * float(singular_values[0])
        )
        lower_rank = _rank_at_tolerance(lower_scaled, tolerance)
        interaction_rank = _rank_at_tolerance(interaction_scaled, tolerance)
        combined_rank = int(np.count_nonzero(singular_values > tolerance))
        overlap = lower_rank + interaction_rank - combined_rank
        if overlap > 0:
            raise ValueError(
                f"{context} is not interaction-only on the final fit_mask: "
                f"its span overlaps [intercept, left main effect, right main "
                f"effect] in {overlap} direction(s) "
                f"(lower_order_rank={lower_rank}, "
                f"interaction_rank={interaction_rank}, "
                f"combined_rank={combined_rank}, "
                f"rank_tolerance={tolerance:.6g})."
            )


def _as_factor_basis(values, *, name):
    basis = np.asarray(values, dtype=float)
    if basis.ndim == 1:
        basis = basis[:, np.newaxis]
    elif basis.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D; got shape {basis.shape}.")
    if basis.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one column.")
    return basis


def _column_normalized(basis):
    """Return a finite, unit-column-scaled copy for rank comparisons."""
    scales = np.max(np.abs(basis), axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    scaled = basis / scales
    norms = np.linalg.norm(scaled, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    return scaled / norms


def _rank_at_tolerance(basis, tolerance):
    singular_values = np.linalg.svd(basis, compute_uv=False)
    return int(np.count_nonzero(singular_values > tolerance))


def _constant_is_in_column_space(basis):
    """Return whether a finite basis contains an intercept direction."""
    if basis.shape[0] == 0:
        return False

    # Normalize columns before the rank comparison so the semantic check is
    # invariant to factor units and remains finite for very large inputs.
    scaled = _column_normalized(basis)
    constant = np.ones((basis.shape[0], 1)) / np.sqrt(basis.shape[0])
    augmented = np.hstack([scaled, constant])
    singular_values = np.linalg.svd(augmented, compute_uv=False)
    tolerance = (
        max(augmented.shape)
        * np.finfo(float).eps
        * float(singular_values[0])
    )
    augmented_rank = int(np.count_nonzero(singular_values > tolerance))

    basis_rank = _rank_at_tolerance(scaled, tolerance)
    return augmented_rank == basis_rank


def make_interaction_basis(left_basis, right_basis):
    """Build row-wise tensor products in deterministic TSGAM column order.

    One-dimensional inputs are treated as one-column bases. Each factor must be
    offset-free: the constant vector cannot lie in its column space over the
    jointly valid rows. This prevents direct intercept-column leakage, but is
    not by itself a complete interaction-only audit; ``exog_interaction`` also
    checks the tensor-product span against lower-order terms on the final
    fitting mask during problem assembly.

    A row is valid only when every entry in both factor bases is finite. Invalid
    rows are represented by exact zeros in the returned design, allowing the
    design to remain finite while ``valid_mask`` removes those rows from
    fitting. Products that overflow or underflow to zero despite finite,
    nonzero factors raise with a scaling error before a component or CVXPY
    problem is constructed.

    If ``left_basis`` has ``q`` columns and ``right_basis`` has ``r`` columns,
    the output columns are ordered as
    ``left[0]*right[0], ..., left[0]*right[r-1], left[1]*right[0], ...``.
    Thus ``design @ coef.reshape(-1, order="C")`` equals
    ``sum((left @ coef) * right, axis=1)`` for a ``(q, r)`` coefficient matrix.
    """
    left = _as_factor_basis(left_basis, name="left_basis")
    right = _as_factor_basis(right_basis, name="right_basis")
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            "left_basis and right_basis must have the same number of rows; "
            f"got {left.shape[0]} and {right.shape[0]}."
        )

    valid_mask = np.all(np.isfinite(left), axis=1) & np.all(
        np.isfinite(right), axis=1
    )
    left_clean = np.where(np.isfinite(left), left, 0.0)
    right_clean = np.where(np.isfinite(right), right, 0.0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        product_tensor = (
            left_clean[:, :, np.newaxis] * right_clean[:, np.newaxis, :]
        )
    design = product_tensor.reshape(left.shape[0], -1)
    if not np.all(np.isfinite(design[valid_mask])):
        raise ValueError(
            "interaction products overflowed or underflowed for finite factor "
            "values; rescale the factor bases before constructing the "
            "interaction."
        )
    nonzero_operands = (
        (left_clean[:, :, np.newaxis] != 0.0)
        & (right_clean[:, np.newaxis, :] != 0.0)
        & valid_mask[:, np.newaxis, np.newaxis]
    )
    if np.any((product_tensor == 0.0) & nonzero_operands):
        raise ValueError(
            "interaction products overflowed or underflowed for finite factor "
            "values; rescale the factor bases before constructing the "
            "interaction."
        )
    for basis, name in (
        (left[valid_mask], "left_basis"),
        (right[valid_mask], "right_basis"),
    ):
        if _constant_is_in_column_space(basis):
            raise ValueError(
                f"{name} must be offset-free on jointly valid rows; its column "
                "space contains the constant vector. Remove intercept/DC "
                "columns before constructing an interaction."
            )
    design[~valid_mask] = 0.0

    design.setflags(write=False)
    valid_mask.setflags(write=False)
    left_clean.setflags(write=False)
    right_clean.setflags(write=False)
    return InteractionBasis(
        design=design,
        valid_mask=valid_mask,
        left_basis=left_clean,
        right_basis=right_clean,
        left_width=left.shape[1],
        right_width=right.shape[1],
    )
