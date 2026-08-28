"""Numerical diagnostics and exact reparameterization for basis components.

Whitening in this module is deliberately full-rank and penalty-preserving. It
changes the coordinates presented to a solver, not the modeled function space
or the statistical penalty. Unsupported directions raise; they are never
truncated or silently passed through unwhitened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BasisRankDiagnostics:
    """Singular-geometry diagnostics for the exact fitted basis rows."""

    fit_shape: tuple[int, int]
    singular_values: np.ndarray
    rank_tolerance: float
    numerical_rank: int
    rank_margin: float
    condition_number: float
    column_normalized_condition_number: float


class BasisRankError(ValueError):
    """Raised when an exact full-rank basis reparameterization is impossible."""

    def __init__(self, message: str, diagnostics: BasisRankDiagnostics | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class BasisWhitening:
    """Exact full-rank basis reparameterization ``beta = transform @ gamma``."""

    whitened_basis: np.ndarray
    transform: np.ndarray
    diagnostics: BasisRankDiagnostics
    training_gram_error: float

    def recover_coefficients(self, numerical_coefficients):
        """Map numerical coordinates back to original basis coefficients."""
        gamma = np.asarray(numerical_coefficients, dtype=float)
        if gamma.ndim == 0 or gamma.shape[0] != self.transform.shape[1]:
            raise ValueError(
                "numerical_coefficients must have first dimension "
                f"{self.transform.shape[1]}; got shape {gamma.shape}."
            )
        return self.transform @ gamma


@dataclass(frozen=True)
class ComposedBasisWhitening:
    """Exact two-stage whitening for an ordered collection of basis blocks.

    ``transform`` maps the final numerical coefficients to the coefficients of
    ``raw_basis``. The intermediate block transforms and joint transform are
    retained so the full reparameterization can be audited.
    """

    raw_basis: np.ndarray
    whitened_basis: np.ndarray
    transform: np.ndarray
    block_slices: tuple[slice, ...]
    block_names: tuple[str, ...]
    block_whitenings: tuple[BasisWhitening, ...]
    joint_whitening: BasisWhitening
    training_gram_error: float

    @property
    def diagnostics(self):
        """Rank diagnostics for the jointly whitened concatenation."""
        return self.joint_whitening.diagnostics

    def recover_coefficients(self, numerical_coefficients):
        """Map final numerical coordinates to concatenated raw coefficients."""
        gamma = np.asarray(numerical_coefficients, dtype=float)
        if gamma.ndim == 0 or gamma.shape[0] != self.transform.shape[1]:
            raise ValueError(
                "numerical_coefficients must have first dimension "
                f"{self.transform.shape[1]}; got shape {gamma.shape}."
            )
        return self.transform @ gamma


def _validated_basis_and_mask(basis, fit_mask):
    basis = np.asarray(basis, dtype=float)
    if basis.ndim != 2:
        raise ValueError(f"basis must be 2-D; got shape {basis.shape}.")
    if basis.shape[1] == 0:
        raise ValueError("basis must contain at least one column.")
    if not np.all(np.isfinite(basis)):
        raise ValueError("basis must contain only finite values.")

    mask = np.asarray(fit_mask)
    if mask.dtype != np.bool_:
        raise TypeError("fit_mask must have boolean dtype.")
    if mask.shape != (basis.shape[0],):
        raise ValueError(
            f"fit_mask has shape {mask.shape}, expected ({basis.shape[0]},)."
        )
    if not mask.any():
        raise ValueError("fit_mask selects no rows.")
    return basis, mask


def _resolve_rank_tolerance(singular_values, fit_shape, rank_tolerance):
    s_max = float(singular_values[0]) if singular_values.size else 0.0
    if rank_tolerance is None:
        return max(fit_shape) * np.finfo(float).eps * s_max
    tolerance = float(rank_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("rank_tolerance must be finite and non-negative.")
    return tolerance


def _condition_number(singular_values):
    if singular_values.size == 0 or singular_values[-1] == 0:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def basis_rank_diagnostics(basis, fit_mask, *, rank_tolerance=None):
    """Audit the exact rows used by a fitted basis component.

    Column-normalized condition number is reported as a diagnostic only. It is
    not whitening and is not used to decide the transform.
    """
    basis, mask = _validated_basis_and_mask(basis, fit_mask)
    fitted = basis[mask]
    singular_values = np.linalg.svd(fitted, compute_uv=False)
    tolerance = _resolve_rank_tolerance(
        singular_values, fitted.shape, rank_tolerance
    )
    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    smallest = float(singular_values[-1]) if singular_values.size else 0.0
    rank_margin = (
        float(smallest / tolerance)
        if tolerance > 0
        else (float("inf") if smallest > 0 else 0.0)
    )

    column_norms = np.linalg.norm(fitted, axis=0)
    if np.any(column_norms == 0):
        normalized_condition = float("inf")
    else:
        normalized_singular_values = np.linalg.svd(
            fitted / column_norms, compute_uv=False
        )
        normalized_condition = _condition_number(normalized_singular_values)

    return BasisRankDiagnostics(
        fit_shape=fitted.shape,
        singular_values=singular_values,
        rank_tolerance=tolerance,
        numerical_rank=numerical_rank,
        rank_margin=rank_margin,
        condition_number=_condition_number(singular_values),
        column_normalized_condition_number=normalized_condition,
    )


def whiten_basis(basis, fit_mask, *, rank_tolerance=None, context="basis"):
    """Whiten a full-rank fitted basis without changing its function space.

    For ``B_fit = U diag(s) V.T``, the returned transform is
    ``C = V diag(1 / s)`` and the full-grid numerical basis is ``B @ C``.
    Callers must express original-coordinate penalties through
    ``beta = C @ gamma``.
    """
    try:
        basis, mask = _validated_basis_and_mask(basis, fit_mask)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{context}: {exc}") from exc
    diagnostics = basis_rank_diagnostics(
        basis, mask, rank_tolerance=rank_tolerance
    )
    n_fit, n_columns = diagnostics.fit_shape
    if n_fit < n_columns:
        raise BasisRankError(
            f"{context} is underdetermined on fit_mask: "
            f"shape={diagnostics.fit_shape}, numerical_rank="
            f"{diagnostics.numerical_rank}, required_rank={n_columns}, "
            f"rank_tolerance={diagnostics.rank_tolerance:.6g}, "
            f"singular_values={diagnostics.singular_values.tolist()}.",
            diagnostics,
        )
    if diagnostics.numerical_rank < n_columns:
        raise BasisRankError(
            f"{context} is rank deficient on fit_mask: "
            f"shape={diagnostics.fit_shape}, numerical_rank="
            f"{diagnostics.numerical_rank}, required_rank={n_columns}, "
            f"rank_tolerance={diagnostics.rank_tolerance:.6g}, "
            f"singular_values={diagnostics.singular_values.tolist()}.",
            diagnostics,
        )

    fitted = basis[mask]
    _, singular_values, vt = np.linalg.svd(fitted, full_matrices=False)
    transform = vt.T / singular_values[np.newaxis, :]
    whitened_basis = basis @ transform
    gram = whitened_basis[mask].T @ whitened_basis[mask]
    gram_error = float(np.linalg.norm(gram - np.eye(n_columns), ord=2))
    return BasisWhitening(
        whitened_basis=whitened_basis,
        transform=transform,
        diagnostics=diagnostics,
        training_gram_error=gram_error,
    )


def whiten_basis_by_blocks(
    blocks,
    fit_mask,
    *,
    block_names=None,
    rank_tolerance=None,
    joint_rank_tolerance=None,
    context="composed basis",
):
    """Whiten basis blocks individually and then whiten their concatenation.

    Every block must be full column rank on the exact ``fit_mask``, and the
    concatenation of the block-whitened designs must also be full column rank.
    No columns are truncated and there is no unwhitened fallback. If
    ``C_block`` is the block-diagonal matrix of first-stage transforms and
    ``C_joint`` is the second-stage transform, the returned complete transform
    is ``C_block @ C_joint``. Original-coordinate penalties must therefore be
    expressed through ``beta = transform @ gamma``. ``rank_tolerance`` applies
    to the raw-coordinate block audits. The joint stage uses its own
    scale-derived default unless ``joint_rank_tolerance`` is supplied.
    """
    blocks = tuple(blocks)
    if not blocks:
        raise ValueError(f"{context}: blocks must contain at least one basis.")

    validated_blocks = []
    n_rows = None
    for index, block in enumerate(blocks):
        basis = np.asarray(block, dtype=float)
        if basis.ndim != 2:
            raise ValueError(
                f"{context} block {index}: basis must be 2-D; got shape "
                f"{basis.shape}."
            )
        if basis.shape[1] == 0:
            raise ValueError(
                f"{context} block {index}: basis must contain at least one column."
            )
        if n_rows is None:
            n_rows = basis.shape[0]
        elif basis.shape[0] != n_rows:
            raise ValueError(
                f"{context}: all blocks must have the same number of rows; "
                f"block 0 has {n_rows} and block {index} has {basis.shape[0]}."
            )
        validated_blocks.append(basis)

    if block_names is None:
        names = tuple(f"block_{index}" for index in range(len(blocks)))
    else:
        if isinstance(block_names, str):
            raise ValueError(
                f"{context}: block_names must be an iterable of names, not a string."
            )
        names = tuple(block_names)
        if len(names) != len(blocks):
            raise ValueError(
                f"{context}: block_names has length {len(names)}, expected "
                f"{len(blocks)}."
            )
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"{context}: block_names must be non-empty strings.")
        if len(set(names)) != len(names):
            raise ValueError(f"{context}: block_names must be unique.")

    block_whitenings = tuple(
        whiten_basis(
            basis,
            fit_mask,
            rank_tolerance=rank_tolerance,
            context=f"{context} block {name!r}",
        )
        for basis, name in zip(validated_blocks, names, strict=True)
    )
    block_white_basis = np.hstack(
        [whitening.whitened_basis for whitening in block_whitenings]
    )
    joint_whitening = whiten_basis(
        block_white_basis,
        fit_mask,
        rank_tolerance=joint_rank_tolerance,
        context=f"{context} joint concatenation",
    )

    widths = [basis.shape[1] for basis in validated_blocks]
    stops = np.cumsum([0, *widths])
    block_slices = tuple(
        slice(int(start), int(stop))
        for start, stop in zip(stops[:-1], stops[1:], strict=True)
    )
    total_width = int(stops[-1])
    block_transform = np.zeros((total_width, total_width))
    for block_slice, whitening in zip(
        block_slices, block_whitenings, strict=True
    ):
        block_transform[block_slice, block_slice] = whitening.transform

    raw_basis = np.hstack(validated_blocks)
    transform = block_transform @ joint_whitening.transform
    whitened_basis = raw_basis @ transform
    mask = np.asarray(fit_mask)
    gram = whitened_basis[mask].T @ whitened_basis[mask]
    gram_error = float(np.linalg.norm(gram - np.eye(total_width), ord=2))

    for array in (raw_basis, transform, whitened_basis):
        array.setflags(write=False)
    return ComposedBasisWhitening(
        raw_basis=raw_basis,
        whitened_basis=whitened_basis,
        transform=transform,
        block_slices=block_slices,
        block_names=names,
        block_whitenings=block_whitenings,
        joint_whitening=joint_whitening,
        training_gram_error=gram_error,
    )
