# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
"""Sample-to-group design utilities for grouped signal components.

The group-level trend and sparse formulations are adapted from TSGAM's
period-grouped components while leaving timestamp inference and estimator
lifecycle behavior outside the importable component API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupBasis:
    """Immutable one-hot sample-to-group design and its declared ordering."""

    design: np.ndarray
    valid_mask: np.ndarray
    group_labels: tuple[object, ...]
    source: str

    @property
    def n_groups(self):
        """Number of ordered groups (the design width)."""
        return self.design.shape[1]


def _is_missing_label(value):
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    # Support pandas.NA and pandas.NaT without importing pandas in this pure
    # design utility. Both are stable public scalar types in pandas.
    return type(value).__name__ in ("NAType", "NaTType")


def _ordered_unique(labels):
    ordered = []
    positions = {}
    for label in labels:
        try:
            if label not in positions:
                positions[label] = len(ordered)
                ordered.append(label)
        except TypeError as exc:
            raise TypeError(f"group labels must be hashable; got {label!r}.") from exc
    return tuple(ordered), positions


def _validated_group_order(group_order):
    try:
        order = tuple(group_order)
    except TypeError as exc:
        raise TypeError("group_order must be an iterable of unique labels.") from exc
    if not order:
        raise ValueError("group_order must contain at least one label.")
    if any(_is_missing_label(label) for label in order):
        raise ValueError("group_order cannot contain missing labels.")
    unique, positions = _ordered_unique(order)
    if len(unique) != len(order):
        raise ValueError("group_order labels must be unique.")
    return order, positions


def _from_labels(groups, group_order):
    labels = np.asarray(groups, dtype=object)
    if labels.ndim != 1:
        raise ValueError(f"groups must be 1-D; got shape {labels.shape}.")
    if labels.size == 0:
        raise ValueError("groups must contain at least one sample.")
    valid_mask = np.array(
        [not _is_missing_label(label) for label in labels], dtype=bool
    )
    if not valid_mask.any():
        raise ValueError("groups contains no non-missing labels.")
    observed_order, observed_positions = _ordered_unique(labels[valid_mask])

    if group_order is None:
        resolved_order = observed_order
        positions = observed_positions
    else:
        resolved_order, positions = _validated_group_order(group_order)
        observed_set = set(observed_order)
        declared_set = set(resolved_order)
        missing = observed_set - declared_set
        empty = declared_set - observed_set
        if missing:
            raise ValueError(
                f"group_order omits observed labels: {sorted(missing, key=repr)!r}."
            )
        if empty:
            raise ValueError(
                f"group_order contains labels with no samples: "
                f"{sorted(empty, key=repr)!r}."
            )

    design = np.zeros((labels.size, len(resolved_order)), dtype=float)
    for sample, label in enumerate(labels):
        if valid_mask[sample]:
            design[sample, positions[label]] = 1.0
    return design, valid_mask, resolved_order


def _from_mapping(mapping, group_order):
    design = np.asarray(mapping, dtype=float)
    if design.ndim != 2:
        raise ValueError(f"mapping must be 2-D; got shape {design.shape}.")
    T, n_groups = design.shape
    if T == 0 or n_groups == 0:
        raise ValueError("mapping must contain at least one sample and one group.")
    if not np.all(np.isfinite(design)):
        raise ValueError("mapping must contain only finite values.")
    if not np.all((design == 0.0) | (design == 1.0)):
        raise ValueError("mapping must be one-hot with entries equal to 0 or 1.")
    row_sums = design.sum(axis=1)
    if np.any((row_sums != 0.0) & (row_sums != 1.0)):
        raise ValueError("each mapping row must contain exactly one 1 or be all zero.")
    valid_mask = row_sums == 1.0
    if not valid_mask.any():
        raise ValueError("mapping contains no assigned samples.")
    empty_columns = np.where(design[valid_mask].sum(axis=0) == 0.0)[0]
    if empty_columns.size:
        raise ValueError(
            f"mapping contains empty group columns: {empty_columns.tolist()}."
        )

    if group_order is None:
        resolved_order = tuple(range(n_groups))
    else:
        resolved_order, _ = _validated_group_order(group_order)
        if len(resolved_order) != n_groups:
            raise ValueError(
                f"group_order has {len(resolved_order)} labels, expected {n_groups}."
            )
    return design.copy(), valid_mask, resolved_order


def make_group_basis(groups=None, *, mapping=None, group_order=None):
    """Build an immutable one-hot sample-to-group design.

    Pass exactly one of a one-dimensional label vector ``groups`` or an
    explicit one-hot ``mapping`` matrix. Label columns follow ``group_order``
    when supplied and first appearance otherwise. Missing labels and all-zero
    mapping rows are represented by finite zero rows and excluded through the
    returned ``valid_mask``.
    """
    if (groups is None) == (mapping is None):
        raise ValueError("pass exactly one of groups or mapping.")
    if groups is not None:
        design, valid_mask, labels = _from_labels(groups, group_order)
        source = "labels"
    else:
        design, valid_mask, labels = _from_mapping(mapping, group_order)
        source = "mapping"

    design.setflags(write=False)
    valid_mask.setflags(write=False)
    return GroupBasis(
        design=design,
        valid_mask=valid_mask,
        group_labels=labels,
        source=source,
    )
