"""Tests for grouped/block trend and sparse components."""

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from signaldecomp import (
    exog_linear,
    grouped_sparse,
    grouped_trend,
    make_group_basis,
    make_problem,
    smooth_trend,
    solve,
)


def test_label_basis_uses_first_appearance_and_marks_missing_rows():
    groups = np.array(["b", "b", "a", None, "c", "a"], dtype=object)
    basis = make_group_basis(groups)
    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    assert basis.group_labels == ("b", "a", "c")
    assert np.array_equal(basis.design, expected)
    assert np.array_equal(basis.valid_mask, [True, True, True, False, True, True])
    assert not basis.design.flags.writeable
    assert not basis.valid_mask.flags.writeable


def test_pandas_nat_is_an_unavailable_group_row():
    groups = pd.DatetimeIndex(
        ["2025-01-01", "2025-01-01", pd.NaT, "2025-01-02"]
    )
    basis = make_group_basis(groups)
    assert basis.group_labels == (
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-02"),
    )
    assert np.array_equal(basis.valid_mask, [True, True, False, True])
    assert np.array_equal(basis.design[2], [0.0, 0.0])

    built = make_problem(
        np.arange(4.0),
        [grouped_trend(groups, weight=1.0, role="level")],
    )
    assert np.array_equal(built["fit_mask"], [True, True, False, True])


def test_all_pandas_nat_labels_are_rejected_as_missing():
    groups = pd.DatetimeIndex([pd.NaT, pd.NaT])
    with pytest.raises(ValueError, match="no non-missing labels"):
        make_group_basis(groups)


def test_explicit_group_order_controls_columns_and_matches_mapping_path():
    groups = np.array(["b", "a", "c", "a", "b"], dtype=object)
    order = ("a", "b", "c")
    labels = make_group_basis(groups, group_order=order)
    mapping = make_group_basis(
        mapping=labels.design,
        group_order=order,
    )
    assert labels.group_labels == order
    assert mapping.group_labels == order
    assert labels.source == "labels"
    assert mapping.source == "mapping"
    assert np.array_equal(labels.design, mapping.design)
    assert np.array_equal(labels.valid_mask, mapping.valid_mask)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "exactly one"),
        ({"groups": [0, 1], "mapping": np.eye(2)}, "exactly one"),
        ({"groups": [[0, 1]]}, "1-D"),
        ({"groups": [None, np.nan]}, "non-missing"),
        ({"groups": [[1], [2]]}, "1-D"),
        ({"mapping": np.array([1.0, 0.0])}, "2-D"),
        ({"mapping": np.array([[0.5, 0.5]])}, "one-hot"),
        ({"mapping": np.array([[1.0, 1.0]])}, "exactly one"),
        ({"mapping": np.array([[1.0, 0.0], [1.0, 0.0]])}, "empty"),
        ({"mapping": np.array([[np.nan]])}, "finite"),
    ],
)
def test_group_basis_rejects_malformed_representations(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        make_group_basis(**kwargs)


def test_group_order_must_match_observed_labels_exactly():
    groups = ["a", "b", "a"]
    with pytest.raises(ValueError, match="omits"):
        make_group_basis(groups, group_order=["a"])
    with pytest.raises(ValueError, match="no samples"):
        make_group_basis(groups, group_order=["a", "b", "c"])
    with pytest.raises(ValueError, match="unique"):
        make_group_basis(groups, group_order=["a", "b", "a"])
    with pytest.raises(ValueError, match="expected 2"):
        make_group_basis(mapping=np.eye(2), group_order=["a"])


def test_label_and_mapping_components_produce_identical_solutions():
    rng = np.random.default_rng(90)
    groups = np.repeat(np.arange(12), 5)
    mapping = make_group_basis(groups).design
    levels = np.sin(np.arange(12) / 3)
    y = levels[groups] + 0.03 * rng.standard_normal(groups.size)
    labels_out = solve(
        make_problem(y, [grouped_trend(groups, weight=0.1, role="level")])
    )
    mapping_out = solve(
        make_problem(
            y,
            [grouped_trend(mapping=mapping, weight=0.1, role="level")],
        )
    )
    assert np.allclose(labels_out["values"]["level"], mapping_out["values"]["level"])
    assert np.allclose(
        labels_out["values"]["level_group_values"],
        mapping_out["values"]["level_group_values"],
    )


def test_grouped_trend_is_constant_within_groups_and_smoothing_reduces_drift():
    rng = np.random.default_rng(91)
    groups = np.repeat(np.arange(10), 6)
    truth = np.array([0.0, 1.0, -0.5, 1.5, 0.2, 1.8, 0.1, 1.4, 0.4, 1.0])
    y = truth[groups] + 0.05 * rng.standard_normal(groups.size)
    loose = solve(
        make_problem(y, [grouped_trend(groups, weight=0.0, role="level")])
    )
    stiff = solve(
        make_problem(y, [grouped_trend(groups, weight=1e3, role="level")])
    )
    for group in np.unique(groups):
        fitted = loose["values"]["level"][groups == group]
        assert np.max(fitted) - np.min(fitted) < 1e-12
    assert np.linalg.norm(
        np.diff(stiff["values"]["level_group_values"])
    ) < np.linalg.norm(np.diff(loose["values"]["level_group_values"]))


@pytest.mark.parametrize(
    ("monotonic", "sign"),
    [("increasing", 1.0), ("decreasing", -1.0)],
)
def test_grouped_trend_enforces_monotonicity_and_baseline(monotonic, sign):
    rng = np.random.default_rng(92)
    groups = np.repeat(np.arange(9), 5)
    y = sign * np.arange(9)[groups] + 0.2 * rng.standard_normal(groups.size)
    out = solve(
        make_problem(
            y,
            [
                grouped_trend(
                    groups,
                    weight=0.1,
                    monotonic=monotonic,
                    baseline=0.0,
                    role="level",
                )
            ],
        )
    )
    values = out["values"]["level_group_values"]
    assert abs(values[0]) < 1e-7
    assert np.all(sign * np.diff(values) >= -1e-7)


def test_grouped_sparse_recovers_rare_group_corrections():
    rng = np.random.default_rng(93)
    groups = np.repeat(np.arange(20), 8)
    truth = np.zeros(20)
    truth[[5, 14]] = [3.0, -2.0]
    y = truth[groups] + 0.04 * rng.standard_normal(groups.size)
    out = solve(
        make_problem(
            y,
            [grouped_sparse(groups, weight=0.4, role="events")],
        )
    )
    group_values = out["values"]["events_group_values"]
    assert set(np.flatnonzero(np.abs(group_values) > 0.5)) == {5, 14}
    assert np.allclose(out["values"]["events"], group_values[groups])


def test_missing_group_rows_restrict_fit_mask_with_finite_expression():
    groups = np.array(["a", "a", None, "b", "b"], dtype=object)
    component = grouped_trend(groups, weight=1.0, role="level")
    built = make_problem(np.arange(5.0), [component])
    assert np.array_equal(built["component_mask"], [True, True, False, True, True])
    out = solve(built)
    assert np.all(np.isfinite(out["values"]["level"]))
    assert out["values"]["level"][2] == pytest.approx(0.0)


def test_grouped_metadata_and_aux_preserve_declared_order():
    groups = ["late", "early", "late", "early"]
    component = grouped_trend(
        groups,
        group_order=("early", "late"),
        role="block",
    )
    out = solve(make_problem(np.arange(4.0), [component]))
    basis = out["component_metadata"]["block"]["group_basis"]
    assert basis.group_labels == ("early", "late")
    assert out["values"]["block_group_values"].shape == (2,)


def test_grouped_components_compose_with_sample_and_exogenous_components():
    rng = np.random.default_rng(94)
    T = 180
    groups = np.repeat(np.arange(18), 10)
    z = rng.standard_normal(T)
    y = 0.8 * z + 0.05 * groups + 0.02 * rng.standard_normal(T)
    built = make_problem(
        y,
        [
            exog_linear(z, role="driver"),
            grouped_trend(groups, weight=0.1, role="block"),
            smooth_trend(1e3, role="fine_trend"),
        ],
    )
    assert built["problem"].is_dcp()
    out = solve(built)
    assert abs(out["values"]["driver_beta"] - 0.8) < 0.1


@pytest.mark.parametrize(
    "component",
    [
        grouped_trend(np.repeat(np.arange(4), 3), weight=1.0),
        grouped_trend(
            np.repeat(np.arange(4), 3),
            monotonic="increasing",
            baseline=0.0,
        ),
        grouped_sparse(np.repeat(np.arange(4), 3), weight=1.0),
    ],
)
def test_grouped_component_expressions_are_dcp(component):
    expr, loss, constraints = component.build(12)
    assert expr.is_affine()
    if isinstance(loss, cp.Expression):
        assert loss.is_convex() and loss.is_dcp()
    assert all(constraint.is_dcp() for constraint in constraints)


def test_grouped_penalties_use_declared_group_coordinates():
    groups = np.repeat(np.arange(3), 2)
    trend = grouped_trend(groups, weight=2.0, role="trend")
    _, trend_loss, _ = trend.build(groups.size)
    trend.aux["trend_group_values"].value = np.array([1.0, 3.0, 2.0])
    assert trend_loss.value == pytest.approx(2.0 * (2.0**2 + (-1.0) ** 2))

    events = grouped_sparse(groups, weight=6.0, role="events")
    _, sparse_loss, _ = events.build(groups.size)
    events.aux["events_group_values"].value = np.array([1.0, -2.0, 0.5])
    assert sparse_loss.value == pytest.approx(6.0 / 3 * 3.5)


def test_grouped_builder_argument_validation():
    groups = [0, 0, 1, 1]
    with pytest.raises(ValueError, match="weight"):
        grouped_trend(groups, weight=-1.0)
    with pytest.raises(ValueError, match="weight"):
        grouped_sparse(groups, weight=np.inf)
    with pytest.raises(ValueError, match="monotonic"):
        grouped_trend(groups, monotonic="up")
    with pytest.raises(ValueError, match="baseline"):
        grouped_trend(groups, baseline=np.nan)
