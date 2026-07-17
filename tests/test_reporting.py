"""Tests for the reporting / pandas round-trip + plotting layer."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless backend for plot smoke tests
import matplotlib.pyplot as plt  # noqa: E402

from signaldecomp import (  # noqa: E402
    linear_trend,
    make_problem,
    multiperiodic,
    solve,
    standardize_time_axis,
)
from signaldecomp.reporting import (  # noqa: E402
    components_to_frame,
    plot_decomposition,
    plot_stability,
)
from signaldecomp.validation import expanding_window_stability  # noqa: E402


def _solved_with_gap():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2021-01-01", periods=96 * 20, freq="15min")
    t = np.arange(96 * 20)
    y = 1.0 + 0.001 * t + 0.5 * np.sin(2 * np.pi * t / 96) + 0.05 * rng.standard_normal(t.size)
    y[96 * 5 : 96 * 6] = np.nan  # a fully missing day
    std = standardize_time_axis(pd.Series(y, index=idx))
    built = make_problem(
        std["y"],
        components=[
            multiperiodic(96, num_harmonics=4, weight=1e-2, role="daily"),
            linear_trend(role="trend"),
        ],
    )
    return solve(built), std


def test_frame_columns_exclude_scalar_aux():
    out, std = _solved_with_gap()
    df = components_to_frame(out, index=std["index"], y=std["y"])
    assert set(df.columns) == {"daily", "trend", "residual", "reconstruction", "y"}
    # scalar / non-length-T aux are excluded
    assert "trend_b" not in df.columns
    assert "daily_theta" not in df.columns


def test_frame_index_alignment_and_default_range_index():
    out, std = _solved_with_gap()
    df = components_to_frame(out, index=std["index"])
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == len(std["index"])
    # no index -> RangeIndex
    assert isinstance(components_to_frame(out).index, pd.RangeIndex)


def test_reconstruction_is_sum_of_structural_and_matches_observed():
    out, std = _solved_with_gap()
    df = components_to_frame(out, index=std["index"], y=std["y"])
    assert np.allclose(df["reconstruction"], df["daily"] + df["trend"])
    obs = ~np.isnan(std["y"])
    assert np.allclose(
        (df["reconstruction"] + df["residual"])[obs], std["y"][obs], atol=1e-6
    )


def test_imputed_dense_by_default_and_y_preserves_gap():
    out, std = _solved_with_gap()
    df = components_to_frame(out, index=std["index"], y=std["y"])
    # components imputed over the gap (dense)
    assert not df["daily"].isna().any()
    assert not df["reconstruction"].isna().any()
    # y column keeps the 96 missing entries
    assert int(df["y"].isna().sum()) == 96


def test_mask_remasks_components_but_not_y():
    out, std = _solved_with_gap()
    obs = ~np.isnan(std["y"])
    df = components_to_frame(out, index=std["index"], y=std["y"], mask=obs)
    assert int(df["daily"].isna().sum()) == 96
    assert int(df["residual"].isna().sum()) == 96
    assert int(df["reconstruction"].isna().sum()) == 96
    assert int(df["y"].isna().sum()) == 96  # y untouched by mask


def test_length_mismatches_raise():
    out, std = _solved_with_gap()
    T = out["values"]["residual"].shape[0]
    with pytest.raises(ValueError, match="index length"):
        components_to_frame(out, index=pd.RangeIndex(T - 1))
    with pytest.raises(ValueError, match="y shape"):
        components_to_frame(out, y=np.zeros(T - 1))
    with pytest.raises(ValueError, match="mask shape"):
        components_to_frame(out, mask=np.ones(T - 1, dtype=bool))


def test_rejects_non_solved_output():
    with pytest.raises(ValueError, match="no 'residual'"):
        components_to_frame({"values": {"trend": np.zeros(10)}})


# --- plotting smoke tests (Agg backend) -----------------------------------


def _solved_two_role():
    rng = np.random.default_rng(0)
    T = 400
    t = np.arange(T)
    y = 1.0 + 0.01 * t + 0.5 * np.sin(2 * np.pi * t / 50) + 0.1 * rng.standard_normal(T)
    build_fn = lambda sig: make_problem(
        sig,
        components=[
            linear_trend(role="trend"),
            multiperiodic(50.0, num_harmonics=3, weight=1e-2, role="seas"),
        ],
    )
    return solve(build_fn(y)), y, build_fn


def test_plot_decomposition_from_out_smoke():
    out, y, _ = _solved_two_role()
    fig = plot_decomposition(out, y=y)
    # top (signal+fit) + 2 roles + residual = 4 panels
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_decomposition_from_df_path_smoke():
    out, y, _ = _solved_two_role()
    df = components_to_frame(out, y=y)
    fig = plot_decomposition(None, df=df)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_stability_snapshot_only_smoke():
    _, y, build_fn = _solved_two_role()
    stab = expanding_window_stability(y, build_fn, min_window=100, step=75)
    fig = plot_stability(stab)  # default role = first = "trend"
    # snapshot spaghetti (+ colorbar) + rmsd = 3 axes, no scalar panels
    assert len(fig.axes) == 3
    plt.close(fig)
    # explicit role selection
    fig2 = plot_stability(stab, role="seas")
    assert fig2 is not None
    plt.close(fig2)


def test_plot_stability_with_scalar_smoke():
    _, y, build_fn = _solved_two_role()
    stab = expanding_window_stability(
        y, build_fn, min_window=100, step=75, tol=1e-3,
        extractor=lambda o: o["values"]["trend_b"],
    )
    fig = plot_stability(stab, role="trend")
    # spaghetti(+cbar) + rmsd + history + delta = 5 axes
    assert len(fig.axes) == 5
    plt.close(fig)


def test_plot_stability_bad_role_raises():
    _, y, build_fn = _solved_two_role()
    stab = expanding_window_stability(y, build_fn, min_window=100, step=75)
    with pytest.raises(ValueError, match="not in snapshots"):
        plot_stability(stab, role="nonexistent")
