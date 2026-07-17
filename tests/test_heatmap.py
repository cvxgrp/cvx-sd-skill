"""Tests for the sub-daily heat-map fold + plot."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # non-interactive backend for headless plot smoke tests
import matplotlib.pyplot as plt  # noqa: E402

from signaldecomp.heatmap import (  # noqa: E402
    fold_from_standardized,
    fold_to_2d,
    plot_heatmap,
    steps_per_day,
)
from signaldecomp.time_axis import standardize_time_axis  # noqa: E402


def test_steps_per_day():
    assert steps_per_day(3600) == 24
    assert steps_per_day(900) == 96
    assert steps_per_day(86400) == 1  # exactly daily is allowed (degenerate)


def test_steps_per_day_rejects_super_daily_and_nondivisor():
    with pytest.raises(ValueError, match="coarser than one day"):
        steps_per_day(100000)
    with pytest.raises(ValueError, match="evenly divide"):
        steps_per_day(3601)


def test_fold_column_is_one_day_order_f():
    # value == hour-of-day, tiled over 3 days -> every column is [0..23].
    y = np.tile(np.arange(24, dtype=float), 3)
    D = fold_to_2d(y, 3600, trim_empty=False)
    assert D.shape == (24, 3)
    for j in range(3):
        assert np.array_equal(D[:, j], np.arange(24))


def test_fold_preserves_nan_gap():
    y = np.tile(np.arange(24, dtype=float), 2)
    y[5] = np.nan
    D = fold_to_2d(y, 3600, trim_empty=False)
    assert np.isnan(D[5, 0])


def test_fold_trims_empty_edge_days():
    y = np.full(24 * 4, np.nan)
    y[24:72] = 1.0  # days 0 and 3 all-NaN
    D = fold_to_2d(y, 3600, trim_empty=True)
    assert D.shape == (24, 2)
    # without trimming, all four days are kept
    D_all = fold_to_2d(y, 3600, trim_empty=False)
    assert D_all.shape == (24, 4)


def test_fold_rejects_non_whole_days():
    with pytest.raises(ValueError, match="whole number of days"):
        fold_to_2d(np.arange(25.0), 3600)


def test_fold_rejects_super_daily():
    with pytest.raises(ValueError, match="coarser than one day"):
        fold_to_2d(np.arange(24.0), 100000)


def test_fold_from_standardized_roundtrip():
    idx = pd.date_range("2021-06-01", periods=96 * 3, freq="15min")
    std = standardize_time_axis(pd.Series(np.sin(np.arange(96 * 3) / 10.0), index=idx))
    D = fold_from_standardized(std)
    assert D.shape[0] == 96
    assert D.shape[1] == 3


def test_plot_heatmap_nonnegative_smoke():
    D = np.abs(np.random.default_rng(0).standard_normal((24, 10)))
    fig = plot_heatmap(D, signed=False, title="nonneg", cbar_label="power")
    assert fig is not None
    assert len(fig.axes) >= 1  # heatmap + colorbar
    plt.close(fig)


def test_plot_heatmap_signed_smoke():
    D = np.random.default_rng(1).standard_normal((24, 10))
    D[3, 3] = np.nan  # a gap to exercise set_bad
    fig = plot_heatmap(D, signed=True, cbar_label="residual")
    assert fig is not None
    plt.close(fig)


def test_plot_heatmap_with_day_axis_smoke():
    D = np.random.default_rng(2).standard_normal((24, 90))
    day_axis = pd.date_range("2021-01-01", periods=90, freq="D")
    fig = plot_heatmap(D, day_axis=day_axis, year_lines=True)
    assert fig is not None
    plt.close(fig)
