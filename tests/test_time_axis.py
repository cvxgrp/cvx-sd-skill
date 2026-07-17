"""Tests for the time-axis standardization layer."""

import numpy as np
import pandas as pd
import pytest

from signaldecomp.time_axis import (
    derive_delta,
    scan_rates,
    standardize_time_axis,
)


def test_standardize_jittered_hourly_with_gap_and_duplicate():
    rng = np.random.default_rng(0)
    base = pd.date_range("2021-03-01 00:00", periods=48, freq="h")
    idx = base + pd.to_timedelta(rng.integers(-20, 20, size=48), unit="s")
    s = pd.Series(np.sin(2 * np.pi * np.arange(48) / 24), index=idx)
    s = s.drop(s.index[10:15])          # a ~5h gap
    s = pd.concat([s, s.iloc[[0]]])     # a duplicate timestamp
    out = standardize_time_axis(s)
    # empirical modal gap (~3599 from jitter) snaps to the standard hourly rate
    assert out["delta"] == 3600.0
    assert out["freq"] == "h"
    # grid anchored at midnight of the first day
    assert out["index"][0] == pd.Timestamp("2021-03-01 00:00:00")
    # the gap shows up as NaN (mask feed)
    assert 4 <= int(np.isnan(out["y"]).sum()) <= 6
    # jitter does not fragment the single rate
    assert out["scan_rates"]["multiple_rates"] is False
    assert len(out["y"]) == len(out["index"])


def test_derive_delta_multiday_does_not_wrap():
    # 2-day gaps must read as 172800s, not wrap at 24h (the .seconds footgun).
    idx = pd.DatetimeIndex(["2021-01-01", "2021-01-03", "2021-01-05"])
    assert derive_delta(idx) == 172800.0


def test_derive_delta_needs_two_timestamps():
    with pytest.raises(ValueError, match="two timestamps"):
        derive_delta(pd.DatetimeIndex(["2021-01-01"]))


def test_scan_rates_detects_genuine_two_rates_and_transition():
    rng = np.random.default_rng(1)
    a = pd.date_range("2021-01-01", periods=100, freq="300s") + pd.to_timedelta(
        rng.integers(-3, 3, 100), "s"
    )
    b = pd.date_range(a[-1] + pd.Timedelta("60s"), periods=100, freq="60s") + (
        pd.to_timedelta(rng.integers(-1, 1, 100), "s")
    )
    sr = scan_rates(a.append(b))
    assert sr["multiple_rates"] is True
    # two rates recovered near their nominal values (jitter -> not exact)
    rates = sorted(sr["rates_seconds"])
    assert len(rates) == 2
    assert abs(rates[0] - 60) <= 3
    assert abs(rates[1] - 300) <= 5
    assert len(sr["transitions"]) >= 1


def test_scan_rates_single_rate_no_transitions():
    idx = pd.date_range("2021-01-01", periods=200, freq="900s")
    sr = scan_rates(idx)
    assert sr["multiple_rates"] is False
    assert sr["modal_seconds"] == 900.0
    assert sr["transitions"] == []


def test_standardize_freq_is_authoritative():
    # An asserted freq overrides the empirical rate: 37s data forced to 1-min.
    idx = pd.date_range("2021-01-01 00:00", periods=100, freq="37s")
    s = pd.Series(np.arange(100.0), index=idx)
    out = standardize_time_axis(s, freq="min")
    assert out["delta"] == 60.0
    assert out["freq"] == "min"


def test_standardize_nonstandard_rate_not_snapped():
    # 47s is > 1% from every standard rate: keep empirical delta, freq None.
    idx = pd.date_range("2021-01-01", periods=200, freq="47s")
    out = standardize_time_axis(pd.Series(np.arange(200.0), index=idx))
    assert out["freq"] is None
    assert out["delta"] == 47.0


def test_standardize_refuses_coarser_than_daily():
    idx = pd.date_range("2021-01-01", periods=12, freq="30D")
    with pytest.raises(ValueError, match="coarser than daily"):
        standardize_time_axis(pd.Series(np.arange(12.0), index=idx))


def test_standardize_freq_daily_authoritative():
    # Regression: freq="D" is a calendar offset that pd.Timedelta refuses; the
    # authoritative path must resolve it to 86400 via the fixed-seconds table.
    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    out = standardize_time_axis(pd.Series(np.arange(400.0), index=idx), freq="D")
    assert out["delta"] == 86400.0
    assert out["freq"] == "D"
    assert out["index"][0] == pd.Timestamp("2020-01-01 00:00:00")


def test_standardize_freq_rejects_calendar_offset():
    # A coarser calendar offset (weekly) is out of scope for the fixed grid.
    idx = pd.date_range("2020-01-01", periods=52, freq="7D")
    with pytest.raises(ValueError, match="fixed-duration rate"):
        standardize_time_axis(pd.Series(np.arange(52.0), index=idx), freq="W")


def test_standardize_dataframe_requires_value_column_when_ambiguous():
    idx = pd.date_range("2021-01-01", periods=10, freq="h")
    df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)}, index=idx)
    with pytest.raises(ValueError, match="value_column is required"):
        standardize_time_axis(df)
    out = standardize_time_axis(df, value_column="b")
    assert len(out["y"]) == len(out["index"])


def test_standardize_rejects_non_pandas():
    with pytest.raises(TypeError, match="Series or DataFrame"):
        standardize_time_axis(np.arange(10))
