"""Tests for the domain-agnostic log/multiplicative transform layer."""

import numpy as np
import pandas as pd
import pytest

from signaldecomp import linear_trend, make_problem, multiperiodic, solve
from signaldecomp.transform import prepare_input, recover_components, recover_frame


def _positive_signal(seed=0, T=400):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    y = 2.0 + 0.001 * t + 0.5 * np.sin(2 * np.pi * t / 50) + 0.1 * rng.standard_normal(T)
    y[100:110] = np.nan  # a gap
    return y


def _solve(y):
    built = make_problem(
        y,
        components=[
            linear_trend(role="trend"),
            multiperiodic(50.0, num_harmonics=3, weight=1e-2, role="seas"),
        ],
    )
    return solve(built)


def test_prepare_input_passthrough_without_log():
    y = _positive_signal()
    out = prepare_input(y, log_transform=False)
    obs = ~np.isnan(y)
    assert np.allclose(out[obs], y[obs])
    out[0] = -999.0
    assert y[0] != -999.0


def test_prepare_input_log_masks_nonpositive_and_takes_log():
    y = np.array([1.0, 2.0, 0.0, -3.0, 4.0, np.nan])
    out = prepare_input(y, log_transform=True)
    assert np.isnan(out[2]) and np.isnan(out[3]) and np.isnan(out[5])
    assert np.allclose(out[[0, 1, 4]], np.log([1.0, 2.0, 4.0]))


def test_recover_additive_reconstruction_plus_residual_equals_y():
    y = _positive_signal()
    ya = prepare_input(y, log_transform=False)
    out = _solve(ya)
    df = recover_components(out, log_transform=False)
    assert set(df.columns) == {"trend", "seas", "residual", "reconstruction"}
    obs = ~np.isnan(y)
    assert np.allclose((df["reconstruction"] + df["residual"])[obs], y[obs], atol=1e-6)
    assert np.allclose(df["reconstruction"], df["trend"] + df["seas"])


def test_recover_multiplicative_reconstruction_times_residual_equals_y():
    y = _positive_signal()
    yl = prepare_input(y, log_transform=True)
    out = _solve(yl)
    df = recover_components(out, log_transform=True)
    assert set(df.columns) == {"trend", "seas", "residual", "reconstruction"}
    obs = ~np.isnan(y)
    assert np.allclose((df["reconstruction"] * df["residual"])[obs], y[obs], atol=1e-6)
    assert np.allclose(df["reconstruction"], df["trend"] * df["seas"])
    assert (df["trend"] > 0).all() and (df["seas"] > 0).all()


def test_recover_respects_index():
    y = _positive_signal(T=300)
    out = _solve(prepare_input(y))
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    df = recover_components(out, index=idx)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 300
    with pytest.raises(ValueError, match="index length"):
        recover_components(out, index=pd.RangeIndex(299))


def test_recover_rejects_non_solved_output():
    with pytest.raises(ValueError, match="no 'residual'"):
        recover_components({"values": {"trend": np.zeros(10)}})


def test_recover_frame_additive_matches_components():
    y = _positive_signal()
    out = _solve(prepare_input(y, log_transform=False))
    df = recover_frame(out, log_transform=False, y=y)
    assert set(df.columns) == {"trend", "seas", "residual", "reconstruction", "y"}
    # additive: reconstruction is the sum; residual referenced to 0
    assert np.allclose(df["reconstruction"], df["trend"] + df["seas"])
    obs = ~np.isnan(y)
    assert np.allclose((df["reconstruction"] + df["residual"])[obs], y[obs], atol=1e-6)
    assert np.allclose(df["y"], y, equal_nan=True)


def test_recover_frame_multiplicative_residual_referenced_to_one():
    y = _positive_signal()
    out = _solve(prepare_input(y, log_transform=True))
    df = recover_frame(out, log_transform=True, y=y)
    assert set(df.columns) == {"trend", "seas", "residual", "reconstruction", "y"}
    obs = ~np.isnan(y)
    # multiplicative residual is a FACTOR referenced to 1 (not 0)
    resid = df["residual"].to_numpy()[obs]
    assert 0.5 < resid.mean() < 1.5
    assert (resid > 0).all()
    # reconstruction (product) * residual factor == y
    assert np.allclose((df["reconstruction"] * df["residual"])[obs], y[obs], atol=1e-6)


def test_recover_frame_y_length_mismatch_raises():
    out = _solve(prepare_input(_positive_signal(T=300)))
    with pytest.raises(ValueError, match="y length"):
        recover_frame(out, y=np.zeros(299))
