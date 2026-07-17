"""Tests for the domain-agnostic validation layer."""

import numpy as np
import pytest

from signaldecomp import linear_trend, make_problem, smooth_trend, solve
from signaldecomp import multiperiodic
from signaldecomp.validation import (
    bootstrap_ci,
    expanding_window_stability,
    holdout_select,
    valid_endpoints,
)


def _linear_signal(seed=0, T=400, slope=0.01, gap=(150, 170)):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    y = 1.0 + slope * t + 0.1 * rng.standard_normal(T)
    y[gap[0] : gap[1]] = np.nan
    return y


def _slope_build_fn(sig):
    return make_problem(sig, components=[linear_trend(role="trend")])


def _slope(out):
    return out["values"]["trend_b"]


def test_bootstrap_ci_brackets_point_estimate():
    # A residual-bootstrap CI is centred on the point estimate (not the unknown
    # truth): it must bracket the point estimate, and be ordered lower<=upper.
    y = _linear_signal()
    point = solve(_slope_build_fn(y))["values"]["trend_b"]
    ci = bootstrap_ci(y, _slope_build_fn, _slope, block_size=20,
                      n_resamples=200, random_state=1)
    lo, hi = ci["value"]
    assert lo <= point <= hi
    assert lo < hi


def test_bootstrap_ci_width_matches_analytic_slope_se():
    # For an unregularised linear trend the fit is OLS, so the bootstrap slope
    # CI half-width should be within a small factor of the analytic slope SE.
    y = _linear_signal()
    m = ~np.isnan(y)
    t = np.arange(y.shape[0])[m]
    resid_sd = np.std(solve(_slope_build_fn(y))["values"]["residual"][m])
    analytic_se = resid_sd * np.sqrt(1.0 / np.sum((t - t.mean()) ** 2))
    ci = bootstrap_ci(y, _slope_build_fn, _slope, block_size=20,
                      n_resamples=300, random_state=2)
    lo, hi = ci["value"]
    half_width = (hi - lo) / 2
    # 1 sigma CI half-width ~ 1 SE; allow a generous factor for block/resample.
    assert 0.25 * analytic_se < half_width < 4 * analytic_se


def test_bootstrap_ci_confidence_levels_nest():
    y = _linear_signal()
    ci68 = bootstrap_ci(y, _slope_build_fn, _slope, block_size=20,
                        n_resamples=300, confidence_level=68.2, random_state=3)
    ci95 = bootstrap_ci(y, _slope_build_fn, _slope, block_size=20,
                        n_resamples=300, confidence_level=95.0, random_state=3)
    assert ci95["value"][0] <= ci68["value"][0]
    assert ci95["value"][1] >= ci68["value"][1]


def test_bootstrap_ci_dict_extractor_multi_key():
    # A dict-returning extractor yields a CI per key.
    y = _linear_signal()

    def multi(out):
        return {"slope": out["values"]["trend_b"], "intercept": out["values"]["trend_a"]}

    ci = bootstrap_ci(y, _slope_build_fn, multi, block_size=20,
                      n_resamples=150, random_state=4)
    assert set(ci.keys()) == {"slope", "intercept"}
    for key in ci:
        assert ci[key][0] <= ci[key][1]


def test_bootstrap_ci_preserves_mask():
    # The gap must remain unobserved across resamples: the routine should not
    # error and the point solve's reconstruction covers the full length.
    y = _linear_signal(gap=(100, 140))
    ci = bootstrap_ci(y, _slope_build_fn, _slope, block_size=15,
                      n_resamples=100, random_state=5)
    assert "value" in ci


def test_bootstrap_ci_requires_block_size_positive():
    y = _linear_signal()
    with pytest.raises(ValueError, match="block_size"):
        bootstrap_ci(y, _slope_build_fn, _slope, block_size=0)


# --- expanding-window stability -------------------------------------------


def test_valid_endpoints_snap_to_observed_and_bounds():
    y = np.arange(100, dtype=float)
    y[45:55] = np.nan
    eps = valid_endpoints(y, min_window=20, step=10)
    # every endpoint ends on an observed sample
    assert all(not np.isnan(y[e - 1]) for e in eps)
    assert eps.min() >= 20
    assert eps[-1] == 100  # full length always included
    # the nominal 50 snaps back around the 45..55 gap to 45
    assert 45 in eps


def test_valid_endpoints_no_observed_raises():
    with pytest.raises(ValueError, match="observed"):
        valid_endpoints(np.full(50, np.nan), min_window=10, step=10)


def test_expanding_window_tracks_and_converges():
    rng = np.random.default_rng(0)
    T = 500
    t = np.arange(T)
    y = 1.0 + 0.01 * t + 0.1 * rng.standard_normal(T)
    build_fn = lambda sig: make_problem(sig, components=[linear_trend(role="trend")])
    res = expanding_window_stability(
        y, build_fn, lambda o: o["values"]["trend_b"],
        min_window=100, step=50, tol=1e-3,
    )
    assert res["windows"][-1] == T
    hist = res["history"]["value"]
    assert hist.shape == res["windows"].shape
    assert np.all(np.isfinite(hist))  # all windows solved
    # slope stays near truth across windows
    assert np.all(np.abs(hist - 0.01) < 5e-3)
    # a loose tolerance converges no later than a strict one
    res_loose = expanding_window_stability(
        y, build_fn, lambda o: o["values"]["trend_b"],
        min_window=100, step=50, tol=1e-2,
    )
    ca_strict = res["converged_at"]["value"]
    ca_loose = res_loose["converged_at"]["value"]
    assert ca_loose is not None and ca_strict is not None
    assert ca_loose <= ca_strict


def test_expanding_window_dict_extractor():
    rng = np.random.default_rng(1)
    T = 400
    t = np.arange(T)
    y = 2.0 + 0.005 * t + 0.05 * rng.standard_normal(T)
    build_fn = lambda sig: make_problem(sig, components=[linear_trend(role="trend")])

    def multi(out):
        return {"slope": out["values"]["trend_b"], "intercept": out["values"]["trend_a"]}

    res = expanding_window_stability(y, build_fn, multi, min_window=100, step=50, tol=1e-3)
    assert set(res["history"].keys()) == {"slope", "intercept"}
    assert set(res["converged_at"].keys()) == {"slope", "intercept"}


# --- holdout model selection ----------------------------------------------


def _trend_seasonal_signal(seed=0, T=500, P=50.0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    y = 1.0 + 0.01 * t + 0.8 * np.sin(2 * np.pi * t / P) + 0.05 * rng.standard_normal(T)
    return y, P


def test_holdout_selects_model_that_generalizes():
    # A trend+periodic model must beat trend-only on held-out imputation of a
    # signal that genuinely has seasonality.
    y, P = _trend_seasonal_signal()
    candidates = {
        "trend_only": lambda s: make_problem(
            s, components=[smooth_trend(1e2, role="trend")]
        ),
        "trend+periodic": lambda s: make_problem(
            s,
            components=[
                smooth_trend(1e2, role="trend"),
                multiperiodic(P, num_harmonics=4, weight=1e-2, role="seas"),
            ],
        ),
    }
    res = holdout_select(y, candidates, holdout_fraction=0.2)
    assert res["best"] == "trend+periodic"
    assert res["scores"]["trend+periodic"] < res["scores"]["trend_only"]


def test_holdout_explicit_slice_and_mae():
    y, P = _trend_seasonal_signal()
    candidates = {
        "trend+periodic": lambda s: make_problem(
            s,
            components=[
                smooth_trend(1e2, role="trend"),
                multiperiodic(P, num_harmonics=4, weight=1e-2, role="seas"),
            ],
        ),
    }
    res = holdout_select(
        y, candidates, holdout_slice=slice(200, 260), metric="mae"
    )
    assert res["metric"] == "mae"
    assert np.array_equal(res["holdout_index"], np.arange(200, 260))
    assert res["best"] == "trend+periodic"


def test_holdout_rejects_bad_metric():
    y, _ = _trend_seasonal_signal()
    with pytest.raises(ValueError, match="metric"):
        holdout_select(y, {"m": _slope_build_fn}, metric="bogus")


def test_holdout_failed_model_scores_nan_not_selected():
    # A candidate that always fails to build should get a NaN score and never
    # be selected; a working candidate wins.
    y, P = _trend_seasonal_signal()

    def broken(_s):
        raise RuntimeError("boom")

    candidates = {
        "broken": broken,
        "trend+periodic": lambda s: make_problem(
            s,
            components=[
                smooth_trend(1e2, role="trend"),
                multiperiodic(P, num_harmonics=4, weight=1e-2, role="seas"),
            ],
        ),
    }
    res = holdout_select(y, candidates, holdout_fraction=0.2)
    assert np.isnan(res["scores"]["broken"])
    assert res["best"] == "trend+periodic"
