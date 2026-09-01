"""End-to-end lagged spline response with exact fitted-basis whitening.

This synthetic example keeps the focus on the package contract: construct the
exact fit mask before fitted spline preprocessing, solve a lagged response, and
score a held-out interval only where the driver is available.
"""

from __future__ import annotations

import numpy as np

from signaldecomp import (
    exog_spline,
    linear_trend,
    make_offset_basis,
    make_problem,
    solve,
)


def main():
    rng = np.random.default_rng(19)
    T = 360
    driver = rng.uniform(-2.0, 2.0, T)
    signal = (
        0.4
        + np.sin(driver)
        + 0.35 * np.sin(np.roll(driver, 1))
        + 0.03 * rng.standard_normal(T)
    )
    signal[0] = np.nan  # the one-step offset has no source here
    driver[145] = np.nan

    holdout = np.arange(280, 320)
    training_signal = signal.copy()
    training_signal[holdout] = np.nan

    offsets = (0, 1)  # current value, then one sample in the past
    driver_valid = make_offset_basis(driver, offsets).valid_mask
    fit_mask = np.isfinite(training_signal) & driver_valid

    driver_component = exog_spline(
        driver,
        n_knots=6,
        weight=1e-4,
        role="driver",
        offsets=offsets,
        lag_smooth_weight=1e-3,
        whiten=True,
        fit_mask=fit_mask,
    )
    out = solve(
        make_problem(
            training_signal,
            components=[
                linear_trend(role="baseline", slope_weight=1e-2),
                driver_component,
            ],
        )
    )

    reconstruction = out["values"]["baseline"] + out["values"]["driver"]
    score_index = holdout[driver_valid[holdout] & np.isfinite(signal[holdout])]
    holdout_rmse = np.sqrt(
        np.mean((reconstruction[score_index] - signal[score_index]) ** 2)
    )
    whitening = out["component_metadata"]["driver"]["whitening"]

    print(f"status: {out['status']}")
    print(f"fitted rows: {out['fit_mask'].sum()} / {T}")
    print(f"held-out RMSE: {holdout_rmse:.4f}")
    print(f"raw fitted-basis condition number: "
          f"{whitening.diagnostics.condition_number:.1f}")
    print(f"whitened Gram error: {whitening.training_gram_error:.2e}")
    print(f"original coefficient shape: {out['values']['driver_coef'].shape}")

    assert out["status"] in ("optimal", "optimal_inaccurate")
    assert holdout_rmse < 0.08
    assert whitening.training_gram_error < 1e-10


if __name__ == "__main__":
    main()
