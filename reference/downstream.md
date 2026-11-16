# Extraction, uncertainty, and reporting

Decide what quantity the user needs before choosing its extractor. A curve, a
per-sample slope, an annual percentage, and an event date have different units
and different stability questions. Keep domain conversions outside component
builders; `examples/pv_domain.py` illustrates that boundary for PV degradation.

## Extract and label outputs

```python
import numpy as np
import pandas as pd
from signaldecomp import (
    make_problem, solve, linear_trend, components_to_frame, format_report,
    bootstrap_ci, expanding_window_stability,
)

rng = np.random.default_rng(7)
y = 2 + 0.01 * np.arange(120) + 0.03 * rng.standard_normal(120)
y[30:33] = np.nan

def build_fn(signal):
    return make_problem(signal, components=[linear_trend(role="trend")])

def extractor(out):
    return {"slope_per_day": float(out["values"]["trend_b"])}

out = solve(build_fn(y))
assert out["status"] in ("optimal", "optimal_inaccurate")
point = extractor(out)
index = pd.date_range("2025-01-01", periods=len(y), freq="D")
frame = components_to_frame(out, index=index, y=y)
report = format_report(out, y=y)
```

Here each sample is one day, so the affine slope is per day. For other sampling
intervals, convert explicitly. For a log-domain affine trend with per-sample
slope `b`, compound change over `n` samples is `100 * expm1(b * n)` percent.
Do not apply that formula to a level-domain slope or to a nonlinear trend
without defining how a scalar slope is obtained.

`components_to_frame` distinguishes full-length roles from auxiliary coefficient
arrays. Its reconstruction sums structural roles and excludes the residual.
Passing `mask=out["fit_mask"]` hides predictions outside fitted rows; omitting
that argument retains them. Carry observed, component-valid, fitted, and scored
masks alongside exported results. A finite prediction at an unavailable driver
row is not evidence of a complete exogenous prediction: missing design rows
are zero-filled for assembly. Score only rows where all required inputs exist.

For a log model, use `prepare_input(..., log_transform=True)` before fitting
and `recover_components(..., log_transform=True)` or `recover_frame` afterward.
The recovered components multiply; exponentiated residuals are factors.
Exponentiating the structural sum gives the fitted log-scale central estimate,
not automatically an unbiased conditional mean in original units.

## Conditional bootstrap uncertainty

After the model and weights are fixed, the generic moving-block residual
bootstrap rebuilds and refits the same model for each resampled signal:

```python
intervals = bootstrap_ci(
    y, build_fn, extractor, block_size=4, n_resamples=30,
    confidence_level=95, random_state=11,
)
assert "slope_per_day" in intervals
```

Thirty replicates here are a runnable smoke example, not a recommendation for
reported tail quantiles. Select replicate count and required success fraction
for the application's accuracy needs. An empty result indicates too few
successful replicates; do not report it as a zero-width interval.

Choose `block_size` from remaining residual dependence and examine sensitivity
to it. The implementation forms blocks on the compressed sequence of fitted
residuals, so a block may bridge missing or input-unavailable time spans. With
large gaps, multiple regimes, heteroskedasticity, or irregular fitted support,
use a resampling procedure that respects those features instead of assuming
this helper preserves elapsed-time dependence.

This interval is conditional on the selected model and frozen preprocessing
policy. An exact-mask basis can be rebuilt on the same mask; resampled signal
values must not trigger a new tuning or specification step unless that is the
declared target. A conditional interval does not capture uncertainty from
choosing transformations, components, knots, weights, or selection rules.

End-to-end procedural uncertainty requires an outer resampling procedure that
repeats preparation, selection, fitting, and extraction, including deterministic
failure/null policies. `bootstrap_ci` does not implement that outer workflow
automatically. See [implementation.md](implementation.md) for stage boundaries.

## Expanding-window stability

```python
stability = expanding_window_stability(
    y, build_fn, min_window=60, step=20,
    extractor=extractor, tol=0.001, roles=["trend"],
)
assert "trend" in stability["snapshots"]
```

The helper fits growing prefixes, records component snapshots over the common
grid, and optionally tracks extracted scalars. `tol` uses the extractor's units;
declare it before inspecting the result. Windows end on finite observations,
but component input availability can further restrict the fitted support.
Exogenous closures must slice their drivers to the window and recompute exact
preprocessing masks. Stability is sensitivity to record length, not a confidence
interval or a proof of correctness.

Persist the specification, selected weights, time units, transforms, masks,
solver/status, extractor definition, and resampling seed/settings with results.
Use [diagnostics.md](diagnostics.md) for residual and component checks before
attaching scientific meaning to a scalar or interval.
