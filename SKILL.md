---
name: cvx-signal-decomposition
description: >-
  Convex signal decomposition (cvx-sd): translate a scalar time-series problem
  into a residual plus interpretable structural components — trend, periodic,
  sparse, exogenous — formulate it in CVXPY, solve it, and wire the outputs to
  extraction, validation, and reporting. Use when a user has time-series data
  and wants to separate it into meaningful parts, fit a trend/seasonality/
  outlier model, impute missing values, or when existing smoothing / detrending
  / seasonal-decomposition code should be recognized and rebuilt as a convex
  program.
allowed-tools: Read, Write, Edit, Bash(uv run **)
---

Convex signal decomposition (cvx-sd) separates a scalar time series into a
residual plus a few interpretable structural components — a trend, a periodic
term, sparse spikes, a response to a covariate — by solving one convex problem.
This applies convex optimization theory and the basic concept that a loss is an
encoding of prior belief (an ℓ₂ penalty is a Gaussian belief, ℓ₁ a Laplace one,
a constraint a hard prior); this skill is about formulating decomposition
consistently, so models compose, extend, and can be generated and checked
safely — not about a fixed catalog of components. cvxpy is the modeling language
and disciplined convex programming (DCP) is the type system that ensures
generated problems are convex.

## The substrate

Every decomposition sits on the same four invariants. Get these right and the
rest composes; get one wrong and the model is subtly broken in ways the solver
won't flag.

- **Decomposition.** `y = x1 + x2 + … + xK`. Each component `xk` carries a loss
  `φk` measuring how implausible that shape is; the decomposition minimizes the
  total loss. The smaller a component's loss, the more the model believes it.
- **x1 is always the residual** (mean-square-small, or a robust variant). This
  is not a convention: the residual is the one term that can be eliminated in
  closed form, which is exactly what lets structural components be appended —
  `x2, x3, …` — without renumbering. Downstream code references components by
  **role** (`"trend"`, `"periodic"`), never by index.
- **The mask gracefully handles missing data.** The consistency `y = Σ xk` is
  imposed only on observed entries. Missing data, held-out validation data, and
  unobserved grid points are the same mechanism — exclusion from the mask — and
  the summed components impute the rest. Hold out some known entries and score
  the imputation: that is a simple model selection procedure.
- **Physical time lives in Δ.** `y` is a 1-D vector on a regular grid; a scalar
  `Δ` ties index space to physical time, in whatever unit the application uses
  (seconds, days, months). Periods are real numbers in the same unit and are
  scaled by Δ as late as possible — never hard-coded as integer sample counts.
  (`time_axis.py` is a convenience for deriving `(y, index, Δ)` from raw
  sub-daily timestamps; the contract itself is just those three.)

## What the skill does

Two moves. **Translation-IN:** turn domain intent into components, losses, and
transforms — "the trend can't reverse" becomes a monotone constraint,
"outliers, not noise" becomes a robust residual loss, "proportional
seasonality" becomes a log transform. **Translation-OUT:** wire the solved
components back to what the user actually wanted — a scalar extracted from a
curve, a confidence interval, a stability check, a labeled DataFrame or plot.
The convex solve sits in the middle; the skill is the translation on both
sides, not the components themselves.

## Which situation are you in

The first move differs completely by context; identify it before formulating
anything.

- **Exploration** — data on disk, model unknown ("is there a trend?"). Diagnose
  before you commit. **We recommend marimo here**: build a live notebook where
  sliders and dropdowns let the user *feel* the tradeoffs between model
  families. Take over at the standardize step (`time_axis`, `heatmap`). See
  [marimo.md](reference/marimo.md).
- **Implementation** — model decided, target is production. Be precise and
  deterministic: plain `signaldecomp` (or generated) CVXPY, a correct Δ,
  reproducible build/solve, tests. Rarely a single solve — see
  [implementation.md](reference/implementation.md).
- **Review / edit** — existing code, SD or SD-shaped. Read it, map it onto the
  substrate ("this smoothing spline *is* an x1-residual with a smooth trend and
  no mask"), correct footguns, extend append-only. Much classical modeling —
  regression, splines, GAMs, Fourier — is a convex decomposition in disguise;
  recognizing that is the job. See
  [recontextualization.md](reference/recontextualization.md).

These flow into each other: exploration hands off to implementation; review can
kick off exploration.

## Formulate: compose, don't shop

A component's cost is a **sum of convex terms plus a feasible set**, and those
stack freely. This is the core move — you *compose* a cost that matches the
belief, you don't pick the nearest catalog entry.

Suppose a component should drift down slowly but recover in sharp jumps back to
a baseline of zero — soiling that accumulates, then washes off (an inverted
sawtooth). Nothing in the catalog is that. You compose it: penalize the
downward drift, leave recoveries free, pin the baseline.

```python
x = cp.Variable(T)
loss = w_d * cp.norm(cp.neg(cp.diff(x)), 2) + w_v * cp.norm1(x)  # w_v tiny, ~1e-6
cons = [x <= 0]
```

An L2 (group-lasso) penalty on the *negative* first differences costs the slow
decline; upward steps are unpenalized, so washes snap back freely; a whisper of
L1 pins the level at zero without shaping the fit.

**Reach for `signaldecomp`** when a component *is* a catalog entry — it's tested
and correct on the fiddly details (masked linking, dropped DC column, Δ-scaled
periods). But the entries are worked patterns, not a fence: a `pwl_trend` *is*
`weight * norm1(diff(x, k=2))`. Once you see that, you write the variants the
catalog never anticipated.

**DCP is the check that makes this safe** — compose the pieces, then let CVXPY
confirm the whole is convex. `solve(..., verify_dcp=True)` (the default) refuses
a non-convex model rather than returning a meaningless answer. Construct and
verify; never reason "this must be convex" and ship it.

Composed costs like the sawtooth are *tuning-sensitive*: too heavy a pin
flattens it, too heavy a drift penalty erases the soiling, and both still solve
cleanly. You judge them by **looking at the component**, not by a fit score —
see [formulation.md](reference/formulation.md) for the worked walk-through and
[model-specification.md](reference/model-specification.md) for why.

## The loop, end to end

Build, solve, read components by role. This is the whole cycle — everything else
is choosing the components and reading the outputs.

```python
import numpy as np
import cvxpy as cp
from signaldecomp import (
    make_problem, solve, components_to_frame, plot_decomposition,
    smooth_trend, multiperiodic, period_samples, Component,
    SECONDS_PER_DAY, SECONDS_PER_YEAR,
)

y = ...          # load and prepare data: 1-D array, NaN where missing
delta = SECONDS_PER_DAY  # daily samples; periods scale by delta
# (raw timestamps? standardize_time_axis(df) returns y, delta, and an index)

# A component is a build(T) -> (expr, loss, constraints); catalog builders
# return these, and you can hand-write one when nothing in the catalog fits.
def build_spikes(T):
    x = cp.Variable(T)
    return x, 5.0 * cp.norm1(x), [x >= 0]   # sparse, nonnegative

built = make_problem(
    y,
    components=[
        smooth_trend(1e2, role="trend"),               # catalog builder
        multiperiodic(
            period_samples(SECONDS_PER_YEAR, delta),
            num_harmonics=4, role="seasonal",
        ),
        Component(role="spikes", build=build_spikes),   # hand-composed
    ],
)
out = solve(built)                       # verifies DCP, then solves
trend = out["values"]["trend"]           # solved arrays, keyed by role
resid = out["values"]["residual"]        # x1 is always "residual"
df = components_to_frame(out, y=y)        # labeled DataFrame, gaps imputed
fig = plot_decomposition(out, y=y)       # signal+fit, per-role, residual panels
```

`solve` returns the built dict plus `status` and `values` — a dict from role to
the solved array, including `residual` (x1) and any component aux (a trend
slope, the seasonal coefficients). Components are always addressed by **role**,
never by index, so adding a component later doesn't renumber anything.
`components_to_frame` wraps the full-length components onto a pandas index (pass
`index=` from `standardize_time_axis`), with a `reconstruction` column and the
imputed values filled in at the gaps; `plot_decomposition` returns a Matplotlib
figure (marimo-cell-friendly) with a panel for the signal-plus-fit, one per
structural role, and the signed residual.

The `components` list is the seam: a catalog builder and a hand-written
`Component` sit in it side by side and go through the same solve — the two paths
from "compose, don't shop" plug into one loop.

## Footguns

These fail *silently* — the solve still returns `optimal`, the model still looks
fine. Earned emphasis; the rest live in [gotchas.md](reference/gotchas.md).

- **Scale periods by Δ.** A period is a physical duration, not an integer
  sample count. Express periods and Δ in the *same* unit (whatever the
  application uses) and convert with `period_samples`; hard-coding a period as a
  sample count silently mis-tunes it and breaks on leap years / irregular Δ.
- **A fixed-step grid can't represent DST.** If you build Δ from timestamps,
  supply local *standard* time (no daylight-saving shifts) — a fixed-duration
  grid has no 23h/25h days, so DST-shifted stamps produce spurious gaps.
  (Computing Δ from timestamps yourself? Use `Timedelta.total_seconds()`, never
  `.seconds`, which wraps at 24h.)
- **Don't holdout-tune a structural (Tier-3) knob, or a knob on a component
  that barely moves the reconstruction.** Holdout scores imputation of the
  *reconstruction*: a knob that changes a component's *shape* but not the fit (a
  breakpoint weight), or one on a low-contribution component, has an "optimum"
  that is noise or the wrong objective. Judge those by **looking at the
  component**. See [model-specification.md](reference/model-specification.md).
- **Confidence intervals are a final-model step.** Never bootstrap inside a
  tuning loop or before the model is specified — it conflates parameter
  uncertainty with tuning variation and wastes the expensive resampling on
  models you'll discard.

## Reference

- [formulation.md](reference/formulation.md) — the substrate: x1-residual,
  masked linking, DCP as the verifiable target, composing bespoke components.
- [component-catalog.md](reference/component-catalog.md) — convex component
  vocabulary; excluded non-convex classes and their relaxations.

### Planned

The references below are not yet written; the links are placeholders.

- [periodic-and-time.md](reference/periodic-and-time.md) — Fourier periodics,
  float periods, Δ-scaling, leap years, multi-scale, the trend↔seasonal
  confound.
- [time-axis.md](reference/time-axis.md) — standardizing raw timestamps to
  `(y, index, Δ)`; the heat-map diagnostic.
- [model-specification.md](reference/model-specification.md) — the Tier 1/2/3
  tuning hierarchy; which knobs to holdout-tune, set by magnitude, or judge by
  looking.
- [implementation.md](reference/implementation.md) — spec→production;
  tune-then-solve, runtime tuning, DPP-accelerated scans, CI at the end.
- [downstream.md](reference/downstream.md) — extraction, bootstrap CIs,
  expanding-window stability, reporting, the pandas round-trip.
- [recontextualization.md](reference/recontextualization.md) — recognizing
  latent convex decompositions in classical / hand-rolled code.
- [marimo.md](reference/marimo.md) — exploration as tier-classification by feel;
  the widget as a specification instrument.
- [philosophy.md](reference/philosophy.md) — why convex, under-specification,
  the operating band, the broader line of work.
- [gotchas.md](reference/gotchas.md) — the fuller footgun list.
