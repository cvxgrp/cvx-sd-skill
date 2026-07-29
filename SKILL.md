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

Convex signal decomposition separates a scalar time series into a residual and
interpretable structural components by solving one convex problem. Use
**CVXPY** as the modeling language, **`signaldecomp`** as the scaffold that
enforces the decomposition invariants, and **disciplined convex programming
(DCP)** as the check that keeps generated models convex and composable.

Build on `signaldecomp`; do not treat its catalog as a fixed menu. Read its
short component builders before adapting them, and read an `examples/` file
before translating a new domain. Locate an installed copy with
`python -c "import signaldecomp; print(signaldecomp.__file__)"`; in this repo,
the source is `src/signaldecomp/`.

## The substrate

Preserve four invariants:

- **Decomposition.** `y = x0 + x1 + … + xK`. Each component `xk` carries a loss
  `φk`; minimize their sum.
- **x0 is always the residual.** Append structural components as `x1, x2, …`,
  but address every solved component by **role**, never by index.
- **The mask handles missing data.** Impose `y = Σ xk` only at observed entries.
  Exclude missing, held-out, and unobserved grid points through the same mask;
  the summed structural components impute them.
- **Physical time lives in Δ.** `y` is a 1-D vector on a regular grid; a scalar
  `Δ` ties samples to physical time. Express periods in the same unit and
  convert to samples as late as possible.

Read [formulation.md](reference/formulation.md) before hand-writing components
or reviewing another implementation.

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

Identify the context before formulating:

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
  substrate ("this smoothing spline *is* an x0-residual with a smooth trend and
  no mask"), correct footguns, extend append-only. Much classical modeling —
  regression, splines, GAMs, Fourier — is a convex decomposition in disguise;
  recognizing that is the job. See
  [recontextualization.md](reference/recontextualization.md).

## Formulate: compose, don't shop

A component's cost is a **sum of convex terms plus a feasible set**, and those
stack freely. Compose the cost that matches the belief; do not pick the nearest
catalog entry.

Suppose a component should drift down slowly but recover in sharp jumps back to
a baseline of zero — soiling that accumulates, then washes off (an inverted
sawtooth). Nothing in the catalog is that. You compose it: penalize the
downward drift, leave recoveries free, pin the baseline.

```python
x = cp.Variable(T)
loss = w_d * cp.norm(cp.neg(cp.diff(x)), 2) + w_v * cp.norm1(x)
cons = [x <= 0]
```

The first term penalizes downward drift while leaving recoveries free; the
second pins the baseline; the constraint keeps the component non-positive.
Read [formulation.md](reference/formulation.md) for the worked treatment and
[component-catalog.md](reference/component-catalog.md) for reusable patterns.

After composing a component, verify DCP. `solve(..., verify_dcp=True)` does this
by default. A model can be convex and still be badly tuned, so inspect each
component rather than trusting solver status or fit score alone.

## The loop, end to end

Build, solve, read components by role. This is the whole cycle — everything else
is choosing the components and reading the outputs.

```python
import cvxpy as cp
from signaldecomp import (
    make_problem, solve, components_to_frame,
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
resid = out["values"]["residual"]        # x0 is always "residual"
df = components_to_frame(out, y=y)       # labeled DataFrame, gaps imputed
```

`solve` adds `status` and `values` to the built dictionary. `values` contains
the residual, each component keyed by role, and component auxiliary values.
`components_to_frame` adds the reconstruction and can restore a pandas index.
Catalog builders and hand-written `Component` objects share the same
`components` list and solve path.

## Footguns

These fail *silently* — the solve still returns `optimal`, the model still looks
fine. Earned emphasis; the rest live in [gotchas.md](reference/gotchas.md).

- **Scale periods by Δ.** A period is a physical duration, not an integer
  sample count. Express periods and Δ in the same unit and convert with
  `period_samples`.
- **A fixed-step grid can't represent DST.** If you build Δ from timestamps,
  supply local standard time without daylight-saving shifts. When computing Δ,
  use `Timedelta.total_seconds()`, never `.seconds`.
- **Do not holdout-tune a structural knob.** If a knob changes component shape
  without moving the reconstruction, judge the component by looking.
- **Bootstrap only the final model.** Do not mix tuning variation with
  final-model uncertainty.

## Reference

- [formulation.md](reference/formulation.md) — the substrate: x0-residual,
  masked linking, DCP as the verifiable target, composing bespoke components.
- [component-catalog.md](reference/component-catalog.md) — convex component
  vocabulary; excluded non-convex classes and their relaxations.
- [diagnostics.md](reference/diagnostics.md) — numerical inspection:
  periodograms, folds, variance explained, residual and driver checks.
- [implementation.md](reference/implementation.md) — reproducible builds,
  parameterized repeated solves, DPP, and solver configuration.
- [marimo.md](reference/marimo.md) — exploration as tier-classification by feel;
  the widget as a specification instrument; composing with the marimo skills.
- [model-specification.md](reference/model-specification.md) — Tier 1/2/3
  tuning, coupled-knob reduction, and the implementation handoff.
- [philosophy.md](reference/philosophy.md) — why convex; under-specification;
  the boundary between convex, convex-sequence, and discrete models.

### Planned

The references below are not yet written; the links are placeholders.

- [periodic-and-time.md](reference/periodic-and-time.md) — Fourier periodics,
  float periods, Δ-scaling, leap years, multi-scale, the trend↔seasonal
  confound.
- [time-axis.md](reference/time-axis.md) — standardizing raw timestamps to
  `(y, index, Δ)`; the heat-map diagnostic.
- [downstream.md](reference/downstream.md) — extraction, bootstrap CIs,
  expanding-window stability, reporting, the pandas round-trip.
- [recontextualization.md](reference/recontextualization.md) — recognizing
  latent convex decompositions in classical / hand-rolled code.
- [gotchas.md](reference/gotchas.md) — the fuller footgun list.
