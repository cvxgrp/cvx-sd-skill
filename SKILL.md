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
