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
