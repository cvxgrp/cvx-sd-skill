# Component catalog

The vocabulary of ready-made components, and — just as important — the classes
this skill deliberately *excludes* and what to use instead. Every entry is a
worked `ell + I` pattern (see [formulation.md](formulation.md)); none is
privileged. When a belief isn't here, compose it.

Each builder returns a `Component` you drop into `make_problem(y,
components=[...])`. Weights follow one convention: **larger weight = stronger
belief = more regularized** (smoother, sparser, stiffer). Most builders take a
`role=` string that names the component in the solved output.

## x1: the data-fidelity term (residual)

x1 is always present and always the residual; you choose its loss with
`make_problem(..., residual_loss=...)`. Pass a **string** for the parameterless
defaults, or a **factory result** when you need to set a parameter:

| loss | how to pass | use when |
|------|-------------|----------|
| mean-square (default) | `"l2"` | Gaussian-ish noise |
| sum-absolute (robust) | `"l1"` | heavy tails / outliers |
| Huber | `huber_loss(M=1.0)` | robust but smooth near zero; `M` = quad→linear knee |
| pinball / quantile | `quantile_loss(q=0.5)` | asymmetric over/under-estimation cost |

```python
out = solve(make_problem(y, comps, residual_loss="l1"))
out = solve(make_problem(y, comps, residual_loss=huber_loss(M=0.5)))
```

The residual is keyed `"residual"` in `out["values"]`.

## Trend family

A trend is low-frequency structure. Four builders, differing only in the belief
about *what kind* of low-frequency shape:

- **`linear_trend(role="trend", slope_weight=0.0)`** — an affine trend `a + b*t`.
  Exposes the intercept and slope as aux (`<role>_a`, `<role>_b`); the slope is
  the per-sample rate of change. `slope_weight` optionally ridge-penalizes the
  slope. Belief: "the trend is a straight line."
- **`smooth_trend(weight, order=2, role="trend")`** — mean-square-smooth trend
  penalizing the `order`-th difference. `order=2` (default) penalizes curvature
  (a smooth, freely-bending trend); `order=1` penalizes slope (damps level
  changes). Belief: "the trend is smooth." The workhorse.
- **`pwl_trend(weight, role="trend")`** — piecewise-linear trend via an **L1**
  penalty on the second difference (L1 trend filtering). Yields a trend that is
  piecewise linear with a *small number of knots* — interpretable, breakpoint-
  style. Belief: "the trend is mostly straight with a few bends."
- **`monotone_trend(weight=0.0, increasing=False, role="trend")`** — an isotonic
  trend, non-increasing by default (set `increasing=True` for non-decreasing).
  For quantities that cannot reverse — cumulative degradation, wear. Optional
  `weight` adds second-difference smoothness. Belief: "this only goes one way."

`smooth_trend` vs `pwl_trend` is the key choice: L2-on-2nd-diff gives a
*curving* smooth trend; L1-on-2nd-diff gives a *piecewise-linear* one that
localizes change into a few kinks. Reach for `pwl` when you care about *where*
the trend changes slope.

## Periodic

- **`multiperiodic(periods, num_harmonics=6, weight=0.1, role="periodic")`** — a
  truncated-Fourier component over one or more periods, **DC column removed**
  (the constant offset belongs to the trend intercept, not here — this is what
  keeps trend and periodic from fighting over the mean). `periods` are in
  **samples**: convert from physical time with `period_samples(seconds, delta)`.
  Pass a sequence for multi-scale seasonality (daily + weekly + yearly) in one
  joint basis. `num_harmonics` sets the number of sin/cos pairs per period (more
  = sharper shapes); `weight` regularizes the coefficients. Aux `<role>_theta`
  holds the coefficient vector.

"Seasonal" is just one use — daily, weekly, or any cyclic pattern is expressed
the same way. See [periodic-and-time.md](periodic-and-time.md) for Δ-scaling,
leap years, harmonics-per-scale, and the trend↔seasonal confound.

## Exogenous (covariate) responses

Unlike time-based components, these are functions of an external covariate `z`
(time-aligned, `len(z) == T`). The covariate is captured at construction.

- **`exog_linear(z, weight=0.0, role="exog")`** — a linear response `beta * z`
  (e.g. load proportional to irradiance). Aux `<role>_beta` is the scalar
  coefficient. Belief: "the signal responds linearly to `z`."
- **`exog_spline(z, n_knots=10, knots=None, weight=0.01, role="exog")`** — a
  smooth, possibly nonlinear response via a natural cubic spline `H(z) @ coef`
  (linear beyond the boundary knots; constant column dropped). `weight` is a
  ridge penalty controlling smoothness; more knots = more flexible. Aux
  `<role>_coef`. Belief: "the signal responds smoothly but nonlinearly to `z`"
  (e.g. a U-shaped load-vs-temperature curve).

## Wrappers: adding constraints to any component

These take a component and add an indicator `I` without touching its loss —
composition by wrapping:

- **`bounded(inner, lower=None, upper=None)`** — elementwise box `lower <= x <=
  upper`; either bound may be `None` (one-sided). E.g. `bounded(smooth_trend(1e2),
  lower=0.0)` is a nonnegative smooth trend.
- **`nonneg(inner)`** — shorthand for `bounded(inner, lower=0.0)`.

## Reading the results

`out["values"]` maps each `role` to its solved array, plus `"residual"`, plus
the aux quantities each builder exposes (`trend_a`, `trend_b`, `exog_beta`,
`periodic_theta`, ...). Address everything by role; never by index.

## Excluded: non-convex classes, and what to use instead

This skill is convex-only. Some tempting structures are genuinely non-convex;
forcing them in would break the DCP guarantee that makes generation safe. The
honest move is to know the boundary and reach for the convex stand-in.

**Refuse (no in-scope convex equivalent):**

- **finite-set / Boolean / integer** values (a component constrained to
  `{0, 1}` or a discrete set) — mixed-integer, not convex.
- **exact cardinality / true L0** ("*exactly* k nonzeros") — combinatorial.
- **Markov / regime-switching states** with discrete transitions.

For these, the problem is outside V1; see the operating band in
[philosophy.md](philosophy.md).

**Relax (a convex surrogate captures the intent):**

- **sparsity / "few nonzeros"** → **`sparse(weight)`**, the L1 surrogate for
  cardinality. Zero at most entries, nonzero at a few. Use for outliers, spikes,
  rare events.
- **"a few breakpoints" / near-L0 structure** → L1 gets you most of the way
  (`pwl_trend`, `sparse`). For a *sharper* approximation to L0 — fewer, cleaner
  nonzeros with less magnitude bias — **iteratively reweighted L1 (IRL1)**: solve,
  then re-solve with per-entry weights `1/(|prev| + eps)`, 2–3 times. Each pass
  drives already-small entries toward zero and frees large ones from shrinkage.

  ```python
  weight = 0.02                            # start LOOSE; see note below
  w = np.ones(T - 2)
  for _ in range(3):                       # 2-3 iterations ~ L0
      x = cp.Variable(T)
      loss = weight * cp.norm1(cp.multiply(w, cp.diff(x, k=2)))
      # ... solve with this component ...
      w = 1.0 / (np.abs(np.diff(x.value, n=2)) + 1e-3)
  ```

  The important, counter-intuitive detail: **start from a *loose* (low-weight)
  L1 fit and let the reweighting concentrate it.** A loose plain-L1 pass is
  overcomplete (many small knots); reweighting then drives the spurious ones to
  zero and keeps the real ones, converging to the true breakpoint count in 2–3
  iterations. Starting from a *tight* (high-weight) fit instead collapses the
  component toward a straight line and loses real structure. As always in this
  regime, weights are small — think `~1e-2`, not `~1`.

  IRL1 is the canonical "second problem uses the first's output" pattern, and it
  is DPP-friendly (only the weights change between solves). Fuller treatment,
  including the fast re-solve, is in [implementation.md](implementation.md).
- **monotone-with-jumps** → `monotone_trend` plus a `sparse` component for the
  jumps.

The rule of thumb: if the belief is "few / mostly-zero / rare," there is almost
always a convex L1-flavored relaxation. If it is "exactly discrete," there is
not — and that is a scope boundary, not a formulation to force.
