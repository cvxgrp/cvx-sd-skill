# Component catalog

The vocabulary of ready-made components, and — just as important — the classes
this skill deliberately *excludes* and what to use instead. Every entry is a
worked `ell + I` pattern (see [formulation.md](formulation.md)); none is
privileged. When a belief isn't here, compose it.

Each builder returns a `Component` you drop into `make_problem(y,
components=[...])`. Weights follow one convention: **larger weight = stronger
belief = more regularized** (smoother, sparser, stiffer). Most builders take a
`role=` string that names the component in the solved output.

## x0: the data-fidelity term (residual)

x0 is always present and always the residual; you choose its loss with
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

  ```python
  expr = coef[0] + coef[1] * t             # affine: intercept + slope
  loss = slope_weight * cp.square(coef[1]) # optional ridge on the slope
  ```
- **`smooth_trend(weight, order=2, role="trend")`** — mean-square-smooth trend
  penalizing the `order`-th difference. `order=2` (default) penalizes curvature
  (a smooth, freely-bending trend); `order=1` penalizes slope (damps level
  changes). Belief: "the trend is smooth." The workhorse.

  ```python
  loss = weight * cp.sum_squares(cp.diff(x, k=order))  # order=2: penalize curvature
  ```
- **`pwl_trend(weight, role="trend")`** — piecewise-linear trend via an **L1**
  penalty on the second difference (L1 trend filtering). Yields a trend that is
  piecewise linear with a *small number of knots* — interpretable, breakpoint-
  style. Belief: "the trend is mostly straight with a few bends."

  ```python
  d = cp.diff(x, k=2)
  loss = weight / d.shape[0] * cp.norm1(d)  # L1 on 2nd diff (per-entry: /dim) -> few knots
  ```
- **`pwc_trend(weight, role="trend")`** — piecewise-**constant** trend via an
  **L1** penalty on the *first* difference. Holds a level and shifts in a small
  number of steps — a level-shift / regime / step-change signal. The
  first-difference analogue of `pwl_trend`: `pwl` localizes *slope* changes,
  `pwc` localizes *level* changes. Belief: "the level is constant except for a
  few shifts."

  ```python
  d = cp.diff(x)
  loss = weight / d.shape[0] * cp.norm1(d)  # L1 on 1st diff (per-entry: /dim) -> few level shifts
  ```
- **`monotone_trend(weight=0.0, increasing=False, role="trend")`** — an isotonic
  trend, non-increasing by default (set `increasing=True` for non-decreasing).
  For quantities that cannot reverse — cumulative degradation, wear. The default
  (`weight=0`) is a bare isotonic fit that steps freely between levels — the
  monotonicity constraint alone permits jumps. Optional `weight` adds
  second-difference smoothness if you want the climb to be gradual. Belief: "this
  only goes one way."

  ```python
  cons = [cp.diff(x) <= 0]                        # non-increasing (>= 0 if increasing)
  loss = weight * cp.sum_squares(cp.diff(x, k=2)) # optional smoothness (0 if weight=0)
  ```

`smooth_trend` vs `pwl_trend` is the key choice: L2-on-2nd-diff gives a
*curving* smooth trend; L1-on-2nd-diff gives a *piecewise-linear* one that
localizes change into a few kinks. Reach for `pwl` when you care about *where*
the trend changes slope.

### Difference-penalty lexicon

The trend penalties are all a **norm of a difference of `x`** — an *analysis*
penalty (sparsity/energy in `L @ x`, here `L = k`-th difference). Two choices,
norm and difference-order, give a small named vocabulary:

| | **L1** of the difference (local: few nonzeros) | **L2²** of the difference (global: small total energy) |
|---|---|---|
| **1st diff** `diff(x)` | few *level changes* → **piecewise-constant** (`pwc_trend`) | small total *level variation* → a gently drifting level (`smooth_trend(order=1)`) |
| **2nd diff** `diff(x,2)` | few *slope changes* → **piecewise-linear** (`pwl_trend`) | small total *curvature* → **smooth** (`smooth_trend`) |

The named cases use first differences for level changes and second differences
for slope changes. L1 expresses a local sparsity claim, so the builders divide
by the difference length; L2² expresses total global roughness and is not
length-normalized. This keeps locally sparse weights comparable across record
lengths without changing the meaning of global smoothness.

## Multiperiodic (strictly periodic is a special case)

- **`multiperiodic(periods, num_harmonics=6, weight=0.1, role="periodic")`** — a
  joint **quasi-periodic** model over one or more periods. With a single period
  it reduces to an ordinary truncated-Fourier series — the strictly-periodic
  special case. With several periods its distinctive power is the **cross-terms**.

  The basis is `[offset] + [a Fourier block per period] + [pairwise cross-terms]`,
  with the **DC/offset column dropped** (the constant belongs to the trend
  intercept — this is what keeps trend and periodic from fighting over the mean).
  The **cross-terms are products of one period's harmonics with another's**, so
  the coefficients don't merely set each period's amplitude independently — they
  let **the *shape* of the short-period cycle change as the long period
  advances.** A model with daily and yearly periods can represent a *daily
  profile that reshapes across the seasons* — a summer day and a winter day with
  genuinely different within-day curves — not just the same daily curve scaled up
  and down. Turn the cross-terms off and you are back to the strictly-periodic
  case: a fixed shape whose amplitude alone can vary.

  `num_harmonics` sets harmonics per period; **`max_cross_k`** caps how many
  harmonics per side enter the cross-terms — the knob for how much shape-drift
  the model can express (and the basis width). `periods` are in **samples**
  (convert with `period_samples(seconds, delta)`); pass a sequence for
  multi-scale structure. Aux `<role>_theta` holds the coefficient vector.

  ```python
  expr = B @ theta                    # B = [offset | per-period blocks | cross-terms], DC dropped
  loss = cp.sum_squares(reg @ theta)  # reg diagonal: weight * (2*pi/sqrt(P)) * i for harmonic i
                                      # -> Dirichlet (derivative) energy: higher harmonics cost more
  ```

  The `reg` term is a **Dirichlet-energy penalty**: a diagonal weight
  `weight * (2*pi/sqrt(P)) * i` on harmonic `i` of period `P` (the `sqrt(P)`
  normalizes across periods; the offset is unregularized). It is the leash on all
  that shape-flexibility — high harmonics cost more, biasing toward the smoothest
  periodic shape consistent with the data. `weight` trades shape-fidelity against
  smoothness.

"Seasonal" is just one use — daily, weekly, or any cyclic pattern is expressed
the same way. See [periodic-and-time.md](periodic-and-time.md) for Δ-scaling,
leap years, harmonics-per-scale, and the trend↔seasonal confound.

## Exogenous (covariate) responses

Unlike time-based components, these are functions of an external covariate `z`
(time-aligned, `len(z) == T`). The covariate is captured at construction.

- **`exog_linear(z, weight=0.0, role="exog")`** — a linear response `beta * z`
  (e.g. load proportional to irradiance). Aux `<role>_beta` is the scalar
  coefficient. Belief: "the signal responds linearly to `z`."

  ```python
  expr = beta * z                 # linear response to covariate z
  loss = weight * cp.square(beta) # optional ridge on the coefficient
  ```
- **`exog_spline(z, n_knots=10, knots=None, weight=0.01, role="exog")`** — a
  smooth, possibly nonlinear response via a natural cubic spline `H(z) @ coef`
  (linear beyond the boundary knots; constant column dropped). `weight` is a
  ridge penalty controlling smoothness; more knots = more flexible. Aux
  `<role>_coef`. Belief: "the signal responds smoothly but nonlinearly to `z`"
  (e.g. a U-shaped load-vs-temperature curve).

  ```python
  expr = H(z) @ coef                   # natural cubic spline basis in z (const col dropped)
  loss = weight * cp.sum_squares(coef) # ridge -> smoothness
  ```

## Wrappers: adding constraints to any component

These take a component and add an indicator `I` without touching its loss —
composition by wrapping:

- **`bounded(inner, lower=None, upper=None)`** — elementwise box `lower <= x <=
  upper`; either bound may be `None` (one-sided). E.g. `bounded(smooth_trend(1e2),
  lower=0.0)` is a nonnegative smooth trend.
- **`nonneg(inner)`** — shorthand for `bounded(inner, lower=0.0)`.

  ```python
  expr, loss, cons = inner.build(T)   # inner component unchanged
  cons += [x >= lower, x <= upper]    # add box (either bound optional)
  ```

## Reading the results

`out["values"]` maps each `role` to its solved array, plus `"residual"`, plus
the aux quantities each builder exposes (`trend_a`, `trend_b`, `exog_beta`,
`periodic_theta`, ...). Address everything by role; never by index.

## Sparsity: a pattern, not a single component

"Few nonzeros" appears in two distinct convex forms. Naming them keeps you from
reaching for a bare `l1` when you want something structural.

**Synthesis sparsity** — the component is a sparse combination of dictionary
atoms; sparsity lives in the *coefficients*:

```python
theta = cp.Variable(A.shape[1])
xk    = A @ theta                  # component = sparse mix of A's columns (atoms)
loss  = weight * cp.norm1(theta)   # few atoms selected
```

`A = I` is the special case — exactly the `sparse(weight)` builder (few nonzero
*samples*: spikes, outliers), whose loss is `weight / x.shape[0] * cp.norm1(x)`
(per-entry normalized, since sparsity is a local/density claim). Other dictionaries give other structure: a
step/integrator dictionary -> a component with few jumps; a bank of event
templates -> a few events.

**Analysis sparsity** — a linear transform of the component is sparse; sparsity
lives in `L @ x`:

```python
x    = cp.Variable(T)
Lx   = L @ x
loss = weight / Lx.shape[0] * cp.norm1(Lx)  # few nonzeros; per-entry (/dim) as it is a local claim
```

`L = diff(k=2)` gives `pwl_trend` (few slope changes); `L = diff` gives few level
changes (piecewise-constant).

So two of the headline components are instances of this one idea: **`sparse` is
synthesis with `A = I`; `pwl_trend` is analysis with `L = diff(k=2)`.** That is
"compose, don't shop" in miniature — pick the `A` or `L` that matches your belief.

A bare L1 residual is not always the most informative robust model. To separate
Gaussian noise from isolated outliers, keep the default L2 residual and append
a `sparse` component: `sum_squares(residual) + w * norm1(x_sparse)`.

### Escalating from L1 to IRL1

Start with plain L1 when the belief is “few nonzeros.” Escalate to
**iteratively reweighted L1 (IRL1)** when the fitted component has too many
small nonzeros or L1 shrinkage biases the important ones toward zero. The
second effect matters when amplitudes carry meaning: for a piecewise-constant
component, shrinkage can distort the estimated level changes even when their
locations are correct.

IRL1 re-solves with entrywise weights `1 / (abs(previous) + eps)`. Small entries
become more expensive while large entries are penalized less:

```python
weight = 1.0
w = np.ones(T - 1)
for _ in range(3):
    x = cp.Variable(T)
    d = cp.diff(x)
    loss = weight / d.shape[0] * cp.norm1(cp.multiply(w, d))
    # ... solve with this component ...
    w = 1.0 / (np.abs(np.diff(x.value)) + 1e-3)
```

Start from a loose L1 fit; an overly tight first pass may erase real structure
before reweighting can preserve it. Two or three passes are usually enough to
see whether sparsity, amplitudes, and locations stabilize. Each pass is convex,
but the sequence is a local-search heuristic. See
[implementation.md](implementation.md) for efficient repeated solves.

## Discrete-valued structure

Raise the modeling choice early when the user describes Boolean, integer, or
finite-set states. CVXPY supports Boolean and integer variables and can
formulate mixed-integer convex programs for compatible solvers. The question is
not whether exact discrete modeling exists; it is whether it remains tractable
at the required horizon and number of discrete variables.

When an exact mixed-integer solve is too costly, **relax-round-polish** is a
useful heuristic: relax the discrete set to its convex hull, solve, round to a
feasible value, then fix the rounded values and re-solve the remaining
continuous variables.

```python
b = cp.Variable(T)
theta = cp.Variable(A.shape[1])
relaxed = cp.Problem(
    cp.Minimize(cp.sum_squares(A @ theta + b - y)),
    [b >= 0, b <= 1],
)
relaxed.solve()
b_fixed = np.rint(b.value)
cp.Problem(cp.Minimize(cp.sum_squares(A @ theta + b_fixed - y))).solve()
```

Use the exact mixed-integer formulation when the problem size and available
solver make it practical or when a global discrete guarantee is required. Use
relax-round-polish when long horizons make exact search impractical and a
checked heuristic is acceptable. See [philosophy.md](philosophy.md) for the
operating boundary.
