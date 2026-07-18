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
  loss = weight * cp.norm1(cp.diff(x, k=2))  # L1 on 2nd difference -> few knots
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
*samples*: spikes, outliers). Other dictionaries give other structure: a
step/integrator dictionary -> a component with few jumps; a bank of event
templates -> a few events.

**Analysis sparsity** — a linear transform of the component is sparse; sparsity
lives in `L @ x`:

```python
x    = cp.Variable(T)
loss = weight * cp.norm1(L @ x)    # few nonzeros in the transformed domain
```

`L = diff(k=2)` gives `pwl_trend` (few slope changes); `L = diff` gives few level
changes (piecewise-constant).

So two of the headline components are instances of this one idea: **`sparse` is
synthesis with `A = I`; `pwl_trend` is analysis with `L = diff(k=2)`.** That is
"compose, don't shop" in miniature — pick the `A` or `L` that matches your belief.

Two honest notes. A **bare `l1` on the residual** (`residual_loss="l1"`) is not
usually the best way to get robustness. The classic, more useful construction is
a **decomposition**: keep the default `l2` (`sum_squares`) residual for the
Gaussian bulk and append a **`sparse` component** for the few large outliers —
`sum_squares(residual) + w * cp.norm1(x_sparse)`, two terms on two variables. The
residual takes the bulk; the sparse component takes the spikes. (Similar gestalt
to elastic net, but *not* elastic net — elastic net puts both penalties on one
variable; here they are on two.) Either sparse form can be **sharpened toward
true L0** by reweighting the `l1(...)` (IRL1, below).

## Beyond convex: sequences of convex solves

This skill is convex-only, and a single convex solve is *globally optimal* by
construction — that is the guarantee that makes generation safe. Some tempting
structures are genuinely non-convex and cannot be one such solve. But many are
still approachable as a **sequence of convex solves**: you lift the hard problem
into repeated convex problems, each exact, using the previous solution to guide
the next.

These sequence methods are **local search heuristics: they often work quite well
in practice, and while global optimality cannot be guaranteed, useful solutions
are common.** That is the honest trade — you give up the single-solve certificate
for reach, and you judge the result by *looking*, not by a guarantee.

**Genuinely out of scope** (no good in-scope convex-sequence heuristic; see the
operating band in [philosophy.md](philosophy.md)):

- **Markov / regime-switching states** with discrete transition dynamics.
- anything requiring a *proven* global integer optimum (that is mixed-integer
  programming, a different tool).

**Approachable as a convex sequence:**

- **near-L0 / "a few nonzeros," sharper than plain L1** → **iteratively
  reweighted L1 (IRL1)**. Plain L1 (the `sparse` / `pwl_trend` patterns above, see
  [Sparsity](#sparsity-a-pattern-not-a-single-component)) is a biased L0 surrogate;
  IRL1 sharpens it. Solve, then re-solve with per-entry weights `1/(|prev| + eps)`,
  2–3 times. Each pass drives already-small entries toward zero and frees large
  ones from shrinkage.

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
- **Boolean / finite-set values** → **relax-round-polish**. Relax the discrete
  set `{0, 1}` to its convex hull `[0, 1]`, solve, round to the nearest feasible
  point, then *fix* the rounded values and re-solve any remaining free variables
  (the "polish"). Same class as IRL1 — a non-convex problem lifted into a short
  sequence of convex solves.

  ```python
  # relax: b in [0, 1] instead of {0, 1}
  b = cp.Variable(T)
  solve_problem_with(b, constraints=[b >= 0, b <= 1])   # convex
  # round: snap to the discrete set
  b_fixed = (b.value > 0.5).astype(float)
  # polish: fix b, re-solve the remaining free variables (still convex)
  solve_problem_with(b_fixed, free=other_vars)
  ```

  In practice the relaxed solution often lands near the box corners already, so
  rounding is a small correction; the polish pass recovers the other variables
  exactly given the discrete choice.

The rule of thumb: "few / mostly-zero / rare" almost always has a convex
L1-flavored relaxation (one solve); "discrete-valued" is reachable by a convex
sequence (relax-round-polish, a heuristic); "provably-optimal discrete" is out of
scope. Know which one you are in.
