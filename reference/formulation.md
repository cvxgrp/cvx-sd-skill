# Formulation: the substrate

This is the layer beneath the component catalog: what a signal decomposition
*is* as a convex program, what a component *is*, and why DCP is the check that
lets you generate new ones safely. Read it once; the rest of the skill assumes
it.

## The decomposition

A decomposition writes the observed signal as a sum of `K` estimated
components:

    y = x0 + x1 + ... + xK

`y` is a 1-D signal on a regular grid and **may have missing values**. The
components are the unknowns we solve for, and they are **real-valued
everywhere** — no gaps. That asymmetry is the whole engine of imputation: where
`y` is missing, the summed components still take a value, and that value *is*
the model's estimate of the missing datum.

**x0 is always the data-fidelity term** (the residual). Its cost is the
data-fidelity loss — mean-square by default, or a robust variant — selected
with `make_problem(..., residual_loss=...)`. Everything else (`x1 ... xK`) is a
structural component you append. Because the fidelity term occupies the fixed
first slot, adding or removing structural components never renumbers anything;
downstream code addresses components by **role** (`"trend"`, `"seasonal"`),
never by index. The residual is always keyed `"residual"` in the solved output.

## The mask: missing data as a linear operator

Let `M` be the **mask operator** — the linear map that selects the observed
entries of a signal and drops the missing ones. Missing data is handled by
applying `M` to the *consistency constraint*:

    M y = M (x0 + x1 + ... + xK)

i.e. the components must sum to `y` **at the known entries only**. At missing
entries there is no constraint, so the components are free to take whatever
value their costs prefer — which is exactly the imputed estimate.

`M` is a genuine, well-defined linear operator, not a special case bolted on.
This is why one mechanism covers three things that look different but are not:

- **missing data** — entries absent in the raw signal;
- **held-out validation** — known entries you *pretend* are missing, then score
  the imputation against the truth (this is `holdout_select`);
- **unobserved grid points** — the model is defined on every grid point even
  where you never had data.

All three are just "not in `M`."

**In CVXPY, the mask is boolean indexing.** With `mask = ~np.isnan(y)` (`True`
at observed entries), the consistency constraint is imposed only there by
indexing the summed expression with the boolean array:

    constraints.append(y[mask] == total[mask])

Indexing a CVXPY expression with a NumPy boolean array selects those rows, so the
boolean index *implements* `M` without ever materializing the selector matrix. The built
problem returns this array as `built["mask"]`, and a hand-written masked term
uses the same idiom.

## What a component is

Each component `k` has a cost function

    phi_k(x) = ell_k(x) + I_k(x)

a **penalty** `ell_k` (a convex function — small where the component is
plausible) plus an **indicator** `I_k` (zero on a feasible set, `+infinity`
off it — i.e. a hard constraint). A component may use one, the other, or both:
a smooth trend is a pure penalty; a nonnegativity component is a pure
indicator; the soiling example below uses both.

This maps exactly onto the code. A `Component` carries a `build` callable:

    build(T) -> (expr, loss, constraints)

- `expr` is the component's CVXPY variable/expression (the `x`),
- `loss` is `ell_k` (a scalar CVXPY expression, or 0),
- `constraints` is `I_k` (a list of CVXPY constraints, possibly empty).

A catalog builder like `smooth_trend(...)` *returns* a `Component`; a
hand-written `build` produces the identical object. They are the same thing —
the catalog is a set of worked `build` functions, not a privileged mechanism.
`bounded(inner, ...)` and `nonneg(inner)` are wrappers: they take a component
and add an indicator to it.

For x0, `ell_0` is the data-fidelity loss and `I_0` is empty (the residual is
unconstrained — full domain).

## Loss as (often improper) prior

The penalty `phi_k` encodes a **prior belief** about the component's structure:
smaller cost = more plausible. This is a useful and largely correct intuition —
an `l2` penalty corresponds to a Gaussian belief, `l1` to a Laplace one, an
indicator to a hard prior — and it is what lets you translate "I believe the
trend is smooth" directly into a cost.

But be honest about the correspondence: **many of the most useful component
classes are improper priors** in the strict Bayesian sense — the implied
density does not integrate to a finite value, so it is not a normalizable
distribution at all. The workhorse mean-square-small-of-the-first-difference
(the smooth-trend penalty) is exactly this: its "density" has a divergent
integral, no proper Bayesian reading, and yet it is one of the most valuable
priors in practice. So: treat "loss ≈ prior" as the right *intuition* for
choosing a cost, not as a claim that every model is a literal Bayesian
posterior. An improper prior is a *direction* to steer in (penalize roughness),
not a fully specified distribution — which is the point.

## DCP is the verifiable target

Every decomposition is a convex program, and it must be **DCP** — disciplined
convex programming, the rule system CVXPY uses to certify convexity by
construction. This is what makes *generating* components safe: you do not have
to prove your composed cost is convex by hand, you construct it from DCP-valid
atoms and let CVXPY check.

    out = solve(built, verify_dcp=True)   # the default

`solve` verifies DCP before handing the problem to the solver, and refuses a
non-convex model rather than returning a meaningless "solution." The discipline
is therefore:

1. **construct** the component from CVXPY atoms;
2. **verify** — let `problem.is_dcp()` / `verify_dcp` confirm convexity;
3. **trust** the solve.

**You can verify pieces in isolation, not just the whole problem.** Every CVXPY
`Expression` carries a curvature and a sign, and `is_dcp()` is defined on
expressions, constraints, and objectives — not only on the assembled problem. So
you can check a component's cost *before* it goes into `make_problem`:

    loss = w_d * cp.norm(cp.neg(cp.diff(x)), 2) + w_v * cp.norm1(x)
    loss.curvature      # 'CONVEX'   (also 'AFFINE', 'CONCAVE', 'CONSTANT', 'UNKNOWN')
    loss.sign           # 'NONNEGATIVE'
    loss.is_convex()    # True
    loss.is_dcp()       # True
    (x <= 0).is_dcp()   # True  -- constraints check too

For a valid penalty you want the `loss` to be `convex` (an objective is
`Minimize(convex)`); the component's `expr` itself is typically `affine` (a bare
variable). This lets you localise a DCP failure to the offending term instead of
getting one opaque rejection from the whole model.

Two things to know about the analysis. It is **sound but conservative**: a
"correct" curvature is always right, but it can decline to certify something that
is convex in truth if it is not written in a DCP-composable form. The classic
case is `cp.sqrt(1 + cp.square(x))` — genuinely convex, but not certified (CVXPY
1.9.2 labels it `QUASICONVEX`, `is_dcp() == False`); rewritten as
`cp.norm(cp.hstack([1, x]), 2)` it certifies as `CONVEX`. When a term you believe
is convex won't verify, the fix is usually to re-express it, not to abandon it.
And curvature nests: constant is affine, and affine is both convex and concave.

Never reason your way to "this must be convex" and ship it. Construct it and
check. For the per-atom rules (which atoms are convex/concave, how curvature and
monotonicity compose — e.g. whether `cp.log(x)` is usable where you want it),
defer to the CVXPY documentation at cvxpy.org; this skill does not restate them.
The default solver is `"CLARABEL"` (open-source); it is always overridable via
`solve(..., solver=...)`.

## Composing a bespoke component: a worked example

The point of the substrate is that you compose `phi_k = ell_k + I_k` to match a
belief the catalog does not cover. Consider a **soiling-and-washing** signal: a
quantity that drifts *down* slowly (soiling accumulates) and *recovers* in sharp
jumps back to a baseline of zero (a washing) — an inverted sawtooth. No catalog
entry is this. Compose it:

    x = cp.Variable(T)
    loss = w_d * cp.norm(cp.neg(cp.diff(x)), 2) + w_v * cp.norm1(x)
    cons = [x <= 0]

Reading the pieces as `ell + I`:

- `w_d * cp.norm(cp.neg(cp.diff(x)), 2)` — an **unsquared** L2 (group-lasso)
  penalty on the *negative* first differences: it costs downward steps, so the
  decline is slow and gradual. Upward steps are unpenalized, so recoveries snap
  back freely.
- `w_v * cp.norm1(x)` with **w_v tiny (~1e-6)** — a whisper of L1 that pins the
  level at the zero baseline. It is a tiebreaker, not a shaping force: without
  it the level is not uniquely determined; too much of it flattens the
  component entirely.
- `[x <= 0]` — the indicator `I`: the signal is a loss relative to baseline, so
  it is constrained non-positive.

This is DCP-valid (verify, don't assert), and with coherent weights it recovers
the sawtooth: slow penalized declines, sharp free recoveries to zero.

**These composed costs are tuning-sensitive, and they fail silently.** Too heavy
a pin (`w_v`) flattens the component; too heavy a drift penalty (`w_d`) erases
the soiling — and *both degenerate fits still return `optimal` and pass DCP*.
Nothing in the solver status tells you the component is wrong. You judge it by
**looking at the recovered component** (does it show the sawtooth?), not by a
fit score — a structural (Tier-3) quantity in the sense of
[model-specification.md](model-specification.md). See
[component-catalog.md](component-catalog.md) for the catalog of ready-made
`ell`/`I` patterns, and the note on which convex relaxations stand in for the
non-convex classes this skill excludes.
