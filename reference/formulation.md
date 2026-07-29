# Formulation: the substrate

This is the layer beneath the component catalog: the decomposition, the mask,
the component cost, and the DCP check that makes composition safe.

## The decomposition

A decomposition writes the observed signal as a sum of `K` estimated
components:

    y = x0 + x1 + ... + xK

`y` is a 1-D signal on a regular grid and may contain missing values. The
components are real-valued on the whole grid, so their sum estimates `y` even
where it is missing.

**x0 is always the data-fidelity term** (the residual), with its loss selected
through `make_problem(..., residual_loss=...)`. Append structural components as
`x1 ... xK`; read solved components by role (`"trend"`, `"seasonal"`), never by
index. The residual is keyed `"residual"`.

## The mask: missing data as a linear operator

Let `M` select the observed entries. Apply it to the consistency constraint:

    M y = M (x0 + x1 + ... + xK)

i.e. require the components to sum to `y` only where `y` is known. One mechanism
therefore handles:

- **missing data** — entries absent in the raw signal;
- **held-out validation** — known entries you *pretend* are missing, then score
  the imputation against the truth (this is `holdout_select`);
- **unobserved grid points** — the model is defined on every grid point even
  where you never had data.

All three mean “not in `M`.” In CVXPY, implement `M` with boolean indexing:

```python
mask = ~np.isnan(y)
constraints.append(y[mask] == total[mask])
```

`make_problem` returns the boolean array as `built["mask"]`; no selector matrix
is materialized.

## What a component is

Each component `k` has a cost function

    phi_k(x) = ell_k(x) + I_k(x)

a penalty `ell_k`—small where the component is plausible—plus an indicator
`I_k`, which is zero on a feasible set and `+infinity` outside it. A component
may use either or both: smoothness is a penalty; nonnegativity is an indicator.

This maps exactly onto the code. A `Component` carries a `build` callable:

    build(T) -> (expr, loss, constraints)

- `expr`: the component's CVXPY variable or expression;
- `loss`: `ell_k`, a scalar CVXPY expression or zero;
- `constraints`: `I_k`, a possibly empty list of CVXPY constraints.

A catalog builder and a hand-written `build` produce the same `Component`
interface. Wrappers such as `bounded` and `nonneg` add indicators without
changing the inner penalty. For x0, `ell_0` is the data-fidelity loss and
`I_0` is empty.

## Loss as (often improper) prior

A component cost encodes a prior belief: smaller cost means more plausible.
L2 suggests a Gaussian belief, L1 a Laplace belief, and an indicator a hard
prior. Use this as a formulation intuition, not a claim that every objective is
a literal Bayesian posterior. Difference penalties commonly imply improper,
non-normalizable priors while remaining useful directions such as “prefer less
roughness.”

## DCP is the verifiable target

Every decomposition must satisfy **DCP**, the rule system CVXPY uses to certify
convexity by construction. Compose from DCP-valid atoms and let CVXPY check:

    out = solve(built, verify_dcp=True)   # the default

`solve` refuses a non-DCP model before calling the solver. The discipline is:

1. **construct** the component from CVXPY atoms;
2. **verify** — let `problem.is_dcp()` / `verify_dcp` confirm convexity;
3. **solve** only after verification.

Check a component before adding it to the full problem:

    loss = w_d * cp.norm(cp.neg(cp.diff(x)), 2) + w_v * cp.norm1(x)
    loss.curvature      # 'CONVEX'
    loss.sign           # 'NONNEGATIVE'
    loss.is_convex()    # True
    loss.is_dcp()       # True
    (x <= 0).is_dcp()   # True  -- constraints check too

The loss must be convex and the component expression is normally affine.
Piecewise checks localize failures. DCP analysis is sound but conservative: it
may reject a convex expression written in a form its rules cannot certify. For
example, rewrite `cp.sqrt(1 + cp.square(x))` as
`cp.norm(cp.hstack([1, x]), 2)`. If a known-convex term fails DCP, first seek an
equivalent DCP form; never bypass verification. Defer per-atom composition rules
to the CVXPY documentation.

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
- `w_v * cp.norm1(x)` with **w_v small (~1e-4)** — a light L1 that pins the
  level at the zero baseline. It is a tiebreaker, not a shaping force: too
  *little* and the level walks off (it does not return to zero after a wash);
  too *much* and it drags the whole trough up toward zero, erasing the soiling
  depth. The working value is the *smallest* one that stops the walk-off.
- `[x <= 0]` — the indicator `I`: the signal is a loss relative to baseline, so
  it is constrained non-positive.

Verify DCP, then tune in sequence: set `w_v` to the smallest value that returns
the level to zero after each recovery (~`1e-4` in the worked case), then tune
`w_d` for a coherent gradual decline (~`1e-2`). These scales are illustrative,
not defaults.

Bad tuning still returns `optimal`: too much pinning flattens the trough, while
too much drift penalty erases it. Judge the recovered component, not solver
status or fit score alone. This is a structural choice in the sense of
[model-specification.md](model-specification.md). See
[component-catalog.md](component-catalog.md) for ready-made `ell + I` patterns.
