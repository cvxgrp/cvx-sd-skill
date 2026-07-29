# Implementation

Implementation begins with a specified model: components, forms, transforms,
time grid, residual loss, and hyperparameter-selection rules are already
decided. The job is to make that specification reproducible and robust.

## Decide what ships

Ask first: **is there a representative tuning dataset separate from production
data?** The answer determines the architecture.

| Specification result | Production behavior |
|---|---|
| Tier-1 value | Ship as a constant |
| Tier-2 value selected on representative tuning data | Freeze and ship as a constant |
| Tier-2 rule without representative tuning data | Run the selector on each new batch |
| Tier-3 structural rule | Run the rule whenever new data may change the feature |

The deliverable is therefore often a tune-then-solve pipeline:

```python
tuned = tune(y)
built = build(y, tuned)
out = solve(built)
```

Keep these stages separate. `tune` returns explicit values and evidence;
`build` deterministically constructs the selected model; `solve` verifies DCP
and computes the result. Persist the tuned configuration with the output.

## Reuse a fixed problem with Parameters

When only numeric weights change, represent them with `cp.Parameter` and reuse
the same CVXPY problem:

```python
weight = cp.Parameter(nonneg=True, value=1.0)
built = make_problem(y, [smooth_trend(weight, role="trend")])
assert built["problem"].is_dcp(dpp=True)

outs = []
for value in np.logspace(-1, 2, 8):
    weight.value = value
    outs.append(solve(built, warm_start=True))
```

DPP—disciplined parametrized programming—lets CVXPY reuse canonicalization
work. Warm starts may further reduce solve time when the chosen solver supports
them. DPP improves execution; it does not choose the scan range or tuning tier.

Parameter reuse requires the same data, length, mask, variables, constraints,
and component structure. Rebuild when any of those change. In particular:

- holdout selection changes the observed mask;
- bootstrap changes the signal values;
- expanding-window stability changes the data and often `T`;
- adding, removing, or changing a component form changes the problem graph.

IRL1 can reuse a problem when its entrywise reweighting is expressed as a
vector `cp.Parameter`. Relax-round-polish normally uses at least two problem
states because the polish fixes rounded values. The decision to use either
method belongs to [component-catalog.md](component-catalog.md); this section
only governs execution.

## Make repeated solves robust

A production tuning loop must treat individual candidates as fallible:

1. Catch solver failures per candidate rather than aborting the scan.
2. Record the requested value, solver, status, objective or score, and relevant
   structural diagnostics.
3. Reject non-finite component values and scores.
4. Define the fallback before deployment.

For a Tier-2 scan, do not claim a precise optimum when candidate scores are
nearly tied. A useful rule is to choose the **largest weight whose holdout error
is no more than `delta` above the minimum**:

```python
finite = np.isfinite(errors)
if not finite.any():
    raise RuntimeError("all tuning candidates failed")
best = np.min(errors[finite])
eligible = finite & (errors <= best + delta)
selected_weight = np.max(weights[eligible])
```

This selects the most regularized near-optimal fit under the library convention
that larger weights impose stronger regularization. Define `delta` before
examining the scan, in absolute error units or as a documented relative
tolerance. If every candidate fails, return an explicit failure or a
deliberately chosen safe configuration—never silently select an arbitrary
index.

For a Tier-3 scan, apply the structural acceptance rule to full-data fits and
preserve its null path. If no candidate is admissible, return the specified
null or most-regularized model rather than forcing a feature. See
[model-specification.md](model-specification.md).

## Solver policy

`solve` defaults to CLARABEL and accepts any keyword supported by
`problem.solve`, including `warm_start=True`. Keep the solver user-configurable.
Pin solver-specific tolerances only when the application has evidence for them,
and record non-default settings with the result.

Treat `optimal_inaccurate` as usable only under an application policy: inspect
residual consistency and the extracted quantity before accepting it. Do not
switch solvers merely to hide a malformed or badly scaled model; verify DCP and
inspect scaling first.

## Test the contract

Tests should verify the model’s meaning, not one solver’s exact floating-point
path:

- the problem is DCP, and parameterized scans are DPP;
- roles and auxiliary keys are stable;
- masked entries remain unconstrained by observations;
- reconstruction matches observed data within tolerance;
- expected signs, bounds, monotonicity, or event rules hold;
- tuning has a deterministic flat-score, null, and all-failed policy;
- fixed seeds make resampling reproducible.

Use tolerances appropriate to the solver and compare extracted quantities or
structural invariants instead of entire arrays when exact arrays are not the
contract.

## Uncertainty comes last

Run `bootstrap_ci` only after tuning has selected the operating model for the
dataset. Bootstrapping inside a tuning loop mixes selection variation with
final-model uncertainty and multiplies an already expensive repeated solve.

Bootstrap and expanding-window analysis rebuild by design because their data
change. Supply a `build_fn(y) -> built` and an explicit extractor for the
quantity of interest. Choose the required bootstrap `block_size` from residual
dependence; there is no domain-safe default.
