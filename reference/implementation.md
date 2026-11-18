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

## Compilation cost is separate from convexity

DCP certifies the model's convexity under CVXPY's rules; it does not bound the
time or memory needed to compile the expression graph into solver data.
Compilation can dominate a solve, and sparse final solver matrices do not
guarantee sparse intermediate allocations. Check representative problem sizes
before committing to a repeated-solve architecture.

## Reuse a fixed problem with Parameters

Consider parameterization when the same model will be solved repeatedly with
different data or hyperparameters, not as a default for every model. For a
one-off solve, use fixed numerical values. Parameters and DPP caching can add
initial compilation work and memory use; the savings from repeated solves
must justify those costs.

Large hyperparameter sweeps and Monte Carlo studies with fixed problem
structure are potential use cases. When only a few solves are expected,
avoiding expensive caching is usually preferable. With many repeated solves,
faster reuse may justify the additional memory. The tradeoff depends on the
model, problem size, CVXPY version, and solver interface; measure it for the
intended workload.

For example, a repeated weight scan can represent the weight with
`cp.Parameter` and reuse the same CVXPY problem:

```python
weight = cp.Parameter(nonneg=True, value=1.0)
built = make_problem(y, [smooth_trend(weight, role="trend")])
assert built["problem"].is_dcp(dpp=True)

outs = []
for value in np.logspace(-1, 2, 8):
    weight.value = value
    outs.append(solve(built, warm_start=True))
```

There are four distinct kinds of reuse:

| Mechanism | What is reused | Conditions and limits |
| --- | --- | --- |
| Expression graph | Python variables, expressions, constraints, and the `Problem` | Update declared Parameters without rebuilding the graph; this alone promises no compilation or solver cache. |
| DPP caching | The mapping from Parameters to canonical problem data | A DPP-compliant model and a compatible compilation path; the first compilation constructs the map, subsequent calls update numerical data. |
| Solver workspace / factorization | Solver-side state or linear-algebra work | Depends on the solver, CVXPY interface, and which numerical data change; changing a matrix can invalidate a factorization even when the graph is unchanged. |
| Previous solution | An initial iterate for a later solve | Requires support from the solver/interface; requesting `warm_start=True` does not guarantee that an earlier solution is used or that a factorization is retained. |

CVXPY's [DPP tutorial](https://www.cvxpy.org/tutorial/dpp/index.html)
describes the parameter-map cache; its
[warm-start documentation](https://www.cvxpy.org/tutorial/solvers/index.html#warm-start)
describes solver-dependent reuse. DPP—disciplined parametrized programming—is
an execution mechanism, not a choice of scan range or tuning tier.

In this weight-only example, the signal data, length, masks, variables,
constraints, and component structure stay fixed. General CVXPY models can also
parameterize data while retaining the same problem structure, but
`make_problem` takes numerical `y`, not a signal Parameter. The package's
existing validation workflows rebuild by design:

- holdout selection changes the observed mask;
- bootstrap changes the signal values;
- expanding-window stability changes the data and often `T`;
- adding, removing, or changing a component form changes the problem graph.

IRL1 can reuse a problem when its entrywise reweighting is expressed as a
vector `cp.Parameter`. Relax-round-polish normally uses at least two problem
states because the polish fixes rounded values. The decision to use either
method belongs to [component-catalog.md](component-catalog.md); this section
only governs execution.

### Profile the compilation path

Measure graph construction, first compilation, numerical solution, and repeated
calls separately, including peak memory. Record the model size, parameter
shapes, CVXPY/solver versions, and effective options. Solver-reported solve time
alone is not end-to-end runtime. Compare original objectives, feasibility, and
extracted quantities when changing compilation routes; the same mathematical
model need not have identical numerical accuracy under different routes.

When profiling identifies DPP parameter-map construction as the bottleneck,
test `solve(built, ignore_dpp=True)` as a scoped alternative. CVXPY then treats
the problem as non-DPP, forgoing the DPP cache and repeating canonicalization
as parameter values change. The Python graph can still be reused; solver-side
reuse remains a separate, solver-dependent question. This does not disable
the wrapper's DCP check. Preserve explicit caller options rather than silently
overriding their compilation policy. See CVXPY's
[`ignore_dpp` option](https://www.cvxpy.org/tutorial/solvers/index.html#solve-method-options).

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
null rather than forcing a feature. That null may be the most-regularized fit
or a separately built model that omits the optional component. See
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
dataset. The result is conditional uncertainty for that frozen preprocessing,
model specification, tuning result, and extractor. It does not measure
end-to-end procedural uncertainty.

To target end-to-end procedural uncertainty, rerun preprocessing, tuning,
selection, fitting, and extraction inside each outer bootstrap replicate. That
is a different and substantially more expensive procedure; report it
separately rather than describing a final-model bootstrap as end-to-end.

Bootstrap and expanding-window analysis rebuild by design because their data
change. Supply a `build_fn(y) -> built` and an explicit extractor for the
quantity of interest. Choose the required bootstrap `block_size` from residual
dependence; there is no domain-safe default.
