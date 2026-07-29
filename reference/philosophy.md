# Philosophy and operating boundary

Convexity is the operating constraint that makes generated decompositions
verifiable. A DCP-valid single solve has a global-optimality certificate; it
can be composed from independently checked components without relying on a
fragile initialization or hidden search path.

That certificate says the stated problem was solved; it does not say the stated
model is uniquely right. Several decompositions may reconstruct the same data
nearly equally well. Losses and constraints encode the beliefs that distinguish
them, and those beliefs still require domain judgment.

## Under-specify with structure

Convex modeling is sometimes mistaken for specifying every detail in advance.
Signal decomposition does the opposite. Specify a structural direction—
smoothness, sparsity, monotonicity, boundedness—not the value of every sample.
Let the solve determine the remaining detail.

These models routinely contain more component values than observations. That
underdetermination is useful: the data alone cannot assign variation to trend,
periodic, sparse, and exogenous terms, while their penalties and constraints
steer the assignment toward interpretable shapes. Structure guides surplus
degrees of freedom; it does not eliminate them.

Brittleness comes from specifying more strongly than the evidence supports. An
oversized weight can flatten a real component, and an unnecessary penalty can
remove behavior already allowed by a constraint. Both models may remain DCP
and return `optimal`. Use the weakest structure that expresses the belief, scan
nearby settings, and inspect the solved components.

## The binding constraint is interaction

Raw variable count is usually not the hard part. Each added component introduces
hyperparameters that renegotiate the boundary with every other component
competing for the same signal. A trend and a long-period term can exchange
low-frequency variation; with more components, those pairwise confounds become
a web.

Count unresolved interactions, not components. A large model with two
holdout-tunable weights may be easier to specify than a small model with five
coupled structural decisions. This is why the workflow builds largest sources
first and classifies knobs into tiers: Tier 1 freezes insensitive choices, Tier
3 removes structural choices from reconstruction-based search, and only Tier 2
remains in the coupled holdout problem. See
[model-specification.md](model-specification.md).

## The operating boundary

Use one convex solve for convex losses, constraints, and relaxations. Common
examples include smoothness, monotonicity, bounds, robust fidelity, and L1
sparsity.

Use a short sequence of convex solves when the sequence itself is an acceptable
heuristic:

- iteratively reweighted L1 for a sharper approximation to “few nonzeros”;
- relax-round-polish for Boolean or finite-set values.

Each subproblem is solved exactly, but the sequence is local search and does
not restore a global guarantee. Inspect stability and the resulting component,
not only solver status.

CVXPY can also formulate mixed-integer convex programs with Boolean and integer
variables for compatible solvers. These models are not outside the modeling
language, but they leave the scalable single-convex-solve regime: exact search
can become intractable as the time horizon and number of discrete variables
grow. Solver availability also becomes part of the design.

The practical rule is: “few,” “rare,” and “mostly zero” usually admit an
L1-flavored convex relaxation. For discrete-valued structure, compare three
options early:

- solve the exact mixed-integer formulation when its scale is practical or its
  guarantee is required;
- use relax-round-polish when a tractable heuristic is acceptable;
- reformulate the state description when neither is suitable.

Markov or regime-switching dynamics may still call for specialized methods,
especially over long horizons. The boundary is therefore computational
tractability and the guarantee the application needs—not whether CVXPY can
express an integer variable. Outside this skill does not necessarily mean
difficult or approximate: some discrete signal classes have efficient exact
specialized operators that do not fit the generic CVXPY/DCP substrate.

## Division of responsibility

This skill occupies the modeling layer:

- the user owns domain semantics, acceptable sensitivity, and the meaning of
  extracted quantities;
- this skill translates those beliefs into components, specification rules,
  validation, and reporting;
- CVXPY and the selected solver own canonicalization and numerical solution.

These are boundaries, not walls. Surface adjacent concerns such as drift,
re-tuning cadence, or solver availability when they affect the model, but do
not silently invent the user’s domain policy or production architecture.
