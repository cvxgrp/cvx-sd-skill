# Philosophy and operating boundary

Convexity is the operating constraint that makes generated decompositions
verifiable. A DCP-valid single solve has a global-optimality certificate; it
can be composed from independently checked components without relying on a
fragile initialization or hidden search path.

That guarantee does not make the model uniquely correct. Signal decomposition
is under-specified: several component sets may reconstruct the same data nearly
equally well. Losses and constraints encode the beliefs that distinguish those
models, and structural choices still require domain judgment.

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
