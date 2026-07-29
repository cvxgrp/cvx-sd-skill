# Implementation

Use this reference after exploration has produced a model specification. The
implementation job is to make that specification reproducible: build the same
problem from the same inputs, expose deliberate parameters, verify DCP, choose
the solver explicitly when needed, and test the solved outputs.

## Repeated solves

Do not rebuild an identical CVXPY problem when only numeric weights change.
Represent changing weights with `cp.Parameter`, then re-solve the same problem.
This is the DPP-friendly pattern for weight scans and algorithms whose next
solve depends on the previous solution. Warm-start compatible solvers when
available, record every parameter value, and test convergence or stability
across the sequence.

The decision to use IRL1 or relax-round-polish is part of model formulation,
not implementation. See [component-catalog.md](component-catalog.md) for their
triggers and tradeoffs. Once selected, implement their repeated solves through
the same parameterized path.
