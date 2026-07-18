---
name: meta-pattern-convex-substrate-skill-family
description: PARKED (do not pursue in V1) — cvx-sd is the first instance of a repeatable pattern for domain-specific convex-modeling skills (agent + convex substrate + composable atoms). Generalizes to Markowitz++, OPF, MPC, forecasting. Ship one exemplar first.
metadata:
  type: project
---

**PARKED — explicitly do NOT pursue in V1. Primarily a note on project
trajectory.**

The whole skill-design pattern here generalizes: give an agent the **substrate /
composable atoms / footguns** of a canonical convex formulation, and let
agent+analyst **compose beyond** the canonical spec (compose-don't-shop +
DCP-as-verifiable-target + judgment-layer-over-thin-correct-core). This works
for any convex domain with a canonical formulation:
- Markowitz++ (portfolio) — analysts develop formulations beyond the paper's
  spec, treating its components as composable atoms;
- optimal power flow, robust MPC, probabilistic forecasting (the abstract's
  "continuous line of work").

Each sibling = the SAME architecture, DIFFERENT substrate. So **cvx-sd is really
the first worked instance of a repeatable pattern for domain-specific
convex-modeling skills** — a meta-thesis sitting ABOVE the abstract's three
claims: *the agent + convex-substrate pairing is itself the reusable artifact.*

**Why parked (the discipline):** out of scope for cvx-sd; premature (cvx-sd
isn't even drafted through SKILL.md layer 6); and a scope-creep hazard — the
surest way to NOT ship a sharp cvx-sd is to start architecting the family before
the first member exists. The pattern earns generalization only by having a
working exemplar; cvx-sd being genuinely good is what makes the meta-pattern
credible. Ship the proof, then the family.

**Where it may appear:** at most a single forward-looking clause in
`philosophy.md` (possibly not even in V1). Otherwise just trajectory.

Ties to [[graceful-degradation-model-capability]],
[[fourier-parameterized-covariance-roadmap]],
[[binding-constraint-hyperparameter-specification]],
[[convex-not-brittle-underspecification]], and the abstract's line of work.
