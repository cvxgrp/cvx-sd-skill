---
name: binding-constraint-hyperparameter-specification
description: The deepest framing of the whole project — the binding constraint in convex SD is NOT statistics or runtime (both solved) but the number and INTERACTION of hyperparameters, which grows super-linearly with component count. Justifies the entire specification/tuning apparatus.
metadata:
  type: project
---

**The single deepest framing surfaced in this project. Likely a central
argument of `philosophy.md`.**

The classical worries are SOLVED in this domain:
- **Overparameterization / statistical ill-specification** — not a problem. SD
  problems are routinely **p > n** (more parameters than data); the structure
  (penalties, constraints, priors) regularizes. This is the point (abstract
  claim 2), not a flaw.
- **Problem size / runtime** — not a problem. 10k+ variables on a laptop, more
  with standard tricks. "Below the band — CVXPY owns it."

**What remains as the binding constraint: hyperparameter specification —
specifically the super-linear growth of hyperparameter INTERACTION as components
are added.** Each component drags in hyperparameters, and they *couple*: every
weight renegotiates the boundary with every other component competing for
overlapping signal. The trend↔seasonal confound is the 2-component case; at many
components it's a web, and weights stop being independent knobs — moving one
changes the *meaning* of the others.

**This is the deepest justification for the entire specification/tuning
apparatus.** The Tier 1/2/3 hierarchy's real job is **shrinking the dimension of
the coupled specification problem**:
- Tier 1 (set by magnitude, fix, forget) — *removes* knobs from the interacting
  set.
- Tier 3 (judge by looking, don't holdout) — *removes* knobs from the
  score-based search (they'd corrupt it — both failure modes still return
  optimal+DCP; see [[soiling-sawtooth-composability-example]]).
- Only Tier 2 (materially contributes, holdout-tunable) stays in the coupled
  optimization. THAT count is the one that matters.

**Discipline it implies: parsimony of components justified by contribution.**
Compose-don't-shop is powerful but each component taxes you in interacting
knobs. Add a component only if it earns its place (check share of
reconstruction energy via `format_report`); a component that barely moves the
reconstruction still muddies the others and its "optimum" is noise (can't
holdout-tune a knob on a component that doesn't move the score).

**Greedy / hierarchical specification order** (dominant component first — its
weight is Tier-1-numb *because* it dominates — then the next against that fixed
backdrop) is NOT just a proposed procedure: it is **what marimo exploration
naturally produces** when building append-only. **Append-only construction order
= greedy tier-classification order = widget interaction order** — three views of
one process (see [[exploration-tier-classification-marimo]]).

**"Is 20 components reasonable?" — the domain-coherent answer:** refuse the
premise. Component count is not the cost function (p>n is fine statistically;
20 components solves trivially). Count instead the number of **interacting,
holdout-tunable (Tier-2)** knobs — that's what grows the specification problem
super-linearly. 20 components with 18 Tier-1 + 2 Tier-2 knobs is healthy; 20
with 12 mutually-interacting Tier-2/3 knobs is a nightmare regardless of fit.
Tier-classify the knobs; build greedily so you classify each as you add it; if
you have many interacting Tier-2 knobs, simplify.

→ `philosophy.md` (central-difficulty framing) + `model-specification.md`
(taxonomy-as-dimensionality-reduction + the greedy ordering procedure). Ties to
[[tier3-structural-tuning-examples]], [[graceful-degradation-model-capability]],
[[convex-not-brittle-underspecification]].
