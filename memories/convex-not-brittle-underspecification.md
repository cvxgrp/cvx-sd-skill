---
name: convex-not-brittle-underspecification
description: The philosophy.md opening argument — convex optimization is widely MISperceived as rigid/over-specified/brittle; the truth is the opposite (under-specified by design, p>n, structure steers rather than rigidifies). Brittleness is a symptom of over-specifying. Markowitz++ corroborates.
metadata:
  type: project
---

**Likely the opening argument of `philosophy.md`, and the thesis's most
persuasive single move for a skeptical reader** — you don't ask them to like
convex optimization, you show them they've had it backwards.

**The misconception (widespread, incl. among sophisticated ML people):** convex
optimization is the brittle, over-engineered, hand-specified past — you must
specify every term in advance, and it breaks when reality departs from your
rigid spec. Convex = rigid = brittle = know-everything-up-front. Cast as the
opposite of "ride the dynamics."

**The reality (exactly backwards):** a well-posed convex program IS "specify
only somewhat, then ride the dynamics" ([[beer-eno-epigraph]]). You don't
specify the trend's values — you specify *smoothness* and ride the solve. You
don't specify where the breakpoint is — you specify *sparsity of kinks* and ride
the L1 dynamics to find it. The penalty IS the "somewhat"; the solve IS the
dynamics. You UNDER-specify with structure and let the optimization carry you.

**The p>n point is the technical PROOF, not just rhetoric.** "Convex
optimization is over-specified" is a testable claim and it's false on the
numbers: over-specified = fewer effective DOF than the data demands (rigid). But
SD problems are routinely **p > n** — MORE parameters than data, the signature
of an *under*-determined system, the opposite of over-specification. What makes
p>n work is the structure: penalties/constraints don't rigidify, they *steer*
an under-determined problem toward the solution you want. Structure = the
direction you want to go; the surplus DOF = the dynamics you ride. **The
technical fact (p>n, regularized) and the philosophical stance (specify
somewhat, ride the dynamics) are the same fact stated two ways.**

**Brittleness is a symptom of OVER-specifying — a misuse, not the nature of the
tool.** The degenerate models produced in-session (too-heavy penalty crushing
the fit — the degenerate soiling-sawtooth,
[[soiling-sawtooth-composability-example]]) came from *failing* to "specify only
somewhat." The tuning hierarchy explicitly teaches you NOT to over-specify
(Tier 1: don't pin down what you don't need to). Brittleness = failing to
follow Beer.

**This reframes abstract-claim-2** ("more signal from less data") from a modest
efficiency claim into a **correction of a category error**: the field thinks
convex optimization is brittle over-specification when it is in fact the
disciplined art of UNDER-specification — encoding just enough belief to steer an
under-determined system and riding the rest.

**Markowitz++ corroboration (philosophy.md, ONE sentence + cite only).** Boyd's
recent "Markowitz at 50" / Markowitz++ is a strong independent witness: classic
mean-variance portfolio optimization is THE canonical "convex is brittle"
example (tiny input changes swing the optimal portfolio wildly). Markowitz++
robustifies it into a stable, practical tool by exactly this route —
structured/robust under-specification rather than fragile point-specification —
and it's by the framework's own author (Boyd), reinforcing the abstract's
"continuous line of work." **STRICTLY a corroborating citation; do NOT import
any portfolio-optimization content into the skill's operational surface** —
different problem, pulling it in is scope creep against the operating-band
discipline. Verify exact title/venue/year when tools available.

→ `philosophy.md` opening (epigraph → misconception → p>n rebuttal → tuning
hierarchy as the discipline of correct under-specification → Markowitz++ as
witness). Ties to [[binding-constraint-hyperparameter-specification]] and
[[beer-eno-epigraph]].
