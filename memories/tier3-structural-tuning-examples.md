---
name: tier3-structural-tuning-examples
description: User has concrete examples of automated Tier-3 (component structure analysis) hyperparameter tuning to share in a later session; fill into the prose then.
metadata:
  type: project
---

The user has real, concrete examples of how to do **automated Tier-3
hyperparameter tuning** — the "component structure analysis" method — to
provide in a **later session**. This is a to-be-filled-in detail for the prose.

**Context:** Tier 3 (from the tuning hierarchy in
`plans/notes-vision-and-scope.md`) is for hyperparameters governing
**shape-valued / structurally-sensitive components** where *shape fidelity ≠
reconstruction fit*, so holdout tuning optimizes the wrong objective and
actively fails. These are tuned by **inspecting the component** (plots,
stability snapshots, domain checks) rather than by holdout score. Named
examples so far: **step changes** (pwl / change-point weight controls readiness
to admit a step) and **accumulated buildup/removal** (monotone / integral-type,
e.g. soiling accumulation + cleaning removal — judge whether the cumulative
curve matches physical reality).

**What's still missing (what the user will supply):** the hard "how to
automate 'look at the component and set it'" detail — e.g. step-detection
sensitivity, accumulation-rate estimation. Concrete automated structural
methods, not just "eyeball the plot."

**Where it lands in the prose:** `reference/model-specification.md` (the Tier
1/2/3 hierarchy), Tier-3 section. Until the user provides the examples, keep the
Tier-3 automated-method detail as a placeholder / "inspect the component"
qualitative guidance; do NOT invent the automated methods.

The `FUTURE:` line in `plans/notes-vision-and-scope.md` (Tier-3 subsection)
records the same intent inline; this memory is the durable cross-session
pointer. Related: [[prose-phase-resume-point]] (if written).
