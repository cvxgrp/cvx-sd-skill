---
name: exploration-tier-classification-marimo
description: The exploration work phase (in marimo) is where the user classifies hyperparameters into Tiers 1/2/3 by feel; this is the dimensionality reduction that makes specification tractable. Sharpens the SKILL.md explore bullet and marimo.md.
metadata:
  type: project
---

The skill's exploration job is to make **judgment-by-looking** fast and legible
— NOT to decide the model a priori. Sharper framing reached this session:

**Exploration in marimo is where the user classifies hyperparameters into Tiers
1/2/3 by feel:**
- slider is numb across an order of magnitude → **Tier 1** (set by magnitude,
  fix, forget).
- reconstruction / holdout score moves as you slide → **Tier 2**
  (holdout-tunable, materially contributes).
- component *shape* changes (kinks appear, sawtooth flattens, breakpoint jumps)
  but the fit score barely notices → **Tier 3** (structural, judge by looking,
  do NOT holdout-tune).

This tier-classification is the **dimensionality reduction** that makes the
coupled specification problem tractable (see [[binding-constraint-hyperparameter-specification]]).
The widget makes classification a matter of sliding-and-looking rather than
analysis — it is the **low-judgment on-ramp to the tuning hierarchy**, parallel
to the package being the low-context on-ramp to correctness (see
[[graceful-degradation-model-capability]]).

The greedy/hierarchical specification order (dominant component first — its
weight reads Tier-1-numb *because* it dominates — then the next against that
backdrop) is **what marimo exploration naturally produces** when you build
append-only: **append-only construction order = greedy tier-classification
order = widget interaction order**, three views of one process.

marimo tool names (reference detail, belongs in `marimo.md` not SKILL.md): the
skill is `marimo-team/marimo-pair`; related helpful set is `marimo-team/skills`.
Widget mapping: append-only components → dropdowns/radios; weights → sliders;
DPP tells you which knobs are instant (weight on fixed data, no rebuild) vs.
trigger a re-solve (change T or component set). Recommend (not "prefer")
marimo. "Recommend, don't depend" — core library stays marimo-free.

Ending a recommendation with clarifying questions back to the user is the skill
WORKING, not falling short (exploration is a dialogue with the data).

→ context-router explore bullet in `SKILL.md` (carry the *why*, not just
"recommend marimo"); `reference/marimo.md` spine.
