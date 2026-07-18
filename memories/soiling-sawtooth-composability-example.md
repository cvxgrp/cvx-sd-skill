---
name: soiling-sawtooth-composability-example
description: Verified worked example of composing a bespoke convex component (soiling/washing inverted sawtooth); the canonical SKILL.md composability example and the deep-dive owed to formulation.md + model-specification.md.
metadata:
  type: project
---

The composability example in `SKILL.md` layer 5 ("Formulate: compose, don't
shop") is a **soiling-and-washing inverted sawtooth** component. It is the
canonical illustration that *a component cost = a sum of convex terms + a
feasible set, stacked freely* — and it is NOT a catalog entry. The full,
verified walk-through is owed to `reference/formulation.md`, and its
tuning-sensitivity is owed to `reference/model-specification.md` (Tier 3). This
memory holds the verified detail so a future session can write those without
re-deriving.

## The component (exactly as it goes in the skill)

```python
x = cp.Variable(T)
loss = w_d * cp.norm(cp.neg(cp.diff(x)), 2) + w_v * cp.norm1(x)  # w_v tiny, ~1e-6
cons = [x <= 0]
```

**Semantics (soiling accumulation punctuated by washings):** a non-positive
signal (loss relative to a clean baseline of 0) that **drifts down slowly**
(soiling) and **recovers in sharp jumps back to 0** (washes). Shape = inverted
sawtooth.

**Term-by-term — which penalty governs what (this is the part that is easy to
get backwards):**
- `w_d * cp.norm(cp.neg(cp.diff(x)), 2)` — UNSQUARED L2 norm (group-lasso
  flavor, NOT `sum_squares`) on the **negative** first differences. This
  **penalizes the downward drift** — going down costs. `cp.diff(x)` is
  `x[t+1]-x[t]`, so `cp.neg(cp.diff(x))` picks out the down-steps.
- Upward steps are **completely unpenalized** — so recoveries/washes snap back
  to baseline freely and sharply. (Big recoveries must be free; that is what
  makes the sawtooth teeth.)
- `w_v * cp.norm1(x)` with **w_v tiny (~1e-6)** — the L1-on-values is a
  **tiebreaker that pins the level at 0**, not a shaping force. Group-lasso on
  neg-diffs alone does not uniquely pin the absolute level (a flat direction
  remains), so a whisper of L1 anchors baseline. Too large and it flattens the
  whole component.

## Verified coherent weights & the two silent failure modes

Synthetic truth: linear soiling drift (slope ~-0.03/step) with washes (reset to
0) every 30 steps, T=120, light noise; residual x1 default. Results (all
`is_dcp() == True`, all `status == optimal`):

| w_d | w_v  | outcome | corr to truth | ups | downs |
|-----|------|---------|--------------|-----|-------|
| 1.0 | 1e-2 | DEGENERATE: flat at 0 (L1 too strong) | 0.735 (spurious) | 0 | 0 |
| 1.0 | 1e-6 | DEGENERATE: no soiling, drift over-penalized | ~0.00 | 2 | 0 |
| 0.3 | 1e-6 | COHERENT sawtooth | 0.995 | 3 | 116 |
| 0.1 | 1e-6 | COHERENT sawtooth | 0.997 | 3 | 116 |

Coherent recipe: **w_d small (~0.1-0.3), w_v ~1e-6.** The coherent fits recover
the 3 washes at exactly t=30/60/90 (the `ups`), drift down on nearly every
other step (`downs`), stay non-positive. ASCII sparkline of the recovered
component at w_d=0.1 (higher glyph = closer to 0 baseline):
`*++=--:.. *++=--:.. *++=-::.  *++=-::.` — four visible teeth.

## The lessons this example teaches (why it's load-bearing)

1. **Composability is the thesis of the skill** ("not a component library"): two
   penalty terms + a hard constraint → one bespoke component that no catalog
   builder provides.
2. **DCP is checked, not asserted.** In-session I first got the *narration*
   backwards (mislabeled which behavior the terms produce) AND chose degenerate
   weights. The code was right; recall/tuning was wrong. Doctrine for the prose:
   construct → `is_dcp()` → verify the *shape by looking* → trust. Never reason
   "this must be convex / must produce a sawtooth" and ship it. The
   `cp.norm(cp.neg(...), 2)` composition DID verify DCP-True (norm of a
   nonnegative arg is DCP-valid), but the honest move was to run it, not to
   argue it.
3. **This is the archetypal Tier-3 (structural) hyperparameter case.** Both
   failure modes still return `optimal` + DCP — they fail SILENTLY. Only
   *looking at the component* (does it show teeth?) distinguishes coherent from
   degenerate. A holdout/fit score would NOT reliably catch the flat-degenerate
   case and would optimize the wrong objective. This is exactly why Tier-3 says
   "tune by inspecting the component, do NOT holdout-tune." Strong forward-link
   to [[tier3-structural-tuning-examples]] — this soiling/washing case is a
   concrete instance of the accumulation/removal Tier-3 archetype the user has
   promised more automated methods for.

## For formulation.md (owed)

Use the table + sparkline above verbatim as the worked walk-through: show the
coherent fit, then the two silent degeneracies, and land the "judge by looking"
point. Re-run to regenerate a real plot (`plot_decomposition` per-role panel)
rather than the ASCII sparkline when writing the actual reference. The
reproduction script pattern: synthetic sawtooth truth + the component above +
`make_problem`/`solve`, sweep (w_d, w_v).
