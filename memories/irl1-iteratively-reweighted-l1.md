---
name: irl1-iteratively-reweighted-l1
description: Explicit trick to add — iteratively reweighted L1 (IRL1) as a convex route to L0/cardinality; the canonical "second problem depends on the first's output" pattern. Debiases surviving structure and kills spurious terms at once.
metadata:
  type: reference
---

**Iteratively reweighted L1 (IRL1)** — a named trick the skill should teach
explicitly. Plain L1 is a *biased* surrogate for L0 (cardinality): it shrinks
large true terms and leaves small spurious ones. IRL1 approximates L0 with a
**sequence of convex solves**, weights taken from the previous solution:

```python
# e.g. for a piecewise-linear (breakpoint) trend: sparse 2nd difference
w = np.ones(T - 2)
eps = 1e-3
for _ in range(3):                     # 2-3 iterations ~= L0
    x = cp.Variable(T)
    loss = weight * cp.norm1(cp.multiply(w, cp.diff(x, k=2)))
    # ... make_problem / solve ...
    d2 = np.abs(np.diff(x.value, n=2))
    w = 1.0 / (d2 + eps)               # reweight from THIS solve's output
```

Each iteration: an entry already ~0 gets a huge weight → crushed exactly to zero
(kills spurious near-kinks); a genuinely large entry gets a tiny weight → freed
from shrinkage (true break sharpens toward unbiased magnitude). So IRL1 does
**both** debiasing AND spurious-removal at once, and converges toward the L0
solution ("exactly a few, at full magnitude"). 2–3 reweightings usually suffice.
(Candès–Wakin–Boyd reweighting.)

**Why it matters here:**
- It is the **canonical "second problem depends on the first's output"**
  pattern — stage k+1's weights are computed from stage k's solution.
- **Generalizes to any sparse component**, not just breakpoints (sparse
  outliers, `sparse`, r-sparse-ish structure). → belongs in
  `component-catalog.md` + `formulation.md` as a general technique.
- It is the **better** answer to the convex-only boundary on "exactly one
  breakpoint at unknown location" (which is non-convex): IRL1 stays fully
  convex (every solve is convex) yet recovers most of what L0 would give.
  Correct the earlier plain-L1 breakpoint answer to lead toward IRL1.
- **DPP-safe:** the reweighting changes only per-entry *weights*; expressed as a
  `cp.Parameter`, the canonicalization is reused across iterations (a worked
  case of a Parameter-fast runtime scan). Cross-link `implementation.md`.
- Still Tier-3-flavored: judge the resulting breakpoints by looking; if the
  break location wanders across the iterations/weights, be skeptical it is a
  single true break.

**Breakpoint two-stage context** (where this arose): use the L1 relaxation as a
*finder* (locates the kink), human *looks* (Tier 3), then a clean
exactly-specified convex second problem *conditioned on* that location — e.g.
refit an unpenalized two-segment continuous PWL trend with hinge basis
`[1, t, (t-tau)_+]` to remove L1 bias. IRL1 is the more elegant single-scheme
version. Bootstrap the whole two-stage procedure (after spec is frozen) for a
CI on location + slope-change.

→ `component-catalog.md` + `formulation.md`; cross-link Tier-3 and
`implementation.md` (DPP).
