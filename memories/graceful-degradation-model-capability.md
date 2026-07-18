---
name: graceful-degradation-model-capability
description: Design principle — the skill degrades gracefully across model capability. Big models use the judgment layer and generate; small/coding models lean on the thin-but-solid package and inherit correctness by construction. Degrades in bespokeness, never correctness.
metadata:
  type: project
---

**Two audiences, one correct substrate — a deliberate design principle, not an
accident.**

- **Large-context / strong-reasoning models** lean on the *judgment layer*:
  compose-don't-shop, recontextualization, judgment-by-looking, the tuning
  hierarchy. They read `reference/`, imitate the house style, and generate
  bespoke CVXPY. Context-hungry, high ceiling.
- **Smaller / coding-specialized models** lean on the *thin-but-solid package*:
  call `make_problem` / `solve` / the catalog, wire up `components_to_frame` and
  the plots. Low-context, deterministic. They treat `signaldecomp` as a
  well-typed library and let the *library* carry the correctness the bigger
  model would carry in reasoning.

**The key property:** correctness lives in the same place for both. The
invariants a big model must *remember* to honor (x1-residual, masked linking,
Δ-scaled float periods, DC-column drop) are enforced **by construction** in the
package. So a weaker model doesn't get *wrong* answers — it gets *less bespoke*
ones. **The skill degrades in bespokeness, never in correctness.** Most skills
degrade the other way (a weak model produces subtly broken output because the
skill assumed reasoning the model lacked).

**Implications:**
1. Keep the package API usable with **near-zero judgment** — good defaults, clear
   roles, sensible errors, obvious common cases. For the small-model audience
   the scaffold *is* the product; this retroactively justifies effort on the
   scaffold despite "the code is just a scaffold."
2. `SKILL.md` may signal the two paths in one sentence ("fits the catalog? call
   `signaldecomp`. Need structure it doesn't have? compose and verify.") — also
   a routing hint that lets a model self-select by capability.
3. Reconciles "the code is just a scaffold (for the reasoning model)" with "it's
   a real product (for the coding model)" — two doors into the same correct
   substrate, not a contradiction.

**Context-footprint angle:** the skill wants large context to be used at its
*ceiling* (SKILL.md + relevant reference file(s) + scaffold source + user
data/code + a growing marimo notebook). Progressive disclosure (SKILL.md →
reference/) is the mitigation — any single task loads a *slice*, not the whole
corpus. This is why keeping SKILL.md tight and the reference files *independent*
matters (audit for cross-dependence when writing them). And the marginal
context the skill adds is smaller than it looks because the model already has
the convex-optimization knowledge in-weights — the skill ships consistency,
footguns, and the verifiable target, not the domain math.

→ `philosophy.md` spine + possible one-sentence signal in `SKILL.md`.
Related: [[exploration-tier-classification-marimo]] (marimo = low-judgment
on-ramp, parallel to package = low-context on-ramp).
