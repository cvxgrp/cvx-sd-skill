# Notes: Vision, Work Contexts, and Scope

Brainstorm summary (pre-prose). Captures the framing decided in discussion for
the eventual `SKILL.md` / `reference/` prose. Not yet implemented; this is
design intent to draw on when writing the markdown files. The code layer it
refers to is built and committed (see `signal-decomposition-skill.md`).

---

## The ultimate vision

Go from **a data file on disk** to **a marimo notebook with a widget** that lets
a user explore *families of models* via sliders, dropdowns, and radio buttons.
Recommend **marimo + the marimo Pear skill** for initial data exploration. The
agent also helps with **model implementation in production code** and with
**working on existing signal-decomposition projects / codebases**.

The skill is a **translation layer**, not a component library or a pipeline
framework.

---

## Three work contexts

The agent's *first* branch is identifying which context it is in; the right
first move differs completely. Distinct entry states, agent postures, and
success criteria -- but all sit on the **same invariant substrate** (x1-residual,
masked linking, Δ-scaling, convex-only). They differ in posture and
deliverable, not in the underlying model.

### 1. Exploration -- "I have data and don't yet know the model."
- Entry: a file on disk, or a vague intent ("is there a trend?").
- Posture: diagnostic, suggestive. Recommend marimo + Pear for first look;
  standardize (`time_axis`); diagnose (`heatmap`); propose candidate model
  families; build the slider/dropdown widget to let the user *feel* tradeoffs.
- Success: the user understands their data and has converged on a model family
  they trust. Output is often the interactive notebook.

### 2. Implementation -- "I know the model; put it into production."
- Entry: a decided (specified) model + a target codebase/pipeline.
- Posture: precise, deterministic. Plain `signaldecomp` calls; correct Δ;
  reproducible build/solve; robustness, error handling, tests. (Why the library
  is plain-importable and dependency-light.)
- Success: clean, correct, maintainable code integrated into their system.

### 3. Review / edit -- "There's existing SD (or SD-shaped) code; improve it."
- Entry: an existing codebase -- either real `signaldecomp`/CVXPY code, or code
  *implicitly* doing SD (recontextualization: a smoothing-spline seasonal+trend
  fit, a hand-rolled detrender).
- Posture: interpretive, corrective. Read it, map onto the substrate ("this is
  an x1-residual with a smooth trend and no mask"), spot footguns (period not
  scaled by Δ, `.seconds` wrap, non-convexity), extend via append-only. Meet
  code where it is; no rewrite unless warranted.
- Success: existing work understood, corrected, or extended.
- OPEN QUESTION: is this one context or two? Reviewing *actual SD code* vs.
  *recognizing latent SD in non-SD code* differ in posture (the recognition
  step is its own skill). Lean: one context, two modes.

**Flow, not silos:** exploration hands off to specification->implementation;
review can kick off exploration. Show the transitions.

---

## Specification -- the phase between exploration and implementation

Exploration converges on a model *family*. Implementation productionizes a
*fully-specified* model. Between them: **commit to the exact weights /
hyperparameters.** The highest-judgment step. Its core is a tuning hierarchy --
the agent should **classify each hyperparameter into a tier, then apply the
right method.** The tier is set by *what the component measures and how much it
contributes*, not by the hyperparameter's type.

### Tier 1 -- set by order of magnitude (cheap, insensitive)
Solution not sensitive within an order of magnitude. Set by reasoning about
scale / component-magnitude ratios; move on. **Do NOT burn holdout budget on
these.** (e.g. a smoothing weight on a clearly-dominant, slowly-varying
component: 1e2 vs 3e2 doesn't matter.)

### Tier 2 -- holdout-tunable, ONLY for components that materially contribute
Holdout tuning (`holdout_select` / weight scan) is meaningful only for
hyperparameters governing components that **contribute significantly to the
reconstruction** -- because holdout scores imputation of the *reconstruction*,
so a knob on a minor component barely moves the score and its "optimum" is
noise. Rule: holdout-tune knobs that move the reconstruction; otherwise drop to
Tier 1. (This is *why* `holdout_select` scores against the reconstruction.)
- OPEN: estimating "significant contribution" -- a possible helper that ranks
  components by share of reconstruction variance/energy, or by holdout-score
  degradation when ablated. Might be a new primitive; might be over-engineering
  vs. eyeballing `plot_decomposition`. Undecided.

### Tier 3 -- set by structural analysis of the component itself (NOT holdout)
Some components measure **sensitive, structurally-meaningful quantities** where
*shape fidelity != reconstruction fit*, so holdout tuning optimizes the wrong
objective and actively fails. Tune by **inspecting the component** (plots,
stability snapshots, domain checks).
- Examples: **step changes** in output (pwl/change-point-like -- weight controls
  readiness to admit a step; too loose smears it, too tight invents spurious
  ones; judge by looking at detected steps). **Accumulated buildup / removal**
  (monotone/integral-type -- e.g. soiling accumulation + cleaning removal;
  judge whether the cumulative curve matches physical reality).
- These are exactly the **shape-valued components** whose stability we reworked
  SNAPSHOTS for -- not a coincidence. Structural tuning needs to *see* the
  component: `plot_stability` snapshots + `plot_decomposition` per-role panels
  are the instruments.
- FUTURE: user has examples of automated Tier-3 structural methods to share
  (step-detection sensitivity, accumulation-rate estimation). These will fill
  in the hard "how to automate 'look at the component and set it'" detail.

### Footgun (earned emphasis, for gotchas)
Don't holdout-tune a hyperparameter whose component (a) contributes little to
the reconstruction [score is noise], or (b) measures a structurally-sensitive
quantity [wrong objective].

---

## The specification -> implementation handoff

How tuned hyperparameters cross into production -- **per tier**, and it drives
the production code architecture.

- **Tier 1:** travels as a literal / config constant. No runtime cost.
- **Tier 2, tuning set EXISTS:** tune during spec, **freeze** the value, ship
  as a constant. Production does a *single* solve. (Holdout runs at spec time.)
- **Tier 2, NO tuning set:** can't freeze (only data is the production data).
  The **tuner ships to production and runs at runtime** (holdout loop per run /
  batch) -> many solves.
- **Tier 3:** **always runtime**, no exception -- the structural feature is
  specific to the new data. The automated structural method runs live on each
  dataset.

**Headline:** *Implementation is rarely a single solve.* Only the all-Tier-1
(or frozen-Tier-2) case is one solve. Runtime holdout tuning and/or any Tier-3
component -> production performs **many** solves per run. The deliverable is
often a **tune-then-solve pipeline** (`tune(y) -> build(y, tuned) -> solve`),
not a lone `make_problem`/`solve`.

- **DPP ties in again:** runtime *weight* scans on fixed data are the DPP-safe
  `cp.Parameter`-reuse case (scan without rebuild) -> the many-solves Tier-2
  loop can be accelerated. Tier-3 methods that change *structure* need rebuilds.
  So Tier-2-vs-Tier-3 even maps onto which runtime solves are Parameter-fast.
- The runtime tuning loop is itself a **production artifact** (Tier-2-no-set and
  Tier-3): teach writing *robust* runtime tuning code (a solve fails mid-scan;
  the holdout optimum is flat/noisy), beyond the one-shot interactive
  `holdout_select`.
- **"Do you have a tuning dataset?"** is a top-level implementation question the
  agent must ask -- it bifurcates the whole production architecture.

### Confidence intervals: final model ONLY
CI is a property of the *committed, fully-specified* model. Running it mid-spec
or inside a tuning loop conflates parameter uncertainty with tuning-induced
variation (and wastes the expensive bootstrap on models you'll discard). CI
sits at the very end, after spec is frozen (or after runtime tuning selects the
operating point for *this* dataset). Footgun: never bootstrap inside the tuning
loop or before the model is specified.

---

## Operating band / graceful boundary (the three-band framing)

The skill provides **techniques, not pipelines.** It occupies a band and hands
off gracefully in both directions:

- **Above (user owns):** domain semantics (what a component *means*, curve->scalar
  domain math) AND system/pipeline architecture (scheduling, drift / re-tuning,
  monitoring, serving, orchestration).
- **The skill's band:** the SD substrate + translation-IN + specification/tuning
  + translation-OUT techniques. The reusable building blocks.
- **Below (CVXPY owns):** convex solver mechanics (CLARABEL internals, DCP/DPP
  machinery).

**Graceful boundary, not a wall.** The techniques are designed to *compose* into
larger systems (frozen constants, embedded tuners, Parameter-accelerated scans
are all pipeline-ready). When a user raises a system-level question (re-tuning
cadence, drift), the agent should **recognize it as the natural extension it is,
surface it, and help build on the skill's techniques** -- WITHOUT trying to
design the user's whole platform. "We are not designing all pipelines for all
people." Illustrative case: freezing a Tier-2 constant -> the agent should
surface "do you expect drift? you could schedule a re-spec built on this tuner"
-- offer the foothold, let the user own the architecture.

This band definition answers the "when do I stop?" question an agent otherwise
gets wrong in both directions: refusing useful adjacent help, or over-reaching
into designing someone's system.

---

## marimo widget as a specification instrument

The widget spans exploration *and* specification, not just exploration:
- **radio / dropdown -> discrete model-family choices** = append-only component
  swaps (trend type; add seasonal y/n; robust loss y/n). The role-based,
  append-only model *is* the discrete widget axis.
- **sliders -> continuous knobs** = component `weight`s, `num_harmonics`, huber
  `M`, quantile `q`.
- Sliding a *weight* on fixed data = DPP-safe, instant (no rebuild). Changing
  `T` or the component set = rebuild. This becomes **user-visible**: which
  sliders are instant vs. which trigger a re-solve.
- Sliding a weight while watching a component change *is* interactive Tier-3
  structural tuning; showing the holdout score update *is* Tier-2. So the widget
  is where a human does the tuning-hierarchy judgment by feel.

Our plot functions already return figures (marimo-cell-friendly);
`components_to_frame` gives a labeled DataFrame a marimo table/plot consumes.

**Recommend, don't depend:** keep the core library marimo-free; marimo
integration is a documented pattern (`reference/marimo.md`?) + maybe an
`examples/` widget notebook, not a coupling. Relationship to the marimo **Pear**
skill: likely a clean handoff -- Pear for the *explore* stage, this skill takes
over at *standardize*.

---

## Implications for prose / `reference/` structure (to decide later)

- `SKILL.md` "how to think" could open by having the agent **identify the work
  context** (explore / implement / review) -- better than defaulting to any one
  path (esp. not defaulting to notebook-building).
- The **operating-band principle** belongs in `SKILL.md` framing + `philosophy.md`.
- Specification tuning-hierarchy is meaty enough for its own
  `reference/model-specification.md` (Tiers) ; the handoff + runtime-tuning +
  many-solves + CI-at-the-end could be `reference/implementation.md` (or one
  spec->production ref carrying the arc through the seam).
- A context lens may add a thin *routing* overlay over the existing
  topic-organized refs, not a rewrite.
- Examples: possibly one dataset carried through all three contexts (explore ->
  implement -> review/extend) to show the arc; or distinct per context.
  `pv_degradation` fits implementation; `recontextualization_spline` is review.

---

## Sequencing summary

Exploration -> Specification (Tier 1/2/3 tuning) -> [handoff: tuning-set?] ->
Implementation (often many solves) -> [final] CI. **Review / edit** meets
existing code and can enter at any point.
