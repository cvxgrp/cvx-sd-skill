# Memory index

One line per memory. Content lives in the linked files, not here.

## Prose-phase framing (philosophy.md / SKILL.md)
- [Convex is not brittle — under-specification](convex-not-brittle-underspecification.md) — philosophy.md opening: convex opt is MISperceived as rigid/brittle; truth is under-specified-by-design (p>n), structure steers not rigidifies; Markowitz++ corroborates.
- [Beer/Eno epigraph](beer-eno-epigraph.md) — "specify only somewhat, ride the dynamics"; Eno paraphrasing Beer (NOT verbatim); attribution homework before publishing.
- [Binding constraint = hyperparameter specification](binding-constraint-hyperparameter-specification.md) — deepest framing: not stats/runtime (solved) but hyperparameter INTERACTION growing super-linearly with components; justifies the whole tuning apparatus.
- [Graceful degradation across model capability](graceful-degradation-model-capability.md) — big models use judgment layer + generate; small/coding models lean on thin-but-solid package, inherit correctness by construction; degrades in bespokeness not correctness.
- [Exploration = tier-classification in marimo](exploration-tier-classification-marimo.md) — exploration is where the user classifies knobs into Tiers 1/2/3 by feel; append-only = greedy-tier-order = widget order; the "why marimo."
- [Dogfooding: adaptable boilerplate](dogfooding-adaptable-boilerplate.md) — author already uses the pattern in active research; earned-confidence tone for philosophy.md; source examples/ from real boilerplate.

## Techniques / tricks (formulation.md / component-catalog.md)
- [IRL1 (iteratively reweighted L1)](irl1-iteratively-reweighted-l1.md) — convex route to L0/cardinality; the canonical "second problem depends on first's output"; DPP-safe; generalizes to any sparse component.
- [Soiling-sawtooth composability example](soiling-sawtooth-composability-example.md) — verified bespoke-component example (SKILL.md layer 5); term-by-term semantics, coherent weights, two silent failure modes.
- [Tier-3 structural tuning examples](tier3-structural-tuning-examples.md) — user will supply concrete automated "component structure analysis" tuning methods later; placeholder in prose until then.

## Gotchas / reference
- [Trend↔seasonal confound + record-length rule](trend-seasonal-confound-record-length.md) — trend & low-freq periodic fight over the same signal; corrected rule (1–2yr: keep yearly, drop trend).

## Worked examples (examples/)
- [Hourly load worked example](hourly-load-worked-example.md) — planned 2nd example: daily/weekly/yearly multiperiodic + U-shaped temperature exog_spline; natural marimo slider demo.

## Roadmap / parked (post-V1)
- [Fourier-parameterized covariance](fourier-parameterized-covariance-roadmap.md) — smooth+periodic time-varying covariance via Fourier-parameterized precision matrix (multiperiodic lifted to matrix coefficients); thesis "schema generalizes" showcase; p>1, past V1.
- [Meta-pattern: convex-substrate skill family](meta-pattern-convex-substrate-skill-family.md) — PARKED; cvx-sd is the first instance of a repeatable agent+convex-substrate skill pattern (Markowitz++, OPF, MPC...); ship one exemplar first.

## Process / outreach
- [Collaboration funnel (author contact)](collaboration-funnel-author-contact.md) — scoped author-contact as a collaboration funnel not a support desk; public channel for support, bounded warm invitation for frontier problems, agent prepares a brief.
