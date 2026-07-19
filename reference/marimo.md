# Exploration in marimo

This is the deep-dive for the **exploration** context (see the situation router
in [SKILL.md](../SKILL.md)): data on disk, model not yet decided. The job here
is not to pick the model a priori — it is to make **judgment-by-looking** fast
and legible, and hand off a *specification*. marimo is the recommended surface
for that. Recommend, not depend: the core library stays marimo-free.

## Why marimo: the slider is a tier classifier

Exploration's real work is **classifying each hyperparameter into a tuning tier
by feel** — and a reactive widget makes that a matter of sliding-and-looking
rather than analysis. As you move a weight's slider and watch the decomposition
respond, the tier reveals itself:

- The slider is **numb across an order of magnitude** (nothing visibly changes)
  → **Tier 1**: set by magnitude, fix, forget.
- The **reconstruction / holdout score moves** as you slide → **Tier 2**: it
  materially contributes and is holdout-tunable.
- The component's **shape changes** (kinks appear, a sawtooth flattens, a
  breakpoint jumps) but the **fit score barely notices** → **Tier 3**:
  structural, *judge by looking*, do **not** holdout-tune.

That last case is what the widget is uniquely good at surfacing — a knob that
reshapes a component without moving the fit has a holdout "optimum" that is noise
or the wrong objective, and you can only see it is Tier 3 by watching the shape
while the score sits still. (Full hierarchy:
[model-specification.md](model-specification.md).)

This classification is the **dimensionality reduction** that makes the coupled
specification problem tractable: most knobs turn out Tier 1, a few Tier 2, and
the structural ones you learn to judge by eye. The widget is the **low-judgment
on-ramp to the tuning hierarchy** — parallel to the package being the
low-context on-ramp to correctness.

## Build additively, largest sources of variation first

You don't know a priori which component dominates — determining that *is* part
of exploration. Look at the data, and work with the user, to identify the
largest sources of variation, then model them **additively, biggest first**: a
gross trend or the strongest seasonal cycle before the subtle stuff. Each
component you add is judged against everything already placed, so its knobs are
classified against a settled backdrop rather than all at once — which is what
keeps the coupled tuning problem tractable. The order emerges from the data and
the conversation; it is not decided up front.

## How the agent inspects data without seeing it

You cannot see a rendered plot — but you are not blind to the data. Exploration
runs on two channels working together.

**Compute the structure the eye would catch.** Most visual judgments have a
numerical form the agent *can* read:

- **Dominant periods** — a periodogram (`np.abs(np.fft.rfft(y - y.mean()))`)
  ranks candidate cycles as numbers; the top peaks point straight at which
  `multiperiodic` periods to try. (Low-frequency peaks are trend leakage, so
  treat it as a candidate-finder, not proof.)
- **Which source is largest** — fit a candidate and report the drop in residual
  variance. This makes "largest sources of variation first" *rankable*: on an
  hourly signal, trend-only might explain 32%, adding the daily cycle jumps it to
  96%, the weekly refinement to 98% — so daily dominates, weekly is a
  refinement, and the build order is now evidence, not a guess.
- **Daily/periodic shape** — fold with `fold_to_2d(y, delta)` and take
  `np.nanmean(D, axis=1)`; the time-of-day profile (peak hour, trough hour,
  flatness) is legible as an array without a heatmap.
- **What's still missing** — after a trial fit, inspect the *residual*
  numerically: leftover autocorrelation, residual grouped by time-of-day,
  the largest residual entries (candidate spikes). This is how you iterate
  toward the next component without seeing the fit.

**Let the user be the visual sensor — and ask the right questions.** The user
sees the widget; you interpret what they report ("the trend dips in winter,"
"there's a spike near the end"). The numeric channel *generates the questions*:
a periodogram peak at ~7 days becomes "do you expect a weekly cycle?" — a
targeted, data-grounded question, not a vague "what do you see?" That exchange,
not a silent plot, is the exploration.

## Widget mapping

- **which structural components are in the model** (append-only) → dropdowns /
  radio buttons.
- **weights** → sliders (log-scale, given the order-of-magnitude story).
- **DPP tells you which knobs are instant.** A weight on fixed data re-solves
  fast (same parametrized problem, no rebuild); changing `T` or the component
  *set* triggers a full re-solve. This aligns with marimo's own reactivity — a
  weight slider re-runs only its dependent cells — so put the instant knobs on
  sliders for live feel and treat structural changes as deliberate rebuilds.
  (See [implementation.md](implementation.md) for DPP.)

## Composing with the marimo skills

You do not teach the agent marimo from scratch — compose three skills, each
owning one layer:

- **cvx-sd (this skill)** — what the decomposition *means*: the substrate,
  invariants, DCP, the tier hierarchy. The domain.
- **[`marimo-team/skills`](https://github.com/marimo-team/skills)**, specifically
  its **`marimo-notebook`** sub-skill — how to *author* a correct reactive
  notebook file: cell structure, PEP 723 dependencies, script-mode detection,
  `marimo check`, the reactivity idioms. The artifact.
- **[`marimo-team/marimo-pair`](https://github.com/marimo-team/marimo-pair)** —
  how to *drive a live session* with the user: run Python in the user's actual
  kernel, inspect live state, commit durable changes. The session.

These map onto exploration exactly: it is a **live dialogue with the data**
(`marimo-pair`), building a **durable notebook** (`marimo-notebook`), that
**decomposes a signal** (cvx-sd).

### Two intersections that bite decomposition notebooks

Defer to the marimo skills for the mechanics; these are the cvx-sd-specific
traps worth naming here:

- **A PEP 723 notebook must be opened with `--sandbox`**, or marimo ignores the
  inline dependencies — including `signaldecomp`. Declare deps in the
  `# /// script` header (like the `examples/` prep scripts) and launch sandboxed.
- **During a live session the running kernel is the source of truth — drive it,
  do not edit the `.py`.** File edits will not reach the kernel and may be
  overwritten. This is `marimo-pair`'s core rule; the practical consequence for
  us is that a `built`/`out` object defined in one cell must not be redefined in
  another (marimo enforces one owning cell per public name), so structure the
  build/solve/read cells with distinct names and let reactivity re-run them.

### Trying it

Put this skill repo alongside `marimo-pair` and `skills` in a folder with **just
a data file**, and ask the agent to explore the signal. It should standardize
the time axis, start a (sandboxed, if PEP 723) marimo session, build an
append-only notebook with the widgets above, and drive the live kernel so you
can classify the knobs by feel together.

## Exploration ends in a specification

You leave exploration with a *specified model* — components chosen, knobs
tier-classified — not just a chart. And ending your recommendation with open
questions back to the user ("should the weekly cycle be here at all?") is the
skill *working*, not falling short: the handoff is a spec, not a verdict.
