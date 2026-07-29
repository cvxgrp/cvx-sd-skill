# Exploration in marimo

Use marimo in the **exploration** context: data is available, but the model is
not yet decided. A reactive notebook makes judgment-by-looking fast and leaves
behind a model specification. The core library remains marimo-free.

## Why marimo: the slider is a tier classifier

Use reactive widgets to classify each hyperparameter by how the decomposition
responds:

- The slider is **numb across an order of magnitude** (nothing visibly changes)
  → **Tier 1**: set by magnitude, fix, forget.
- The **reconstruction / holdout score moves** as you slide → **Tier 2**: it
  materially contributes and is holdout-tunable.
- The component's **shape changes** (kinks appear, a sawtooth flattens, a
  breakpoint jumps) but the **fit score barely notices** → **Tier 3**:
  structural, *judge by looking*, do **not** holdout-tune.

A Tier-3 knob has no meaningful holdout optimum: it changes the component but
not the scored reconstruction. Classifying knobs this way reduces a coupled
tuning problem to a few numeric searches plus explicit structural judgments.
See [model-specification.md](model-specification.md) for the full hierarchy.

## Build additively, largest sources of variation first

Use the data and the user’s domain knowledge to identify the largest sources of
variation, then model them **additively, biggest first**. Classify each new
component against the components already placed rather than tuning everything
at once.

**Close the search with a critique, not a verdict.** Before handing off, name
at least three uncertainties or possible improvements for the user to judge:
for example, an ambiguous functional form, an uncertain weight, or an untested
period. Do not silently resolve choices the data underdetermines.

## How the agent inspects data without seeing it

Use two channels together:

- **Compute what the eye would catch.** Periodograms, nested variance explained,
  folded profiles, residual inspection, and component-versus-driver checks
  produce numeric evidence. See [diagnostics.md](diagnostics.md).
- **Let the user be the visual sensor.** Interpret what they report from the
  live plots. Turn numeric leads into targeted questions—a seven-day
  periodogram peak becomes “do you expect a weekly cycle?”—rather than asking
  vaguely what they see.

## Widget mapping

| Choice | Widget and behavior |
|---|---|
| Component set or functional form | Dropdown/radio; changing it deliberately rebuilds the model |
| Numeric weight | Log-scale slider |
| DPP-compatible parameter | Slider with a fast re-solve of the same parameterized problem |

Expose functional-form choices such as linear versus spline or smooth versus
piecewise-linear; the data may not settle them without domain judgment. See
[implementation.md](implementation.md) for DPP and repeated solves.

### Let weight sliders reach the penalty nullspace

For interactive exploration, make a regularization slider's upper range wide
enough that the user can see the component's high-weight limit. Difference
penalties have especially legible limits: second-difference smooth and
piecewise-linear trends approach affine functions, while a first-difference
piecewise-constant trend approaches a constant. Watching smooth and PWL trends
approach the same affine limit by different paths is a useful way to understand
what their finite-weight penalties do.

The asymptote is rarely the interesting solution or a model-selection target.
Treat it as a teaching and diagnostic reference: it shows what increasing
stiffness removes, makes the penalty's nullspace visible, and provides a quick
check that the control and component behave as expected. Initialize the slider
at the selected weight, but allow enough additional log-scale range to reveal
the limit.

## Composing with the marimo skills

Compose three skills, each owning one layer:

- **cvx-sd** — the decomposition, DCP, and tuning hierarchy;
- **[`marimo-team/skills`](https://github.com/marimo-team/skills)**, specifically
  **`marimo-notebook`** — authoring a correct reactive notebook;
- **[`marimo-team/marimo-pair`](https://github.com/marimo-team/marimo-pair)** —
  driving the user’s live kernel and committing durable changes.

### Two intersections that bite decomposition notebooks

Defer to the marimo skills for the mechanics; these are the cvx-sd-specific
traps worth naming here:

- **Open a PEP 723 notebook with `--sandbox`**, or marimo ignores its inline
  dependencies, including `signaldecomp`.
- **During a live session the running kernel is the source of truth — drive it,
  do not edit the `.py`.** File edits do not update the kernel and may be
  overwritten. Give each public name one owning cell; use distinct names for
  build, solve, and read stages.

## Exploration ends in a specification

Leave exploration with the components chosen, knobs tier-classified, unresolved
questions recorded, and a durable notebook—not just a chart.
