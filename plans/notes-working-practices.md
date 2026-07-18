# Notes: Working practices for this repo (read before resuming)

Practical conventions and tooling for this codebase, plus the resume point.


## marimo tooling (the notebook "ground truth")

- **`marimo check <nb.py>`** validates the cell DATAFLOW GRAPH (variable
  defs/uses across cells, signatures) plus lints — the notebook analog of
  pytest/type-check. Clean run = structure + wiring sound. It does not verify
  runtime numbers; for that run live: `uv run --group examples marimo edit <nb.py>`.
- **`marimo check --fix <nb.py>`** auto-fixes `markdown-indentation` warnings.
- Idioms in use: bare `mo.md(...)` as a cell's last statement renders (even when
  the cell also exports vars); `mo.output.append(mo.md(...))` for mid-cell
  output; expensive/button-gated work wrapped in
  `with mo.status.spinner(subtitle="...") as _spinner:` with
  `_spinner.update(subtitle="...")` between phases. Widgets are DEFINED
  unconditionally (so downstream cells can read `.value`) but may be DISPLAYED
  conditionally.
- Per-trend-type weight semantics (relevant to any trend UI): `linear_trend`
  has no fit weight (only an optional `slope_weight` ridge);
  `smooth_trend`/`pwl_trend`/`monotone_trend` take a `weight`. A "lambda_trend"
  control applies to all trend types except linear.

## Prose phase (the resume point)

All code is complete, tested (91), and committed; the next phase is `SKILL.md` +
`reference/`.

- Faithfulness to the built API is checkable: grep the real `src/signaldecomp`
  and its `__init__` exports rather than describing from memory.
- Conceptual backbone: `plans/notes-vision-and-scope.md` (three work contexts —
  exploration / specification / implementation, plus review-edit; the Tier
  1/2/3 hyperparameter tuning hierarchy; the spec->production handoff; the
  three-band operating boundary) and the invariants in
  `plans/signal-decomposition-skill.md`.
- Register (from the plan): terse technical, marimo-style reference, progressive
  disclosure to `reference/`, earned emphasis (WARNING/MUST/DO NOT) reserved for
  real footguns only.
- Prose build order: `SKILL.md` first, then `reference/formulation.md`,
  `time-axis.md`, etc.
- Questions to settle with the user before drafting: (1) full first draft to
  red-pen vs. section-by-section with the user steering voice; (2) any existing
  skill whose register to match; (3) source material for the philosophy/thesis.
- The PV worked example (`examples/pvdaq4_degradation.py` + `pv_domain.py` +
  `prep_pvdaq4_daily_y.py`) is a ready, exercised illustration of the
  domain-boundary lesson — good material to reference in the prose.

## Queued code TODO

- `holdout_select`: add a strided/periodic block holdout ("N samples every N*m")
  alongside the current single center block (see "Open / deferred decisions" in
  the main plan). Better for seasonal data. Not started.
