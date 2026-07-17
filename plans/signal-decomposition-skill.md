# Signal Decomposition Skill — Build Plan

## What this is

An Anthropic Agent Skill that helps a coding agent translate a user's scalar
time-series problem into a convex **signal decomposition (SD)** formulated in
**CVXPY**, solve it, and wire the outputs to downstream analysis. Grounded in
the Meyers & Boyd "Signal Decomposition Using Masked Proximal Operators"
framework, but using CVXPY as the modeling language (dropping the bespoke
masked-prox / BCD / ADMM solver, and with it non-convex support).

The skill is a **translation layer**, not a component library. Its value is in
three layers: (1) an invariant substrate, (2) translation-IN (domain intent ->
components/losses/transforms), (3) translation-OUT (wiring solve outputs to
extraction / CI / stability / reporting / plotting / the user's own packages).

## Hard scope (V1)

- **Scalar (1-D) time series only** (`p=1`), input vector `y` of shape `(T,)`.
- **Convex only.** Non-convex classes are recognized and refused/relaxed.
- Vector-valued (`p>1`) signals, and broader substrate domains (OPF, MPC,
  forecasting, sizing) are **roadmap horizon, explicitly not built in V1.**

## Core invariants (the substrate — never get these wrong)

1. `y` is a 1-D vector on a **regular grid**; scalar **`Δ` (seconds)** ties
   index space to physical time.
2. The **mask** = observed entries. Missing data, held-out validation data, and
   unobserved entries are the **same mechanism** (exclusion from the mask).
3. Decomposition: `y = x1 + x2 + ... + xK` **over observed entries only**
   (masked linking / consistency equality, via NumPy boolean indexing).
4. **`x1` is ALWAYS the residual** (mean-square-small, or robust: l1/huber/
   quantile). Structural components are `x2, x3, ...` -> **append-only** model
   extension. Downstream tools reference components by **semantic role**, not
   positional index.
5. **Convex only**; **CLARABEL** is the default solver (always user-exposable).
6. **Physical periods MUST be scaled by `Δ`** before touching the sample grid.
   Periods live as floats in physical time (seconds); integer conversion via
   `floor`/round happens **as late as possible**. Yearly period = 365.2425
   physical days. Leap days left in, not special-cased.
7. **Periodic components via truncated Fourier** (real-valued float period),
   built with `spcqe` (`make_basis_matrix`, `make_regularization_matrix`);
   the DC column is dropped (offset carried by trend intercept).
8. **`Δ` derivation:** modal (most-common) rounded inter-sample gap, in
   **seconds** (canonical internal unit). Use total-seconds semantics
   (`.seconds` truncates at day boundaries -- footgun).

## Parameterization

- Opt-in `cp.Parameter` is an optimization **only for fixed-data weight /
  regularization scans** (DPP-safe; verify `is_dcp(dpp=True)`).
- **Data-varying loops (bootstrap, expanding-window stability) REBUILD and
  re-solve** -- they change `y`/`T`, not just a weight, so Parameter reuse
  doesn't apply.

## Downstream primitives (domain-agnostic, taught as operations on "a quantity
extracted from a solve")

- **Extractor contract:** `extractor(variables, ...) -> dict[str, scalar|array]`.
  Keys following a convention (e.g. prefix) get auto-bootstrapped.
- **Confidence intervals:** moving-block bootstrap on residuals. Defaults:
  1σ (`confidence_level=68.2`), 500 resamples, `block_size=int(T)` ~ 1 period,
  min-success-fraction guard = 0.5. CIs on **derived scalar metrics**, not raw
  components.
- **Temporal stability:** expanding (forward) window snapped to valid
  endpoints; track component snapshots, derived-quantity history, between-window
  RMSD/delta, and convergence-within-tolerance.
- **Reporting** (markdown) + **plotting** (decomposition panels, stability,
  animation) + **pandas round-trip** (re-wrap outputs on the original index).

## Input / exploration front-end

- **Standardize time axis** (general core lifted from user's
  `standardize_time_axis`, PV solar-noon logic dropped): coerce to datetime
  index; derive `Δ` (modal diff, seconds); detect+surface multiple scan rates;
  reindex onto a regular grid **anchored at midnight of day 1**, gaps -> NaN
  (-> mask); tz hygiene. Emits `(y, Δ, index)`.
- **Heat map** for **sub-daily** data (general core from `make_2d`/`plot_2d`,
  PV clear-day / power labels dropped): fold vector into day (cols) x
  time-of-day (rows) via `reshape(n_steps, -1, order='F')`, `n_steps =
  round(86400/Δ)`; NaNs render as visible gaps. Usable on raw `y` AND on any
  solved component/residual.

## Pre-processing

- **Standardization** (center/scale) and **log transform** (-> multiplicative
  decomposition; `log 0 -> NaN`). First-class, recurs in real examples.

## Skill format & conventions (marimo-style reference)

- Anthropic Agent Skill: `SKILL.md` + `reference/` + `scripts/` + `examples/`.
- Frontmatter (`name`, dense `description` w/ "use when", tight
  `allowed-tools`); progressive disclosure to `reference/`; terse technical
  register; **earned emphasis** (WARNING/MUST/DO NOT) reserved for real
  footguns only. Light user-facing "how to think" framing block; deep thesis in
  `reference/philosophy.md`.
- `allowed-tools: Read, Write, Bash(uv run python **/scripts/*.py *)`.
- `scripts/` = **plain importable Python** the agent writes against, plus **one
  light CLI** (`decompose_cli.py`) as a convenience entry point.

## File tree

```
signal-decomposition/           (skill root -- exact dir name TBD)
├── SKILL.md
├── scripts/
│   ├── time_axis.py        # standardize_time_axis general core -> (y, Δ, index)
│   ├── heatmap.py          # make_2d + plot_2d general core (sub-daily diagnostic)
│   ├── decompose.py        # masked SD builder (x1-residual) + solve  [FIRST]
│   ├── components.py       # convex component catalog as composable builders
│   ├── periodic.py         # Fourier basis/reg via spcqe; float-period -> sample scaling
│   ├── downstream.py       # extractor contract, bootstrap CI, expanding-window stability
│   ├── reporting.py        # markdown reports + plots
│   └── decompose_cli.py    # light CLI
├── reference/
│   ├── formulation.md      # substrate invariants: x1-residual, masked eq, DCP/DPP, CLARABEL
│   ├── time-axis.md        # standardization, Δ, heat map, sub-daily regime
│   ├── component-catalog.md# convex classes; excluded nonconvex + relaxations
│   ├── periodic-and-time.md# Fourier via spcqe, float periods, leap-year, multi-scale
│   ├── downstream.md       # extraction contract, CI, stability, reporting, pandas round-trip
│   ├── philosophy.md       # thesis, horizon, recontextualization
│   └── gotchas.md          # footguns
└── examples/
    ├── co2_quasiperiodic_trend.py     # quasi-periodic + trend + residual
    ├── traffic_multiscale.py          # multi-scale periodic + sparse outlier + trend (sub-daily)
    ├── pv_degradation.py              # shared worked example (daily; corrected to x1-residual)
    └── recontextualization_spline.py  # "your smoothing-spline seasonal+trend fit is a short SD problem"
```

## Build order

1. **`scripts/decompose.py`** -- canonical masked-SD builder (x1-residual
   convention), solve wrapper, DCP verification. Must run + have a smoke test
   before any prose references it.  <- START HERE
2. `scripts/components.py` -- convex component builders it composes.
3. `scripts/periodic.py` + `scripts/time_axis.py` -- physical-time layer.
4. `scripts/heatmap.py`, `scripts/downstream.py`, `scripts/reporting.py`.
5. `SKILL.md` (full prose) once the code contracts are concrete.
6. `reference/*.md`, starting with `formulation.md` then `time-axis.md`.
7. `examples/*.py` -- diverse, PV corrected to x1-residual convention.

## Open / deferred decisions

- Exact skill root directory name.
- Whether to include the in-scope recontextualization example in V1 (lean: yes).
- Dependencies to `uv add`: `cvxpy`, `spcqe`, `numpy`, `pandas`, `matplotlib`,
  `scipy`, `seaborn` (confirm before adding).

## Testing posture

"Good science, not fast shipping." Each script gets a runnable smoke test
(synthetic signal -> build -> solve -> assert structure recovered). Never bare
`python`/`pytest` -- always `uv run`.
