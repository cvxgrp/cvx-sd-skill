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

---

## PROGRESS & DECISIONS LOG (living section — read this first)

**Status:** the core library, validation, time-axis, heat-map, and reporting
layers are built, tested (**76 passing tests**), and committed. Still to build:
prose (`SKILL.md`, `reference/`) and `examples/`. Several original-plan details
below are SUPERSEDED — see notes.

### Built & committed

- **Package layout: `src/signaldecomp/` (installable, src-layout, editable
  install).** SUPERSEDES the flat `scripts/` layout drawn in the File Tree
  below. Absolute intra-package imports throughout. Curated public API in
  `__init__.py`. Rationale: clean imports, testability, and readiness for a
  possible future thin MCP-server wrapper (skill is the primary product; the
  server is deferred and does NOT drive design).
- **`decompose.py`** — keystone: `make_problem(y, components, residual_loss)` +
  `solve(built, solver=CLARABEL, verify_dcp=True)` + `Component` dataclass
  (`role` + `build(T)->(expr,loss,cons)` + `aux`). x1-residual and masked
  linking equality enforced by construction. `residual_loss` is CALLABLE-FIRST
  (any DCP-compliant `loss_fn(x1)->scalar`); string names are convenience
  aliases. SUPERSEDES the hard-coded 4-loss menu idea.
- **`data_fidelity.py`** — φ_1 loss factories (`l2_loss`, `l1_loss`,
  `huber_loss(M)`, `quantile_loss(q)`); presets + patterns for custom losses.
  (Named `data_fidelity`, not `losses`: every component has a loss; this is
  specifically the residual's data-fidelity term.)
- **`basis.py`** — Fourier basis + regularization matrices, VENDORED from spcqe
  (github.com/cvxgrp/spcqe) and trimmed (trend option removed; standing_wave +
  custom_basis kept). **spcqe dependency REMOVED.** SUPERSEDES invariant #7's
  "built with spcqe" and the dependency list's `spcqe`.
- **`periodic.py`** — `multiperiodic(periods, num_harmonics, weight, role=
  "periodic")` (RENAMED from `fourier_seasonal`; makes no domain assumption).
  Float periods, DC column dropped by construction. `period_samples()` +
  SECONDS_PER_{DAY,WEEK,YEAR} helpers. Multi-period builds one wider `B @ theta`
  with cross-terms (NOT a different object — "tensor basis" framing retracted;
  see notes-spcqe-multiperiod.md).
- **`components.py`** — convex catalog: `linear_trend`, `smooth_trend`,
  `pwl_trend`, `monotone_trend`, `sparse`, `bounded`/`nonneg` wrappers, plus
  `exog_linear(z, ...)` and `exog_spline(z, ...)` (exogenous covariate
  components; covariate captured in factory closure; lag-0 only). Re-exports
  `multiperiodic`.
- **`spline.py`** — natural cubic spline basis, VENDORED from the TSGAM
  estimator (Alliance for Sustainable Energy / Nimish Telang, BSD-3).
- **`validation.py`** — domain-agnostic validation, parameterized by a
  `build_fn(y)->built` (rebuild on new data) and an
  `extractor(out)->scalar|dict` (quantity of interest). Generalized from the
  RdTools PV-degradation reference with domain specifics removed:
  - `bootstrap_ci`: moving-block residual bootstrap, mask-preserving;
    `block_size` is REQUIRED (no default — must match the residual's remaining
    dependence; the agent helps the user pick from time scales/components).
    Defaults: 500 resamples, 68.2% (1σ), min-success-fraction 0.5.
  - `expanding_window_stability` + `valid_endpoints`: growing-window refits,
    endpoints snapped to observed samples. **Component-curve SNAPSHOTS are the
    default** (per-window solved curve for each structural role, NaN-padded, +
    normalized between-window RMSD/mean-abs movement) — so stability works for
    shape-valued components (monotone/smooth/pwl) with no natural scalar. An
    optional `extractor` ADDS scalar history/`|delta|`/convergence (curve->scalar
    domain math stays in the user's extractor). `roles=` filter; ~500MB snapshot
    warning. `min_window`/`step` explicit; `tol` required only with `extractor`.
  - `holdout_select`: contiguous-block hold-out model selection via the native
    masking mechanism; scores imputation of held-out truth (rmse/mae) against
    the reconstruction; takes `{name: build_fn}` candidates.
- **`time_axis.py`** — standardize raw input to a regular grid. Emits dict
  `{y, delta, index, scan_rates, series}`. Lifted from the PV `standardize_
  time_axis`; solar-noon / tz-correction heuristics DROPPED (domain-agnostic).
  Keeps: midnight-of-day-1 anchor, nearest-snap reindex (gaps->NaN->mask),
  dedupe, total-seconds Δ (the `.seconds` 24h-wrap footgun avoided).
  - **Δ / scan-rate detection uses RELATIVE-TOLERANCE clustering** (`rel_tol`,
    default 0.02), NOT fixed-second rounding. SUPERSEDES invariant #8's "modal
    *rounded* gap" and the source's `round_to_seconds=10`: fixed rounding
    fragments jittered continuous timestamps; relative clustering does not.
  - **Δ resolution (RESOLVED):** the empirical modal gap is snapped to the
    nearest *standard pandas frequency* within **1%** (`snap_tol=0.01`) via
    `nearest_standard_freq`, so jittered hourly (3599) -> `("h", 3600)`.
    Non-standard rates (outside 1%) keep the empirical Δ with `freq=None`. A
    user may pass `freq=<pandas string>` to assert the rate authoritatively.
    Standard table is fixed-duration up to daily (`"D"`=86400 hard-coded);
    **coarser than daily is REFUSED** (weekly/monthly need calendar-aware prep,
    not a fixed-second grid). `delta_seconds` input dropped (Δ is an output);
    return dict gains `"freq"`.
  - **DST:** module carries a WARNING to supply *local standard time, no DST*
    (a fixed-second grid can't represent 23h/25h DST days). To expand in the MD.
- **API contracts (as built; prose will formalize these):**
  - `solve(built) -> {..., "status", "values"}`; `values` maps role -> solved
    numpy array, PLUS each component's `aux` entries. `values["residual"]` is x1.
  - **`aux` naming convention:** components expose extra named quantities keyed
    `"<role>_<name>"` (e.g. `periodic_theta`, `trend_a`/`trend_b`,
    `<role>_beta`/`_coef`). Downstream keys off these role-based names.
  - **Scalar collapse (`_solved_value`):** scalar aux (e.g. a slope) come back
    as plain Python floats, not 0-d arrays; vectors as numpy arrays.
  - **`bounded`/`nonneg` wrappers share the inner component's `role` and `aux`**
    (bounding a trend still exposes its slope); they add constraints, not loss.
- **`heatmap.py`** — sub-daily fold diagnostic. `fold_to_2d(y, delta)` folds a
  1-D signal (raw `y` OR any solved component array) into a (time-of-day x day)
  matrix via `reshape(n_steps, -1, order='F')`, `n_steps=round(86400/delta)`.
  Guards: whole-days divisibility, sub-daily-only (delta<=86400), day-evenly-
  divides. Trims all-NaN edge days. `fold_from_standardized(std_out)` wraps the
  time_axis dict. `plot_heatmap(D, signed=...)`: `plasma` for non-negative;
  `signed=True` -> `seismic` + `TwoSlopeNorm(vcenter=0)` (zero=white) for
  residuals/signed components; NaN gaps render gray (`with_extremes(bad=)`),
  month/year tick logic kept. Lifted from PV `make_2d`/`plot_2d`; clear-day /
  power labels and **seaborn** DROPPED. Human visual review via
  `scratch/visual_heatmap.py` (signed off).
- **`reporting.py`** — translation-OUT. `components_to_frame(out, index=, y=,
  mask=)`: re-wrap solved components as a DataFrame (one col per full-length
  role + `residual` + `reconstruction`, `y` if passed; scalar/non-T aux
  excluded). Imputed-dense by default; `mask=` restores NaN at unobserved
  entries. `plot_decomposition(out, y=, index=, df=)`: stacked signal+fit /
  per-role / signed-residual panels; local `rc_context` (no global style
  mutation), colorblind cycle. `plot_stability(stability, role=)`: snapshot
  spaghetti (plasma by window length) + normalized RMSD, plus scalar
  history/delta panels when the run used an extractor. Generalized from PV
  `plot_decomposition`/`plot_stability` (x1/x2/x3 -> role-based; %/yr, tableau
  global-style, and `plot_trend` DROPPED). Visual review via
  `scratch/visual_reporting.py` (signed off).
- **`tests/`** — real pytest suite (`test_decompose`, `test_periodic`,
  `test_components`, `test_exog`, `test_validation`, `test_time_axis`,
  `test_heatmap`, `test_reporting`), **76 tests**. Plot code has Agg-backend
  smoke tests + human visual review (see `scratch/`). SUPERSEDES per-module
  `__main__` tests.
- **Dependencies (actual):** cvxpy, numpy, pandas, matplotlib, scipy; dev:
  pytest. NO spcqe, NO seaborn.

### IN FLIGHT (resume here)

- `reporting.py` complete (pandas round-trip + `plot_decomposition` +
  `plot_stability`), tested (76 pass), visually reviewed (signed off), and wired
  into the public API. Also in this batch: `expanding_window_stability` reworked
  to snapshots-default, and a latent `time_axis` bug fixed (`freq="D"` calendar
  offset couldn't go through `pd.Timedelta`; now uses the fixed-seconds table +
  regression test). Being committed now (no signature). Next: prose + examples.

### Next up

- `SKILL.md` + `reference/` prose, now that the code contracts are final.
- `examples/*.py`.
- (Deferred: markdown-report generation — `components_to_frame` + the two plots
  cover the reporting need for now; a text/markdown summary can come with prose.)

### Naming note

- Package is `signaldecomp`; repo/project is `cvx-signaldecomp-skill`.
- The File Tree, invariant #7, Parameterization, and Dependencies sections
  below are the ORIGINAL plan and are partially superseded as noted above; kept
  for the design intent they still carry.

---

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
   built with the **vendored** `basis.py` (`make_basis_matrix`,
   `make_regularization_matrix`; from spcqe, spcqe dependency removed); the DC
   column is dropped (offset carried by trend intercept).
8. **`Δ` derivation:** modal (most-common) inter-sample gap in **seconds**
   (canonical internal unit), via **relative-tolerance clustering** of gaps
   (not fixed rounding), then **snapped to the nearest standard pandas
   frequency within 1%** (else kept empirical). Measured with
   `Timedelta.total_seconds()` (`.seconds` wraps at day boundaries -- footgun).
   Fixed-duration table capped at daily (`"D"`=86400); coarser is refused.

## Parameterization

- Opt-in `cp.Parameter` is an optimization **only for fixed-data weight /
  regularization scans** (DPP-safe; verify `is_dcp(dpp=True)`).
- **Data-varying loops (bootstrap, expanding-window stability) REBUILD and
  re-solve** -- they change `y`/`T`, not just a weight, so Parameter reuse
  doesn't apply.

## Downstream primitives (domain-agnostic; AS BUILT in `validation.py`)

All parameterized by two user callables (no domain assumptions):
- **`build_fn(y) -> built`** — rebuilds the problem on new/modified data
  (wraps the user's `make_problem(...)`). Data-varying loops rebuild, not
  Parameter-reuse.
- **`extractor(out) -> scalar | dict[str, scalar]`** — pulls the quantity(ies)
  of interest from a `solve()` output dict. NO auto-by-prefix convention; the
  user names quantities explicitly. Scalar or dict handled uniformly.

- **`bootstrap_ci`:** moving-block residual bootstrap, mask-preserving.
  `block_size` is **REQUIRED (no default)** — must match the residual's
  remaining dependence. Defaults: 500 resamples, `confidence_level=68.2` (1σ),
  `min_success_fraction=0.5`. Percentile CI per extracted key.
- **`expanding_window_stability` + `valid_endpoints`:** expanding window snapped
  to observed endpoints; per-window history, between-window |delta|, and
  "stay-within-tol-of-final" convergence. `min_window`/`step`/`tol` explicit
  (no magic time defaults).
- **`holdout_select`:** `{name: build_fn}` candidates; single contiguous central
  held-out block (V1; K-fold is roadmap), `holdout_fraction=0.2`; scores
  imputation of held-out truth against the **reconstruction** (rmse/mae);
  failed candidates score NaN and are not selected.
- **Reporting** — BUILT in `reporting.py`: pandas round-trip
  (`components_to_frame`) + `plot_decomposition` + `plot_stability`. (Markdown
  text-report generation deferred to the prose phase.)

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
- **TODO `holdout_select` — add strided/periodic block holdout.** As built it
  uses a SINGLE CENTER block (`holdout_fraction`, centered; or explicit
  `holdout_slice`). Add a periodic scheme: hold out N contiguous samples every
  N*m samples (e.g. 1 week every 5th week = 20%), so the held-out set samples
  every phase of a seasonal cycle rather than one contiguous regime. Better for
  seasonal data. Likely a `holdout="center"|"periodic"` option (or a
  block-length + stride pair). Keep center as an option; blocked (not random)
  still respects local correlation. Not yet built.
- Dependencies (RESOLVED, as built): `cvxpy`, `numpy`, `pandas`, `matplotlib`,
  `scipy`; dev `pytest`. spcqe VENDORED (not a dep); seaborn NOT used.

## Testing posture

"Good science, not fast shipping." Each script gets a runnable smoke test
(synthetic signal -> build -> solve -> assert structure recovered). Never bare
`python`/`pytest` -- always `uv run`.
