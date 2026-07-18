---
name: hourly-load-worked-example
description: Planned second worked example (beside PV) — hourly electrical load with daily/weekly/yearly periodicity and a U-shaped temperature response. Natural marimo slider demo; carries several teaching points.
metadata:
  type: project
---

Planned second `examples/` case alongside the PV degradation example. Emerged
from a live "what do you recommend?" walkthrough this session.

**Problem:** hourly electrical load with daily + weekly + seasonal patterns, and
more load when temperature is very hot OR very cold (U-shaped temperature
response).

**Recommended decomposition:**
- `Δ` = 3600 s (hourly, via `time_axis` helper).
- Joint `multiperiodic([daily, weekly, yearly])` via `period_samples(...)` — one
  wider basis with cross-terms (daily shape interacts with weekly/seasonal).
  Periods: `SECONDS_PER_DAY`, `SECONDS_PER_WEEK`, `SECONDS_PER_YEAR`.
  **Harmonics-per-scale nuance:** daily wants MANY harmonics (sharp
  morning/evening ramps), yearly wants FEW (smooth). A single `num_harmonics`
  may not serve both → legitimate reason to split into two `multiperiodic`
  components with different harmonic counts (clean append-only extension).
- `exog_spline(temperature, ...)` for the U-shape (response in temperature, not
  time; covariate must be time-aligned to the load vector).
  - **Compose-past-the-catalog option:** if a free spline wiggles instead of a
    clean U, ENFORCE convexity-in-temperature (spline basis + a 2nd-difference
    ≥ 0 constraint on the response ordered by temperature) — a bespoke DCP
    component to construct + `verify_dcp`. Start with free `exog_spline`, look
    at the recovered curve, reach for enforced-convex only if it misbehaves.
- Residual: `l2` to start; switch to robust (`huber`/`quantile`) or add a
  `sparse` component for anomalous demand days (heat waves, grid events).
- **Weekly caveat:** "weekday vs weekend" is near-square, not smoothly periodic;
  a Fourier weekly term works but may want enough harmonics for the flat-then-
  drop shape; a categorical weekday/weekend effect is an alternative. Start
  Fourier, inspect the recovered weekly shape.

**Record-length interaction** (see
[[trend-seasonal-confound-record-length]]): the yearly term vs. trend decision
depends on how many years of history — for 1–2 yr keep yearly periodic, drop
trend.

**Why it's a good example:**
- Natural marimo slider-demo (three periodic scales + temperature spline is a
  great feel-the-tradeoffs notebook) → carries
  [[exploration-tier-classification-marimo]].
- Carries the trend↔seasonal confound + record-length rule concretely.
- The enforced-convex temperature response is a clean compose-don't-shop case.
- A related mini-example: the breakpoint two-stage / IRL1 trick
  ([[irl1-iteratively-reweighted-l1]]).

**Sourcing note:** prefer building examples from the author's *real*
initial-analysis boilerplate where possible (see
[[dogfooding-adaptable-boilerplate]]), sanitized + domain-annotated as PV was.
