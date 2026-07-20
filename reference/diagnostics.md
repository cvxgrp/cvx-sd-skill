# Diagnostics: reading data and fit without seeing a plot

This is the register-independent toolkit for **inspecting a signal and a fit
numerically** -- the agent's channel for "seeing" structure it cannot render.
Nothing here is specific to a notebook: use it in a plain script, in a review of
someone else's model, or as the numeric half of a live [marimo](marimo.md)
session. If you are deciding what to model, what to add next, or whether a
component's form is right, this is where the evidence comes from.

The whole toolkit rests on one fact: **most visual judgments have a numerical
form.** A periodogram ranks cycles; a residual variance drop ranks sources; a
fold exposes a daily profile; a component plotted against its driver reveals a
misfit. You read the numbers the eye would have caught.

A caution that recurs throughout: these are **lead-generators, not verdicts.**
They tell you where to look and what to try, and several of them mislead in
specific, nameable ways. The traps are called out where they live.

## Finding structure in the raw signal

### Dominant periods -- periodogram

Rank candidate cycles by spectral energy:

```python
spec  = np.abs(np.fft.rfft(y - y.mean()))
freqs = np.fft.rfftfreq(len(y), d=1.0)          # cycles per sample
order = np.argsort(spec)[::-1]
periods = [1.0 / freqs[i] for i in order if freqs[i] > 0][:6]   # top, in samples
```

The top peaks point straight at which `multiperiodic` periods to try (in
**samples** -- convert physical periods with `period_samples`). On an hourly
signal a peak at 24 says "daily," 168 says "weekly."

Two ways it misleads, both important:

- **Low-frequency peaks are usually trend leakage**, not cycles -- a period near
  the record length (e.g. 2016 on an 84-day hourly series) is the trend showing
  up in the spectrum. Treat the periodogram as a candidate-*finder*, not proof.
- **Modulated cycles show up as *sidebands*, not clean secondary peaks.** If a
  daily cycle's amplitude or shape changes over a week, the spectrum near 24
  splits into neighbors (e.g. peaks at 28 and 21 flanking 24) instead of
  producing a clean line at 168. Naive peak-picking would have you "try period
  28" -- an artifact of the modulation, not a real cycle. Sidebands around a
  strong peak are themselves the tell: **the cycle's shape is evolving**, which
  is the signal to reach for `multiperiodic` *with cross-terms* (see the fold
  diagnostic next), not to add a spurious period.

### Cyclic shape -- fold and read the profile

Fold a sub-daily signal (or any solved component) into a (time-of-day x day)
matrix and read the profile as an array:

```python
D = fold_to_2d(y, delta)          # shape (steps_per_day, n_days)
profile = np.nanmean(D, axis=1)   # mean time-of-day shape
peak_hour, trough_hour = int(np.nanargmax(profile)), int(np.nanargmin(profile))
```

`fold_from_standardized(std_out)` is the same fold straight off a
`standardize_time_axis` result. The peak hour, trough hour, and flatness of
`profile` are legible without a heatmap.

**The trap: do not average over a period that itself modulates the shape.** If
the daily profile *evolves* across the week (a cross-term signal), then
`nanmean(D, axis=1)` averages that evolution away -- a full week of a
weekly-modulated daily cycle cancels back to a stationary-looking profile, and
you will wrongly conclude the shape is fixed. To *detect* shape evolution,
compare **single days (or short windows) at different phases** of the longer
period:

```python
# compare individual days a fraction of the modulating period apart
prof_a, prof_b = D[:, 0], D[:, 3]              # e.g. days 0 and 3 within a week
evolves = np.corrcoef(prof_a, prof_b)[0, 1]    # well below 1.0 -> shape changes
recurs  = np.corrcoef(D[:, 0], D[:, 7])[0, 1]  # near 1.0 one week later -> weekly recurrence
```

A daily profile that changes shape within the week but recurs one week later
(low `evolves`, high `recurs`) is the fingerprint of quasi-periodic structure --
exactly what `multiperiodic`'s cross-terms model (the daily shape reshaping as
the weekly/seasonal phase advances). A stationary profile (`evolves` ~ 1.0) says
plain additive seasonality is enough.

## Ranking the sources of variation

"Model the biggest sources first" (see [marimo.md](marimo.md)) becomes evidence,
not a guess, once you can *rank* contributions. Two methods -- do not conflate
them.

### Nested variance-explained (the rankable build-order evidence)

Fit incrementally and measure the drop in residual variance as each component is
added. This is the honest "how much does this component buy" number:

```python
def resid_var(components):
    o = solve(make_problem(y, components))
    return np.var(o["values"]["residual"])

v_total = np.var(y - y.mean())
v1 = resid_var([linear_trend(role="trend")])
v2 = resid_var([multiperiodic([24.0, 168.0], num_harmonics=4, weight=1e-2, role="p"),
                linear_trend(role="trend")])
print(f"trend only:        {100*(1 - v1/v_total):.1f}%")
print(f"+ daily & weekly:  {100*(1 - v2/v_total):.1f}%")
```

A representative run on a cross-term daily+weekly signal: **trend only 12.3%,
adding the periodic component 96.2%** -- so the periodic structure dominates and
the trend is a minor contributor, and the build order is now backed by numbers.
Because each row refits the whole model, the increments account for how
components share variance rather than double-counting it.

### Reconstruction-energy share (the quick one-shot proxy)

`format_report` already computes, for a *single* fitted model, each component's
share of the reconstruction energy:

```python
print(format_report(out, y=y))
# ## Components (share of reconstruction energy)
# - periodic:  61.6%
# - trend:     38.4%
```

This is a cheap read off one solve, useful for "what is this fit made of." It is
**not** the same as nested variance-explained: energy share partitions the
*fitted reconstruction* among components (they can overlap in what they explain),
whereas the nested method measures each addition's *marginal* reduction of
residual variance. Use energy-share for a quick composition summary; use the
nested refit when you need defensible "this component is worth adding" evidence.

## Diagnosing what is still missing

After a trial fit, the **residual** is where the next component hides. Inspect it
numerically:

```python
r = out["values"]["residual"]
# leftover autocorrelation -> unmodeled temporal structure
ac1 = np.corrcoef(r[:-1], r[1:])[0, 1]
# residual grouped by time-of-day -> an unmodeled daily effect
Rd = fold_to_2d(r, delta); tod_bias = np.nanmean(Rd, axis=1)
# largest residual entries -> candidate spikes for a sparse component
big = np.argsort(np.abs(r))[::-1][:10]
```

Strong lag-1 autocorrelation says a trend or smooth component is under-fit;
structure in `tod_bias` says a daily cycle is missing or too stiff; a few
outsized entries say add a `sparse` component. Each reading names a *specific
next component*, which is how you iterate without seeing the fit.

### Is each component's *form* right? -- a lead, not a mandate

Plot a fitted component against its driver and look -- e.g. an `exog` response
vs. its covariate `z`, or a component's residual vs. that driver:

```python
xk = out["values"]["exog"]
order = np.argsort(z)
# read xk[order] against z[order]: monotonic? U-shaped? kinked?
```

- A **clear** misfit (the response is U-shaped but you fit `exog_linear`; a sharp
  kink the smooth form cannot make) means the form is wrong -- switch it
  (`exog_linear` -> `exog_spline`, `smooth_trend` -> `pwl_trend`).
- A **small or ambiguous** misfit is a **lead, not a mandate.** Parsimony still
  favors the simpler form, and the data often cannot settle the choice. Do
  **not** reformulate on the spot to chase it; flag it for the closing critique
  (see [marimo.md](marimo.md)) as a candidate for the *user* to judge against
  their domain prior. This restraint is deliberate: a diagnostic that reshapes a
  component without materially moving the fit is pointing at a Tier-3 structural
  choice ([model-specification.md](model-specification.md)), which is judged by
  looking and by domain knowledge, not auto-tuned.

## Where these feed

In a live session these numeric reads *generate the questions* you put to the
user (a periodogram peak at ~7 days becomes "do you expect a weekly cycle?") --
the exploration dialogue in [marimo.md](marimo.md), where the user is the visual
sensor and this toolkit is the other channel. And they supply the evidence for
the **closing critique**: the leads you could not settle (an ambiguous form, an
untested period, a weight you are unsure of) are exactly what you surface for the
user rather than silently resolving. The diagnostics find the leads; the handoff
is a specification, not a verdict.
