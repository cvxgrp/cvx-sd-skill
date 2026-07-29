# Diagnostics: reading data and fit without seeing a plot

Use these numerical diagnostics in a script, model review, or live
[marimo](marimo.md) session. They turn visual questions into evidence:
periodograms rank cycles, residual-variance drops rank components, folds expose
cyclic shape, and driver checks reveal misspecified forms. Treat them as
lead-generators, not verdicts.

## Finding structure in the raw signal

### Dominant periods -- periodogram

Rank candidate cycles by spectral energy:

```python
spec  = np.abs(np.fft.rfft(y - y.mean()))
freqs = np.fft.rfftfreq(len(y), d=1.0)          # cycles per sample
order = np.argsort(spec)[::-1]
periods = [1.0 / freqs[i] for i in order if freqs[i] > 0][:6]   # top, in samples
```

The top peaks suggest `multiperiodic` periods in samples; convert physical
durations with `period_samples`.

Two ways it misleads, both important:

- **Low-frequency peaks are usually trend leakage**, not cycles. Treat a period
  near the record length as a candidate, not proof.
- **Modulated cycles show up as *sidebands*, not clean secondary peaks.** If a
  cycle changes shape over a longer period, energy appears beside the main peak.
  Treat nearby sidebands as evidence of evolving cyclic shape and test
  `multiperiodic` cross-terms rather than adding each sideband as a period.

### Cyclic shape -- fold and read the profile

Fold a sub-daily signal (or any solved component) into a (time-of-day x day)
matrix and read the profile as an array:

```python
D = fold_to_2d(y, delta)          # shape (steps_per_day, n_days)
profile = np.nanmean(D, axis=1)   # mean time-of-day shape
peak_hour, trough_hour = int(np.nanargmax(profile)), int(np.nanargmin(profile))
```

`fold_from_standardized(std_out)` folds a `standardize_time_axis` result
directly. Peak, trough, and flatness are legible from `profile`.

**Do not average over a period that modulates the shape.** A mean across a full
modulating cycle can look stationary. Compare short profiles at different
phases:

```python
prof_a, prof_b = D[:, 0], D[:, 3]
evolves = np.corrcoef(prof_a, prof_b)[0, 1]
recurs = np.corrcoef(D[:, 0], D[:, 7])[0, 1]
```

Low `evolves` with high `recurs` indicates quasi-periodic structure:
short-period shape changes with longer-period phase but later recurs.

## Ranking the sources of variation

Rank contributions before choosing a build order. Do not conflate these two
methods.

### Nested variance-explained (the rankable build-order evidence)

Fit incrementally and measure the drop in residual variance as each component is
added:

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

Because each row refits the whole model, the increments account for how
components share variance.

### Reconstruction-energy share (the quick one-shot proxy)

`format_report` computes each component's share of reconstruction energy:

```python
print(format_report(out, y=y))
```

This is **not** nested variance explained. Energy share summarizes the fitted
reconstruction and may overlap across components; nested refits measure each
addition's marginal reduction of residual variance. Use energy share for a
quick composition summary and nested refits to justify an addition.

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

Strong lag-1 autocorrelation suggests under-fit temporal structure; `tod_bias`
suggests a missing or overly stiff daily cycle; a few large entries suggest a
`sparse` component.

### Is each component's *form* right? -- a lead, not a mandate

Inspect a fitted component against its driver:

```python
xk = out["values"]["exog"]
order = np.argsort(z)
# read xk[order] against z[order]: monotonic? U-shaped? kinked?
```

- A clear mismatch—such as a U-shaped response fit as linear—justifies changing
  form (`exog_linear` to `exog_spline`, for example).
- An ambiguous mismatch is a lead for the closing critique, not a mandate to add
  flexibility. If form changes the component without materially changing fit,
  treat it as a Tier-3 decision; see
  [model-specification.md](model-specification.md).

## Where these feed

Use these results to ask targeted questions and document unresolved leads. In a
live session, pair them with the user’s visual reading as described in
[marimo.md](marimo.md); carry unsettled forms, periods, and weights into the
model specification rather than resolving them silently.
