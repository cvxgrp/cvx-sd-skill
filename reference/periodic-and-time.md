# Periodic modeling and physical time

Many energy time series combine interlocking cycles: daily activity changes
between weekdays and weekends, generation profiles change with the season,
and weather responses interact with schedules and human behavior. A useful
model often needs the **shape of one cycle to depend on the phase of another**.
This describes recurring structure, not all variation in every dataset; trends,
weather events, outages, and other departures require their own treatment.

The smooth multiperiodic construction in
[Ogut, Meyers, and Boyd, *Interpretable Net Load Forecasting Using Smooth
Multiperiodic Features*, §2](https://stanford.edu/~boyd/papers/pdf/multiperiodic_forecasting.pdf)
provides a compact way to express these interactions while remaining linear in
the fitted coefficients. The same idea also appears in a different application,
[Ogut et al., *Time Dilated Bundt Cake Analysis of PV Output*, §III](https://stanford.edu/~boyd/papers/pdf/bundt.pdf),
where a daytime PV profile varies over the year. These are sources for the
modeling idea; the exact package conventions are specified separately below.

## Additive cycles versus coupled cycles

For period `P_p` and harmonic cutoff `K_p`, form a vector `h_p(t)` containing
the cosine/sine pairs at frequencies `k / P_p`, for `k = 1, ..., K_p`.
An additive model has the form

\[
f(t) = c + \sum_p a_p^\mathsf{T} h_p(t).
\]

Each cycle has fixed fitted coefficients. For daily and annual cycles, the
annual term can shift the level across seasons, but the daily waveform has
the same harmonic amplitudes and phases throughout the year. Removing cross
terms does **not** leave an amplitude-modulated daily model; it leaves an
additive sum of fixed waveforms.

A pairwise coupled model adds products:

\[
f(t) = c + \sum_p a_p^\mathsf{T}h_p(t)
       + \sum_{p<q} h_p(t)^\mathsf{T} A_{pq} h_q(t).
\]

For a daily/annual pair, the effective daily coefficient vector is
`a_daily + A_daily,annual @ h_annual(t)`. Different daily harmonics can change
by different amounts as annual phase advances. For example,

\[
\sin(\phi_d) + 0.3\cos(\phi_y)\sin(2\phi_d)
\]

changes the second daily harmonic relative to the first. This can reshape a
daily profile rather than merely rescale one fixed curve. The coefficients
remain fixed after fitting; their evaluation at different phases creates the
changing profile. No adaptive coefficient update is implied.

With all cosine/sine pairs and all pairwise products, the number of non-offset
columns is

\[
N = 2\sum_p K_p + 4\sum_{p<q} K_p K_q.
\]

The forecasting paper's annual/weekly/daily example with cutoffs `(2, 3, 4)`
has 122 columns (§2.2, pp. 13–14). This includes pairwise products, not all
orders of a full tensor expansion. The cycles may have a common repeat period
when their periods are commensurate; with incommensurate periods the combined
function can be quasiperiodic. Multiple periods do not by themselves imply
one short, strictly repeating period.

## Physical time and the public component

`multiperiodic` evaluates a truncated Fourier basis at integer sample indices.
Its periods are floating-point sample counts. Convert a physical duration by
dividing by the sampling interval, using the same units for both.

```python
import numpy as np
from signaldecomp import (
    make_problem, solve, linear_trend, multiperiodic, period_samples,
    SECONDS_PER_DAY, SECONDS_PER_WEEK, SECONDS_PER_YEAR,
)

delta = SECONDS_PER_DAY
annual_samples = period_samples(SECONDS_PER_YEAR, delta)
assert annual_samples == 365.2425
t = np.arange(800)
y = 2 + 0.001 * t + np.sin(2 * np.pi * t / annual_samples)
built = make_problem(y, components=[
    linear_trend(),
    multiperiodic(annual_samples, num_harmonics=3, weight=0.01,
                  role="annual"),
])
assert built["problem"].is_dcp()
out = solve(built)
```

Do not round 365.2425 to an integer: fractional periods preserve phase on a
fixed-duration grid without special leap-day indexing. This is a mean-year
cycle, not an exact calendar-anniversary or holiday model. Likewise, a fixed
24-hour UTC cycle does not follow a DST-shifting local activity schedule. See
[time-axis.md](time-axis.md) for preparation choices.

## Choosing the package construction

For an additive daily and weekly model, use separate roles. This is a
functional-form choice; it does not assert statistical independence:

```python
hour = 3600.0
components = [
    linear_trend(),
    multiperiodic(period_samples(SECONDS_PER_DAY, hour),
                  num_harmonics=3, role="daily"),
    multiperiodic(period_samples(SECONDS_PER_WEEK, hour),
                  num_harmonics=2, role="weekly"),
]
```

Passing several periods to one `multiperiodic` call includes the individual
blocks and every pairwise product of their columns:

```python
coupled = multiperiodic(
    [period_samples(SECONDS_PER_YEAR, hour),
     period_samples(SECONDS_PER_WEEK, hour),
     period_samples(SECONDS_PER_DAY, hour)],
    num_harmonics=[2, 3, 4], weight=0.1, role="calendar",
)
coupled_problem = make_problem(np.zeros(24 * 28), [linear_trend(), coupled])
assert coupled_problem["problem"].is_dcp()
assert coupled_problem["variables"]["calendar_theta"].shape == (122,)
```

This snippet checks construction and column count, not whether four weeks
identify annual structure. The public signature is
`multiperiodic(periods, num_harmonics=6, weight=0.1, role="periodic")`.
`max_cross_k` is available in the lower-level `signaldecomp.basis` helpers,
not as an argument to this component builder. A custom subset of cross terms
requires a caller-defined basis/component and its matching penalty.

The coupled design can grow large and contain redundant directions. Even
separate roles may share frequencies
(for example the seventh weekly harmonic and first daily harmonic). Choose
harmonics deliberately, and do not interpret allocation between duplicate
directions as uniquely identified.

## Package penalties and identifiability

In the forecasting paper, each product is associated with its shorter-period
harmonic, and harmonic-number-weighted quadratic roughness terms are combined
using one hyperparameter per period (§2.3). That motivates smoothness across
each cycle, but is not an assertion that every implementation uses the same
normalization, intercept, or regularization parameters.

In this package, the penalty is `sum_squares(W @ theta)`. A single-period
cosine or sine column of harmonic `k` and period `P` receives diagonal entry
`weight * 2*pi*k / sqrt(P)` in `W`. Period blocks are sorted from longest to
shortest; a product column inherits the entry for its shorter-period factor.
There is one public scalar `weight` for the joint component, and it is inside
`W`: multiplying it by ten multiplies the quadratic penalty by one hundred.
This is the package's Fourier-inspired coefficient roughness convention; it
should not be equated with the exact derivative energy of every coupled
waveform or with the source paper's per-period parameterization.

The standard Fourier block contains cosine/sine pairs for each harmonic. More
harmonics admit sharper shapes but can fit noise, alias above the Nyquist
frequency, and duplicate other components. Start with a small count and inspect
folded residuals and holdout behavior. A record shorter than a candidate cycle
provides weak evidence for distinguishing that cycle from a trend.

The builder removes the explicit DC column, assigning the intercept to other
roles. This does not force the fitted component to have zero empirical mean
on a partial or masked cycle, nor does it remove all possible redundancy in
coupled bases. Do not subtract a fitted component's mean without documenting
where that offset goes.

Coefficients are exposed as `<role>_theta`; prefer the
full-length role curve for interpretation unless you have explicitly tracked
the basis column order.

Inspect trend and seasonal components together: a flexible trend can absorb
slow seasonality, while a long-period Fourier term can absorb drift. Similar
reconstruction errors do not establish the same decomposition. Use
[model-specification.md](model-specification.md) and
[diagnostics.md](diagnostics.md) to resolve the model's intended meaning.

## Applications and the forecasting boundary

The net-load forecasting paper combines these time features with past
observations and fits horizon-specific forecasts, including point and quantile
variants (§3). `multiperiodic` supplies a time-feature component; it does not
implement that complete forecasting procedure, causal feature filling,
forecast-horizon alignment, or out-of-sample evaluation. Those choices belong
to the caller's forecasting layer. Future driver values are not automatically
available just because a calendar basis can be evaluated at future times.

For load estimation, the same calendar construction can be combined with
current and lagged weather spline responses, for example additively in log-load
with a shared level. This can describe both recurring schedule changes and
weather sensitivity. Reconstruction of withheld reference loads and validation
against externally measured demand answer different questions: reproducing a
reference dataset does not remove bias in that dataset. Comparisons between
complete calendar-and-weather model specifications also do not isolate the
benefit of cross terms without a controlled ablation.

The Bundt paper illustrates the concept's range beyond autoregressive
forecasting: it fits nonnegative, ordered marginal PV quantiles using annual
Fourier terms, sine functions of position within a dilated PV day, and their
products (§III, p. 4). Sunrise/sunset estimation and energy-preserving PV-day
dilation precede that fit (§II). These specialized coordinates, sine-only
daytime basis, and coupled quantile constraints are not the default
`multiperiodic` model on a uniform physical-time grid. Reproducing the method
requires that preparation and formulation explicitly; its role here is an
example of cross-period shape modeling, not a claim of package parity.
