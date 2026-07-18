---
name: fourier-parameterized-covariance-roadmap
description: Roadmap / thesis-showcase example (past V1) — a smooth+periodic time-varying covariance done by Fourier-parameterizing the precision matrix sequence with matrix-valued coefficients; multiperiodic lifted from scalar to pxp.
metadata:
  type: project
---

**Roadmap example (explicitly past V1: p>1 + matrix variables).** A strong
showcase for the thesis's "one composable schema subsumes/recontextualizes"
claim, because the scalar periodic encoding lifts VERBATIM — only the
coefficient type changes (scalar → pxp matrix).

**Problem:** multivariate time series, believed to have a covariance that is
smooth and periodic in time.

**First, disentangle two readings** (tell the user this):
- **Reading A (usually the real one):** the p series *co-move seasonally* — a
  shared-component multivariate decomposition (common seasonal/trend shape +
  per-series residuals). The "periodic covariance" is *induced* by shared latent
  components. This is the paper's common-term / mean-square-close-entries
  territory (sec 4.4), convex, and the natural first thing to try. Model the
  shared signal and the smooth/periodic covariance falls out.
- **Reading B (the hard, interesting one):** literally a time-varying covariance
  Sigma(t), a sequence of pxp PSD matrices that is smooth (Sigma(t+1)~Sigma(t))
  and periodic (Sigma(t+P)~Sigma(t)).

**Reading B — the elegant formulation (the user's key insight):** don't
regularize a sequence of free PSD matrices with smoothness+periodicity
*penalties* (that's the Graph-Laplacian-over-time-chain framing — valid but
heavier). Instead **parameterize the matrix sequence in a truncated Fourier
basis in t with matrix-valued coefficients** — i.e. `multiperiodic` lifted to
pxp coefficients:

Theta(t) = Theta0 + sum_h [ A_h cos(2*pi*h*t/P) + B_h sin(2*pi*h*t/P) ]

with Theta0, A_h, B_h the pxp matrix coefficients (decision variables), h=1..H.

- **Periodic by construction** (built from period-P sinusoids; no link
  constraint needed — same as scalar periodic).
- **Smooth by construction, tunable by truncation** (fewer harmonics H →
  smoother; smoothness controlled by basis width, not a weight).
- **Variable count collapses:** (2H+1) matrix coefficients instead of T
  matrices (T >> P >> H) — the problem gets *smaller*.
- **Parameterize in the PRECISION Theta(t)=Sigma(t)^-1, not Sigma(t) directly:**
  the Gaussian NLL `-log det Theta(t) + tr(Theta(t) S(t))` is jointly convex in
  Theta (the graphical-model move); `tr(Sigma^-1 S)` is NOT convex in Sigma.
  `log_det` composes fine because Theta(t) is *affine* in the coefficients.
- **PSD:** Theta(t) >= 0 at each sample time is a convex LMI on that affine
  expression — enforce on the sample grid (continuous-in-t PSD is the only
  trickier bit; grid PSD is what a decomposition-at-observed-timestamps needs).
- Optional L1 on off-diagonals → sparse/interpretable dependence (time-varying
  graphical lasso with smooth-periodic structure).

**Penalty-vs-basis duality:** Graph-Laplacian smoothness = the *penalty* framing
(free Theta(t), tied by regularization); Fourier parameterization = the *basis*
framing (Theta(t) constrained by construction). Same identifiability move as the
scalar periodic case; the basis framing is usually better when structure is
genuinely periodic (bakes the belief into the variable).

Why it's a thesis showcase: the exact same convex encoding of "periodic"
(truncated Fourier, smoothness via truncation) transfers from a scalar mean
component to a matrix-valued covariance sequence unchanged. Ties to
[[graceful-degradation-model-capability]] and the abstract's line of work.
