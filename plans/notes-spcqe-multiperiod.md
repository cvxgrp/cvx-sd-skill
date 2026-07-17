# Note: the Fourier basis builds a TENSOR basis across multiple periods

## STATUS: spcqe VENDORED and REMOVED as a dependency

The two functions we use (`make_basis_matrix`, `make_regularization_matrix`,
plus helpers `initialize_arrays`, `cross_bases`) are now vendored, trimmed, and
attributed in `scripts/basis.py` (source: github.com/cvxgrp/spcqe). The `spcqe`
package and its heavy transitive tail (scikit-learn, sig-decomp, qss, tqdm) have
been removed via `uv remove spcqe`. Direct deps are now cvxpy, numpy, pandas,
matplotlib, scipy. Trim: the `trend` option was removed (trend is a separate
component in our design); `standing_wave` and `custom_basis` hooks retained.
Vendored version verified bit-identical to spcqe on the periodic smoke test.

## Empirically verified API facts (now confirmed against source)

`make_basis_matrix(num_harmonics, length, periods, standing_wave=False,
trend=False, max_cross_k=None, custom_basis=None)`

- Basis is a **dense ndarray**; regularization matrix is a **sparse dia_matrix**
  (needs `.tocsr()` before column slicing).
- **Column 0 is the DC/constant** term (all ones). Confirmed. Drop via `[:, 1:]`
  on both basis and regularizer. (See `notes-offset-identifiability.md`.)
- Single period, H harmonics -> `2H + 1` columns (H sin, H cos, 1 DC).

## The tensor-product surprise

Passing MULTIPLE periods does NOT give independent additive seasonals. spcqe
builds the **outer/tensor product** of the per-period bases (cross-terms), so
the daily shape can modulate over the year, etc. -- the multi-periodic
quasiperiodic construction.

Column counts (verified):

| periods         | H | ncols | after DC-drop |
|-----------------|---|-------|---------------|
| [P]             | 3 | 7     | 6             |
| [P]             | 6 | 13    | 12            |
| [P1,P2]         | 3 | 49    | 48            |
| [P1,P2]         | 6 | 169   | 168           |
| [P1,P2,P3]      | 3 | 127   | 126           |
| [P1,P2,P3]      | 6 | 469   | 468           |

For 2 periods: `ncols = (2H+1)**2`. For 3 periods the count (127 at H=3, 469 at
H=6) is smaller than the full `(2H+1)**3 = 343 / 2197`, so spcqe prunes
high-order cross terms (likely governed by `max_cross_k`). Do NOT assume a
simple closed form for >2 periods; compute it or inspect `B.shape`.

## Design implication for the skill (IMPORTANT)

Two genuinely different ways to model multi-scale seasonality, and they mean
different things:

1. **One `fourier_seasonal([P1, P2, ...])`** -> coupled TENSOR basis. Expressive
   (captures cross-scale modulation) but high-dimensional (grows
   multiplicatively) and less interpretable. Use when scales genuinely
   interact (e.g. daily profile changes shape across seasons).
2. **Separate additive components** -- `fourier_seasonal([P1])` +
   `fourier_seasonal([P2])` -- independent, low-dimensional, interpretable.
   Use when scales are (assumed) additive and separable. This is closer to what
   the monograph's traffic example does (separate weekly and yearly terms).

The skill's translation-IN guidance and `periodic-and-time.md` MUST surface this
choice explicitly. Default recommendation: prefer separate additive components
unless cross-scale modulation is specifically wanted, both for interpretability
and to avoid coefficient blow-up. `max_cross_k` is the knob to tame cross terms
if the tensor basis is desired but needs taming.

## Also confirmed

- `spcqe` pulled in `sig-decomp` (OSD package), pandas, matplotlib, sklearn
  transitively -- already installed.
- Empirical seasonal sample-mean is small-but-nonzero (~6e-4) even with DC
  dropped: harmonics over a non-integer number of periods don't average to
  exactly zero. Assert the STRUCTURAL fact (no DC column), not empirical mean.
