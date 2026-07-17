# Note: multi-period Fourier basis -- it's always B @ theta, only the width changes

## STATUS: spcqe VENDORED and REMOVED as a dependency

The two functions we use (`make_basis_matrix`, `make_regularization_matrix`,
plus helpers `initialize_arrays`, `cross_bases`) are now vendored, trimmed, and
attributed in `scripts/basis.py` (source: github.com/cvxgrp/spcqe). The `spcqe`
package and its heavy transitive tail (scikit-learn, sig-decomp, qss, tqdm) have
been removed via `uv remove spcqe`. Direct deps are now cvxpy, numpy, pandas,
matplotlib, scipy. Trim: the `trend` option was removed (trend is a separate
component in our design); `standing_wave` and `custom_basis` hooks retained.
Vendored version verified bit-identical to spcqe on the periodic smoke test.

## The object is always the same: a 2-D design matrix times a coefficient vector

Single-period, multi-period, and multi-period-with-cross-terms all produce the
SAME kind of object: a 2-D basis matrix `B` of shape `(T, n_columns)`, and a
component expressed as `B @ theta` with a 1-D coefficient vector `theta`. The
problem is linear in `theta` and convex in all cases. "Tensor basis" was the
wrong framing -- nothing about the STRUCTURE changes across these cases.

What changes is only **`n_columns`** -- the number of basis functions, i.e. the
dimension of `theta` (the number of free variables in that component). Some
columns happen to ARISE as elementwise products of columns from two per-period
blocks (the cross terms), but that is just their provenance; they are ordinary
columns of one 2-D matrix.

## Column-count facts (verified against source)

Column layout of `make_basis_matrix` (order fixed by construction):

1. Offset: 1 column of ones (the DC/constant term), ALWAYS at index 0. Drop the
   DC degree of freedom with `B[:, 1:]` and `W.tocsr()[:, 1:]`. (See
   `notes-offset-identifiability.md`.)
2. Per-period Fourier block: `2 * num_harmonics` columns per period, interleaved
   `[cos, sin, cos, sin, ...]` (or `num_harmonics` sine columns for a
   standing-wave period).
3. Pairwise cross-term blocks: for each PAIR of periods, the elementwise
   products of the two blocks' columns -> `(2H_i)(2H_j)` columns. Pairwise only
   (no triple-and-higher products). `max_cross_k` caps harmonics per side.

Column counts (verified):

| periods    | H | ncols | after DC-drop |
|------------|---|-------|---------------|
| [P]        | 3 | 7     | 6             |
| [P]        | 6 | 13    | 12            |
| [P1,P2]    | 3 | 49    | 48            |
| [P1,P2]    | 6 | 169   | 168           |
| [P1,P2,P3] | 3 | 127   | 126           |
| [P1,P2,P3] | 6 | 469   | 468           |

For 2 periods: `ncols = 1 + 2H + 2H + (2H)(2H) = 1 + 4H + 4H^2`. For 3 periods
there are C(3,2)=3 pairwise cross blocks (no triple product), giving the
127/469 counts. Do not assume a simple closed form for >2 periods; compute it or
inspect `B.shape`.

## Design choice for the skill: model dimension, not object type

Two ways to model multi-scale seasonality. They are the same KIND of problem
(`B @ theta`, convex); they differ only in how many columns/free variables the
seasonal component has and what those columns can represent:

1. **One `fourier_seasonal([P1, P2, ...])`** -> a single wider `B` that includes
   cross-term columns. Can represent cross-scale modulation (e.g. the daily
   shape changing across the year), at the cost of many more free variables
   (grows multiplicatively in the number of periods). Less interpretable per
   coefficient.
2. **Separate additive components** -- `fourier_seasonal([P1])` +
   `fourier_seasonal([P2])` -- each a narrower `B_i @ theta_i`, no cross-term
   columns, independent and interpretable. Closer to the monograph's traffic
   example (separate weekly and yearly terms).

The skill's translation-IN guidance and `periodic-and-time.md` should present
this as a choice about **model dimension / whether cross-scale interaction is
wanted**, NOT as a choice between different mathematical structures. Default
recommendation: prefer separate additive components unless cross-scale
modulation is specifically wanted, for interpretability and to avoid coefficient
blow-up. `max_cross_k` tames cross-term width if the wider single-component
basis is desired.

## Also confirmed

- Empirical seasonal sample-mean is small-but-nonzero (~6e-4) even with the DC
  column dropped: harmonics over a non-integer number of periods don't average
  to exactly zero. Assert the STRUCTURAL fact (no DC column / coefficient
  count), not the empirical mean.
