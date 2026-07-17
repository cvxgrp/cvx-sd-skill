# Note: DC / offset identifiability in periodic components

## The problem

When a periodic/seasonal component and a trend component coexist, a constant
offset can slosh freely between them with zero change to the objective -- the
non-uniqueness the manuscript flags in sec 2.3. Empirically confirmed during
the `decompose.py` smoke test: sum/fit was correct (fit RMSE ~= noise), but
trend and seasonal each carried an equal-and-opposite ~0.28 offset until the
seasonal was anchored.

## Two equivalent fixes

1. **Basis trim (preferred when the component is basis-represented).**
   `spcqe.make_basis_matrix` returns a Fourier basis whose FIRST column is the
   DC/constant term. Drop it from BOTH the basis and the regularization matrix:

   ```python
   B = spcqe.make_basis_matrix(numharmonics, N, [T])[:, 1:]
   W = spcqe.make_regularization_matrix(numharmonics, lam, [T]).tocsr()[:, 1:]
   ```

   The seasonal component `B @ theta` then cannot represent a constant at all --
   the DC direction is removed from its degrees of freedom. The trend intercept
   carries the DC level. Cleaner than a constraint: fewer variables, no extra
   equality/dual, anchoring intrinsic to the component.

2. **Zero-mean constraint (for non-basis / free-variable components).**
   A free `cp.Variable` periodic has no basis to trim, so anchor with
   `cp.sum(one_period) == 0`. This is what the placeholder `_smooth_periodic`
   in `decompose.py` uses.

## General principle for the skill

- Basis-represented component with a DC ambiguity -> **remove the offset column**
  from the basis (and its regularizer). Do NOT also add a zero-mean constraint;
  that would be redundant.
- Free-variable component with a DC ambiguity -> **zero-mean (zero-sum-over-
  period) constraint**.
- Belongs in `reference/gotchas.md` and drives the `periodic.py` design
  (Fourier-via-spcqe path must slice `[:, 1:]`).

## Also note (from the same worked example)

`spcqe` regularization matrix needs `.tocsr()` before column slicing (it comes
back sparse). The regularizer weight (`lam_seasonal`) is baked into
`make_regularization_matrix`, and the loss is `cp.sum_squares(W @ theta)`.
