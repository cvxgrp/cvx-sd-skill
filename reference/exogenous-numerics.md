# Exogenous component numerics

Read this reference when an exogenous component uses splines, multiple offsets,
or another basis whose conditioning can affect solver behavior.

## Use the right sequence

```text
training-only support and cleaning decisions
    -> training-only knots or domain
    -> exact rank audit
    -> exact full-rank whitening when needed
    -> solve
    -> recover original coefficients and validate components
```

Do not use whitening to repair unsupported basis geometry. It changes numerical
coordinates only; it does not create information, select knots, remove
observations, or justify truncating singular directions.

## Separate support, rank, and conditioning

Treat spline boundaries and knots as fitted preprocessing. Derive them from the
exact training rows after approved cleaning. Never let holdout rows influence
them.

An extremum is not automatically a stable boundary estimator. Depending on the
domain, use explicit physical boundaries, training quantiles, a robust range,
or another prespecified support rule. Inspect:

- resolved boundaries and knots;
- observations per knot interval and outside the boundaries;
- singular values and the numerical-rank threshold;
- the smallest-singular-value margin over that threshold; and
- raw and column-normalized condition numbers.

Keep extrapolation, clipping, and deletion distinct. Natural splines extrapolate
linearly beyond their boundary knots. Clipping changes the response outside the
boundary; deletion changes the fitted population.

Full rank and good conditioning are different properties. A full-rank basis can
still be solver-hostile. Conversely, a rank-deficient basis cannot be repaired
by rescaling. Column normalization diagnoses scale differences but does not
remove correlation between columns and is not whitening.

## Whiten exactly and preserve the penalty

Let `B` be the full-grid basis and `fit_mask` select the exact rows used in the
linking constraint. For

```text
B_fit = U diag(s) V.T,
C = V diag(1 / s),
beta = C gamma,
B_white = B C,
```

the fitted numerical basis has orthonormal columns:

```text
B_white[fit_mask].T @ B_white[fit_mask] = I.
```

Write the component and its original ridge penalty as:

```python
expr = B_white @ gamma
beta = C @ gamma
loss = weight * cp.sum_squares(beta)
```

Do not penalize `gamma` directly. That changes the statistical model in a
basis-dependent way.

For a concatenated offset design, preserve smoothing in original coordinates
too. If `D` differences adjacent offset coefficient blocks, use:

```python
loss = (
    weight * cp.sum_squares(C @ gamma)
    + lag_smooth_weight * cp.sum_squares(D @ C @ gamma)
)
```

## Fail closed on rank

`whiten_basis` uses the default threshold

```text
tau = max(n_fit, p) * eps_machine * s_max.
```

Require `n_fit >= p` and every singular value above the selected threshold.
On failure, revisit support, data validity, knot count, or model scope. Do not
silently truncate directions or fall back to an unwhitened basis. Rank
reduction is a separate, model-changing feature.

Use `basis_rank_diagnostics` for singular values, numerical rank, rank margin,
and condition numbers. Use `spline_support_diagnostics` for support counts and
spline-specific singular geometry.

## Use the exact fitting mask

Distinguish:

- `built["mask"]`: rows where `y` is observed after caller holdouts;
- component `valid_mask`: rows where that component's inputs are available;
- `built["component_mask"]`: the intersection of component-valid rows,
  independent of whether `y` is observed; and
- `built["fit_mask"]`: their exact intersection used in consistency.

Whiten on `fit_mask`, not all finite rows or evaluation rows. A whitened
component records its parameterization mask; `make_problem` rejects a mismatch
with the final linking mask.

Construct the mask before building a whitened component. For one driver:

```python
offsets = (0, 1, 2)
driver_valid = make_offset_basis(z, offsets).valid_mask
fit_mask = np.isfinite(y_train) & driver_valid
component = exog_spline(
    z,
    offsets=offsets,
    whiten=True,
    fit_mask=fit_mask,
)
built = make_problem(y_train, [component])
```

Intersect the validity masks of every component when several have unavailable
inputs. `offset_source_mask(fit_mask, offsets)` maps fitted target rows back to
the driver rows they consume; spline knot policies use that source support.
Reporting and conditional residual bootstrap use `fit_mask`. Holdout selection
scores candidates on the common intersection of their `component_mask` values,
so offset boundaries do not create incomparable validation samples.

For offset components, `signaldecomp` defines:

```text
shifted[t] = original[t - offset].
```

Positive offsets use past inputs; negative offsets use future inputs. Boundary
rows and rows whose source driver is unavailable are zero-filled in the finite
design matrix and excluded through `valid_mask`.

### Keep rollout policy in the caller

`valid_mask` is a numerical support contract, not a calendar policy. Zero-filled
design rows outside that mask are finite placeholders for CVXPY and must not be
reported as predictions. A caller applying fitted coefficients to another
interval must either provide enough source padding for every offset or mark the
unsupported target rows unavailable.

Keep data acquisition and presentation choices in the consuming application.
`signaldecomp` computes offset designs and their support from supplied arrays;
it does not fetch adjacent periods, infer timestamps, or decide whether a
rollout should pad, drop, return `NaN`, or raise. Estimator and rollout layers
own those policies and must propagate the component validity mask.

## Keep independent roles independent

Whiten separate SD components independently. Jointly whitening, for example,
irradiance and temperature would mix their named roles and couple their
separate penalties unless the entire penalty structure were transformed.

When several blocks genuinely form one component with one declared penalty,
use `whiten_basis_by_blocks`. It accepts the blocks themselves rather than
caller-computed column ranges, preserving their declared order and returning
the corresponding slices and names.

For raw blocks `B1, ..., Bk`, it first whitens each block independently, then
whitens the concatenated first-stage designs. If `C_block` is the block-diagonal
matrix of first-stage transforms and `C_joint` is the joint transform, the
complete coordinate map is:

```text
B_raw = [B1 ... Bk]
C = C_block C_joint
beta = C gamma
B_white = B_raw C
```

The result retains the raw and whitened full-grid bases, block slices and
names, each first-stage whitening, the joint whitening, and `C`. Apply the
component's complete original-coordinate penalty through that returned map;
it need not be an isotropic ridge:

```python
whitening = whiten_basis_by_blocks(
    [main_basis, interaction_basis],
    fit_mask,
    block_names=("main", "interaction"),
)
gamma = cp.Variable(whitening.transform.shape[1])
beta = whitening.transform @ gamma
expr = whitening.whitened_basis @ gamma
loss = weight * cp.sum_squares(penalty_operator @ beta)
```

Every individual block and the joint concatenation must be full rank on the
exact mask. A deficiency at either stage raises with stage-specific
diagnostics; no column is removed and no fallback is used.

An explicit `rank_tolerance` applies to each raw-coordinate block audit. Do not
reuse that absolute threshold after whitening: the joint matrix is in new
coordinates with singular values near one. The joint stage therefore derives
its own scale-appropriate tolerance by default. Pass
`joint_rank_tolerance=...` only when the second-stage geometry needs an
explicit threshold of its own.

This primitive is for blocks that genuinely define **one component with one
declared penalty**. Do not use composed-block whitening as a reason to merge
independently interpreted roles. In particular, keep two main effects and
their interaction as three components when they have separate scientific
meanings or weights.

## Interaction coordinates

`make_interaction_basis(left, right)` forms the row-wise tensor product used by
`exog_interaction`. With `left.shape[1] == q` and `right.shape[1] == r`, columns
are ordered with the left index outermost and the right index varying fastest:

```text
left[0]*right[0], ..., left[0]*right[r-1],
left[1]*right[0], ..., left[q-1]*right[r-1].
```

This ordering makes the flat coefficient vector equivalent to
`coef.reshape(q, r, order="C")`. Interaction whitening is the ordinary
one-basis case: whiten the complete tensor-product design on the exact fit mask,
recover the original matrix, and penalize that recovered matrix.
Both factor bases must be offset-free on the jointly valid rows: augmenting
either basis with a constant vector must increase its numerical rank. Otherwise
the tensor product directly inherits an intercept or main-effect direction.
Uncentered scalar drivers are fine unless the driver itself is constant;
“offset-free” means no constant direction in the basis span, not zero sample
mean.

That factor-wise check is not sufficient for interaction-only semantics. For
example, `left=x` and `right=1/x` are each offset-free but their product is an
intercept. `exog_interaction` therefore performs a second audit during
`make_problem`: on the exact final `fit_mask`, the tensor-product span must have
zero intersection with the span of `[1, left, right]`. This mask is resolved
after observed rows and every component-validity mask are intersected, so the
audit also catches lower-order duplication created only by holdouts or missing
inputs.

Non-finite factor rows are availability failures, not values to impute inside
the basis builder; they are zero-filled in the design and excluded through its
validity mask. If finite factors overflow or nonzero factors underflow to an
exact-zero product when multiplied, rescale them; the builder rejects the
product before CVXPY sees it.

## Validate translation out

Check more than solver status. Compare whitened and original formulations on:

- original objective value and fitted reconstruction;
- every latent component;
- recovered original-coordinate coefficients;
- held-out loss and scientific summaries;
- response curves near support boundaries; and
- original-model constraints and residuals.

When a free level can move between components, also compare mean-centered
component differences and verify whether the total level cancels an apparent
raw offset.
