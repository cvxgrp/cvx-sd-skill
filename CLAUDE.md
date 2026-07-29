# Working in this repository

This repository contains the `signaldecomp` Python library and an agent skill
(`SKILL.md` plus `reference/`). The project is `cvx-sd-skill`, but the import is
`signaldecomp`.

## Running tests

Use the module form — the bare `pytest` console script may not be on PATH:

```bash
uv run python -m pytest -q
```

Run `uv sync --group dev` first if pytest is missing. Add tests when changing a
contract.

## The cardinal discipline: verify, don't assert

Run every code snippet in `SKILL.md` and `reference/` against the library before
commit:

- Confirm builder signatures/defaults from the source, not from memory.
- Verify CVXPY expressions with `is_dcp()`, `is_convex()`, and `curvature`.
- Run claimed recipes and confirm the documented behavior.

Read the relevant builder before adapting a component and the nearest example
before translating a new domain. The short source is pattern documentation, not
an opaque implementation.

## Package conventions

- A component is `Component(role, build)` with
  `build(T) -> (expr, loss, constraints)`.
- x0 is the residual; address structural components by role, never index.
- The default solver is `"CLARABEL"`; `solve(..., verify_dcp=True)` is the
  default.
- Missing data uses boolean indexing of the consistency equality.
- `solve` accepts solver keyword arguments and treats `optimal` and
  `optimal_inaccurate` as successful. Test contracts with meaningful tolerances,
  not one solver's exact array.

See `reference/formulation.md` for the full substrate.

## Repeated solves and validation

- Reuse a built problem with `cp.Parameter` only when data, `T`, mask,
  variables, constraints, and component structure stay fixed. Verify
  `problem.is_dcp(dpp=True)` before claiming DPP reuse.
- Holdout, bootstrap, expanding-window analysis, and component-form changes
  rebuild by design. Their public seam is `build_fn(y) -> built`; do not force
  them through fixed-problem parameterization.
- `holdout_select` currently uses one contiguous block. Do not document K-fold
  or periodic/strided holdout as implemented.
- `bootstrap_ci` requires an explicit `block_size`; there is no safe
  domain-independent default. Extract domain quantities through an explicit
  `extractor(out)`.
- Repeated-solve code needs deterministic policies for failed candidates, flat
  score regions, and Tier-3 null results. See `reference/implementation.md`.

## Practical calibration notes

- Builders make exposed weights approximately commensurable; start on comparable
  log ranges and expand from observed behavior.
- A flattened component usually means the weight is too high; noise chasing
  means it is too low.
- Check whether a bare constraint already gives the behavior before adding a
  penalty.
- Tier comes from fitted effect, not penalty class.

See `reference/model-specification.md` for tuning and
`reference/component-catalog.md` for normalization.

## Documentation state

Keep workflow and routing in concise `SKILL.md`; put theory, lookup material,
and worked decisions in one canonical reference. Use the `SKILL.md` index as the
source of truth for written and planned references.

For marimo examples, validate the notebook graph with:

```bash
uv run --group examples marimo check examples/<notebook>.py
```

`marimo check` verifies cell ownership and dataflow, not runtime results; run
important numeric paths in a live kernel as well.

## Private notes

`memories/` and `plans/` are git-ignored local notes. Do not rely on them being
present or place public-facing content there.

## License

Apache-2.0. New source files should be compatible; see `LICENSE` / `NOTICE`.
