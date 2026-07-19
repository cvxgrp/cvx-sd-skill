# Working in this repository

Guidance for AI agents (and humans) contributing to `cvx-sd-skill`.

## What this repo is

Two artifacts that work together:
- **`signaldecomp`** (in `src/`) — a small, tested convex signal-decomposition
  library built on CVXPY.
- **The skill** — `SKILL.md` (the agent entry point) plus `reference/` deep-dive
  docs, teaching a model to *formulate* decompositions well.

Note the split names: the **import package is `signaldecomp`**; the **project /
repo is `cvx-sd-skill`**. `import signaldecomp`, not `import cvx_sd_skill`.

## Running tests

Use the module form — the bare `pytest` console script may not be on PATH:

```bash
uv run python -m pytest -q
```

`uv sync --group dev` first if pytest is missing. The suite should stay green
(currently 97 tests); add tests when you add or change a contract.

## The cardinal discipline: verify, don't assert

Every code snippet in `SKILL.md` and `reference/` is **run against the library
before commit** — signatures, defaults, aux-key names, and DCP validity are
checked, not remembered. When editing docs:

- Confirm builder signatures/defaults from the source, not from memory.
- Confirm any CVXPY expression is DCP with `expr.is_dcp()` / `.is_convex()` /
  `.curvature` (curvature is checkable per-expression, not just per-problem).
- If you claim a recipe "works" (e.g. a convex-sequence heuristic), run it and
  confirm the stated behavior before it goes in a doc.

This is the same discipline the skill teaches its users: construct, verify, then
trust.

- **Read the scaffold, don't just call it.** Before adapting a component, `Read`
  its builder in the `signaldecomp` source (`components.py`; find the install
  with `signaldecomp.__file__`) — they are a few lines each, and the source *is*
  the pattern documentation. Before building in a new problem domain, read the
  nearest `examples/` file. The package is written to be read; treating it as an
  opaque API throws away most of what it teaches.

## Package conventions

- A component is a `Component(role, build)` where
  `build(T) -> (expr, loss, constraints)`; catalog builders and hand-written
  components are the same object. See `reference/formulation.md`.
- x0 is always the residual; structural components are appended and addressed by
  **role**, never index.
- The default solver is `"CLARABEL"`; `solve(..., verify_dcp=True)` is the
  default and refuses non-DCP problems.
- Missing data is handled by boolean-indexing the consistency equality
  (`mask = ~np.isnan(y)`); no selector matrix is materialized.

## Practical calibration notes

- **Regularization weights in this library run small — think `~1e-2`, not
  `~1`.** If a composed component comes out degenerate (flattened, collapsed to
  a line), *lower the weight* before concluding the formulation is wrong.
- **Check whether a bare constraint already gives the behavior before adding a
  penalty.** E.g. bare `monotone_trend` (weight=0) already steps between levels;
  it needs no extra sparse term to "allow jumps."

## Repo layout & private notes

- `src/signaldecomp/` — the library (src-layout, editable install).
- `SKILL.md`, `reference/` — the skill prose.
- `examples/` — worked examples (PV degradation; more planned).
- `tests/` — pytest suite.
- `memories/` and `plans/` are **git-ignored working notes** (local only); do not
  rely on them being present, and do not add public-facing content there.

## License

Apache-2.0. New source files should be compatible; see `LICENSE` / `NOTICE`.
