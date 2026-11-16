# v0.1.0 release preparation

Status: preparation in progress; no release tag created by this work.
The distribution version in `pyproject.toml` is already `0.1.0`.

## Release contents

The first release covers scalar masked decomposition, convex trend/periodic/
sparse/exogenous components, ordered offsets, grouped components, interactions,
strict full-rank penalty-preserving whitening, reporting, conditional residual
bootstrap, and expanding-window diagnostics. The skill includes preparation,
formulation, exploration, implementation, and extraction guidance. See the
[README](README.md#supported-scope-for-v01) for limitations and future work.

It does not establish compatibility with the TSGAM estimator or validate a
downstream scientific workflow merely by exporting matching primitives.

## Downstream acceptance record

The component-lift plan required downstream scientific controls for Slices
1-2. The retained plan explicitly recorded that gate as open. On 2026-11-13,
review of repository history and local plan/memory records found no completion
record tying a downstream run to the integrated package revision. The original
numerics handoff describes earlier downstream validation of that application's
own implementation; that is useful provenance, not acceptance of this lift.

Keep this gate open until the owner provides either:

- A downstream run record: application and package revisions, control fixture
  identity, preprocessing/support policy, solvers/tolerances, original objective,
  reconstruction, component and scientific-summary comparisons, and original
  constraint checks with predeclared acceptance tolerances; or
- An explicit decision to defer this application integration gate and release
  the tested upstream package with downstream parity outside the release scope.

No real application data needs to be copied into this repository. Record a
durable report location and its outcome. Similar held-out scores or synthetic
whitening invariance alone do not close this acceptance item.

## Candidate checks

Run from a clean checkout of the candidate commit:

```bash
uv sync --locked
uv run python -m pytest -q
uv run python -m marimo check examples/gasoline_price_trends.py
uv run python -m marimo check examples/pvdaq4_degradation.py
uv run python examples/lagged_exogenous_response.py
uv build --wheel --sdist
git diff --check
```

Install the wheel in a fresh environment outside the checkout. Confirm the
import resolves inside that environment, execute the copied exogenous example,
and run the copied gasoline notebook with its CSV. The PV notebook requires
separately prepared data and its domain helper, so a graph check alone does not
constitute its numerical execution. Record exactly which paths ran.

The automated test workflow runs package tests, documentation examples, notebook
graph checks, and the exogenous example. A remote CI result remains required
after the candidate is pushed.

### Local check record, 2026-11-13

Working tree based on `70e7388` plus the uncommitted release-preparation changes:

- Python 3.12.13, locked project environment: 207 tests passed, including skill
  reference-link checks and executable recipes in the three new references.
- Both repository marimo notebook graph checks passed; skill validation, offline
  lock validation, and `git diff --check` passed.
- Wheel and source distribution built. The wheel contains the three license/
  attribution files; the source distribution excludes ignored plans, memories,
  presentations, and installed agent resources.
- A fresh environment outside the checkout installed the wheel and marimo
  0.23.14. The isolated import resolved from that environment's `site-packages`.
  Its resolver selected newer compatible dependencies (including CVXPY 1.9.3,
  NumPy 2.5.4, pandas 3.1.1, and SciPy 1.18.1), independently of `uv.lock`.
- The copied exogenous example passed its assertions: 317 fitted rows, holdout
  RMSE 0.0366, and whitened Gram error 1.05e-13. The copied gasoline notebook
  and companion CSV passed graph checking and executed to an HTML export.
- The PV notebook was graph-checked only; no new PV data preparation or runtime
  analysis was performed. Python 3.13/3.14 and Linux await the remote CI matrix.

Review correction: the portable-copy instructions now state the PV notebook's
required `examples/` layout and parent launch directory. A temporary copy with
a synthetic pandas Series pickle verified the notebook's actual loader cell
from that launch directory and import of its sibling domain helper. This is a
path-resolution check, not execution of the PV scientific analysis.

Temporary check artifacts were written under `/tmp/cvx-sd-release-dist` and
`/tmp/cvx-sd-portable.E08DUy`; these paths are local evidence, not durable
release downloads. Re-run candidate checks after review on the committed
revision. The downstream acceptance item above remains open.

## Tagging after approval

Resolve the downstream gate above, review and commit the release changes, and
confirm a green CI result on that exact commit. Then create an annotated
`v0.1.0` tag with release notes summarizing the scope and limitations, and push
that tag after owner authorization. Do not tag the pre-change HEAD while local
release changes are uncommitted. Publishing to a package index is a separate
step; the repository and local wheel install paths work before publication.
