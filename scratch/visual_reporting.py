"""Visual review harness for reporting plots (human eyeball, not a test).

Renders the two reporting plots to PNGs for manual review of qualities the
Agg smoke tests cannot judge (panel legibility, the signed residual fill, the
snapshot spaghetti coloring + colorbar, convergence markers).

Run::

    uv run python scratch/visual_reporting.py

Writes ``scratch/decomposition.png`` and ``scratch/stability.png`` (gitignored)
and prints their paths.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from signaldecomp import (  # noqa: E402
    linear_trend,
    make_problem,
    multiperiodic,
    solve,
    standardize_time_axis,
)
from signaldecomp.reporting import plot_decomposition, plot_stability  # noqa: E402
from signaldecomp.validation import expanding_window_stability  # noqa: E402

HERE = os.path.dirname(__file__)


def _signal(T=730):
    rng = np.random.default_rng(0)
    t = np.arange(T)
    trend = 10.0 + 0.01 * t
    seas = 2.0 * np.sin(2 * np.pi * t / 365.2425)
    y = trend + seas + 0.5 * rng.standard_normal(T)
    y[300:330] = np.nan  # a gap
    idx = pd.date_range("2020-01-01", periods=T, freq="D")
    return pd.Series(y, index=idx)


def main():
    s = _signal()
    std = standardize_time_axis(s, freq="D")
    build_fn = lambda sig: make_problem(
        sig,
        components=[
            linear_trend(role="trend"),
            multiperiodic(365.2425, num_harmonics=3, weight=1e-1, role="seasonal"),
        ],
    )
    out = solve(build_fn(std["y"]))

    # 1. decomposition panels (observed + fit, trend, seasonal, residual)
    fig1 = plot_decomposition(out, y=std["y"], index=std["index"])
    p1 = os.path.join(HERE, "decomposition.png")
    fig1.savefig(p1, dpi=110)
    print(f"wrote {p1}")

    # 2. stability with a scalar extractor (trend slope) -> all four panels
    stab = expanding_window_stability(
        std["y"], build_fn, min_window=180, step=60, tol=5e-4,
        extractor=lambda o: o["values"]["trend_b"],
    )
    fig2 = plot_stability(stab, role="trend")
    p2 = os.path.join(HERE, "stability.png")
    fig2.savefig(p2, dpi=110)
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
