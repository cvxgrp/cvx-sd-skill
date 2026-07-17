"""Visual review harness for the heat-map diagnostic (human eyeball, not a test).

The automated tests (``tests/test_heatmap.py``) only prove a figure is produced
without error -- they cannot judge whether the heat map *looks* right (colormap
legibility, gaps reading as gray, zero being neutral/white in the diverging
case, tick labels). This script renders a single multi-panel gallery to a PNG so
a human can review those qualities.

Run::

    uv run python scratch/visual_heatmap.py

Writes ``scratch/heatmap_gallery.png`` (gitignored) and prints its path. Re-run
after any change to ``heatmap.py`` plotting to re-review.

Panels:
  1. Non-negative sub-daily signal (plasma) -- the normal case.
  2. Signed residual centered at zero (seismic, vcenter=0) -- zero should be
     white, +/- symmetric red/blue.
  3. Signal with injected NaN gaps (plasma) -- gaps should read as gray.
  4. End-to-end: standardize -> decompose -> fold a solved component.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from signaldecomp import (  # noqa: E402
    make_problem,
    multiperiodic,
    smooth_trend,
    solve,
    standardize_time_axis,
)
from signaldecomp.heatmap import fold_from_standardized, fold_to_2d, plot_heatmap  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "heatmap_gallery.png")


def _synth_subdaily(days=60, seed=0):
    """15-min data: a daily bell-shaped cycle + slow seasonal gain + noise."""
    rng = np.random.default_rng(seed)
    steps = 96
    t_day = np.arange(steps)
    bell = np.exp(-0.5 * ((t_day - 48) / 12) ** 2)  # peak midday
    season = 1.0 + 0.3 * np.sin(2 * np.pi * np.arange(days) / days)
    D = np.outer(bell, season)  # (steps, days)
    y = D.reshape(-1, order="F") + 0.02 * rng.standard_normal(steps * days)
    y = np.clip(y, 0, None)
    idx = pd.date_range("2021-01-01", periods=steps * days, freq="15min")
    return pd.Series(y, index=idx)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. non-negative sub-daily (plasma)
    s = _synth_subdaily()
    std = standardize_time_axis(s)
    D1 = fold_from_standardized(std)
    plot_heatmap(D1, signed=False, ax=axes[0, 0], title="1. non-negative (plasma)",
                 cbar_label="value")

    # 2. signed residual centered at zero (seismic, vcenter=0)
    rng = np.random.default_rng(3)
    D2 = rng.standard_normal((96, 60))
    D2[:, 20:24] += 3.0  # a positive streak to show asymmetry handling
    plot_heatmap(D2, signed=True, ax=axes[0, 1],
                 title="2. signed residual (seismic, 0=white)", cbar_label="residual")

    # 3. gaps render as gray
    s_gap = _synth_subdaily(seed=1)
    y = s_gap.to_numpy().copy()
    y[96 * 10 : 96 * 13] = np.nan          # 3 fully missing days
    y[96 * 30 + 40 : 96 * 30 + 60] = np.nan  # a partial-day gap
    s_gap = pd.Series(y, index=s_gap.index)
    std_gap = standardize_time_axis(s_gap)
    D3 = fold_from_standardized(std_gap, trim_empty=False)
    plot_heatmap(D3, signed=False, ax=axes[1, 0],
                 title="3. gaps as gray (plasma)", cbar_label="value")

    # 4. end-to-end: standardize -> decompose -> fold the seasonal component
    s4 = _synth_subdaily(days=90, seed=2)
    std4 = standardize_time_axis(s4)
    built = make_problem(
        std4["y"],
        components=[
            multiperiodic(periods=96, num_harmonics=5, weight=1e-2, role="daily"),
            smooth_trend(weight=1e2, role="trend"),
        ],
    )
    out = solve(built)
    D4 = fold_to_2d(out["values"]["daily"], std4["delta"], trim_empty=False)
    plot_heatmap(D4, signed=True, ax=axes[1, 1],
                 title="4. end-to-end: solved daily component (signed)",
                 cbar_label="daily")

    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
