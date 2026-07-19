# scratch/

Visual review harnesses for the plotting code -- for a **human to eyeball**,
not automated tests. The pytest suite only proves figures render without error;
it cannot judge whether a plot *looks* right (colormap legibility, gaps reading
as gray, zero being neutral in diverging maps, tick labels). These scripts
render galleries to PNGs for manual review.

Generated artifacts (`*.png`, `scratch/*.csv`) are gitignored; the scripts are
kept and committed so outputs can be regenerated and re-reviewed after any change.

## Scripts

- **`visual_heatmap.py`** -- renders `heatmap_gallery.png`, a 4-panel gallery of
  the sub-daily heat-map diagnostic (`signaldecomp.heatmap`): non-negative
  (plasma), signed residual (seismic centered at zero), gaps-as-gray, and an
  end-to-end standardize -> decompose -> fold of a solved component.

  ```
  uv run python scratch/visual_heatmap.py
  ```

  Open the printed PNG path to review. Re-run after any change to
  `heatmap.py` plotting.

- **`visual_reporting.py`** -- renders `decomposition.png` (stacked signal+fit /
  per-role / signed-residual panels) and `stability.png` (snapshot spaghetti +
  normalized RMSD, plus scalar history/delta when an extractor is used) from
  `signaldecomp.reporting`.

  ```
  uv run python scratch/visual_reporting.py
  ```

  Open the printed PNG paths to review. Re-run after any change to
  `reporting.py` plotting.

## Synthetic data + decomposition

- **`make_synthetic_hourly.py`** -- generates a marginally-interesting
  synthetic hourly signal (~3 months): smooth dip-then-rise trend, daily +
  weekly multiperiodic seasonal, a sigmoid response to a smooth exogenous
  covariate `z`, rare large spikes, Gaussian noise, and a 2-day gap. Writes
  two gitignored CSVs to `scratch/`: `synthetic_hourly.csv` (`timestamp, y, z`
  -- what a user sees) and `synthetic_hourly_truth.csv` (the true components,
  for grading recovery). Reproducible (fixed seed).

  ```
  uv run python scratch/make_synthetic_hourly.py
  ```

- **`plot_synthetic_decomp.py`** -- solves the signal with `signaldecomp`
  (`smooth_trend` + daily/weekly `multiperiodic` + `exog_spline` on `z` +
  `sparse` spikes) and renders `synthetic_decomp.png`: recovered components
  (solid) overlaid on ground truth (dashed). Trend and exog are de-meaned for
  a fair shape comparison -- they share an additive-constant DC gauge (see the
  `dc-gauge-freedom-mean-carrying-components` memory note).

  ```
  uv run python scratch/plot_synthetic_decomp.py
  ```
