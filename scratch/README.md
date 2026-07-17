# scratch/

Visual review harnesses for the plotting code -- for a **human to eyeball**,
not automated tests. The pytest suite only proves figures render without error;
it cannot judge whether a plot *looks* right (colormap legibility, gaps reading
as gray, zero being neutral in diverging maps, tick labels). These scripts
render galleries to PNGs for manual review.

Generated images (`*.png`) are gitignored; the scripts are kept and committed so
plots can be re-reviewed after any change.

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
