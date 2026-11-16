# Preparing the time axis

The model consumes a one-dimensional `y` on a regular grid, a sampling interval
`delta`, and optionally an index retained by the caller for reporting. Missing
observations remain on that grid as non-finite values; do not delete them and
compress elapsed time.

## Fixed-rate sub-daily and daily input

`standardize_time_axis` returns a dictionary, not a tuple. It accepts a pandas
Series or DataFrame, sorts timestamps, keeps the first duplicate timestamp,
and constructs a midnight-anchored grid spanning whole days. Inspect duplicate
records before calling it if aggregation or another conflict policy is needed.

```python
import numpy as np
import pandas as pd
from signaldecomp import standardize_time_axis, fold_from_standardized

index = pd.date_range("2025-01-01", periods=48, freq="h")
raw = pd.Series(np.sin(np.arange(48) * 2 * np.pi / 24), index=index)
raw.iloc[12] = np.nan
prepared = standardize_time_axis(raw, freq="h")
y, delta, index = prepared["y"], prepared["delta"], prepared["index"]
assert delta == 3600.0 and len(y) == 48
folded = fold_from_standardized(prepared)
```

Supply `value_column` for a multicolumn DataFrame and `datetime_column` if
timestamps are stored in a column. An explicit `freq` is authoritative. Without
it, the modal inter-sample gap is inferred, clustered for jitter, and snapped
to a standard rate when close enough. Inspect `prepared["scan_rates"]` for
multiple sampling rates before accepting one grid for the record.

The helper uses nearest reindexing with `limit=1`; it is not an exact
timestamp join or a configurable distance-tolerance matcher. A nearby grid row
can inherit an observation. Inspect the prepared series, gaps, coverage, and
alignment; if exact missingness or causal sampling matters, construct the grid
and alignment explicitly in pandas instead. Sub-second or unusual cadences
also warrant explicit preparation rather than relying on inferred rounding.

## Time zones and calendar data

Convert to UTC or a fixed local standard-time zone before preparation. The
helper strips timezone information; it does not convert DST wall time into a
fixed-duration physical axis. A fixed-day grid cannot represent both 23-hour
and 25-hour local days. Use `Timedelta.total_seconds()`, not `.seconds`, when
computing spacing yourself.

Weekly, monthly, and other calendar cadences should be prepared by the caller.
For equally spaced weekly observations, `delta = 7 * 86400` is appropriate.
Monthly timestamps are not uniformly spaced in seconds: choose an explicit
calendar-index model and matching period units, or aggregate to a fixed grid.
Do not pass variable elapsed times to a model that assumes constant `delta`.

## Heat maps and aligned drivers

`fold_to_2d` and `fold_from_standardized` fold whole days into time-of-day rows
and day columns. The interval must divide 86400 seconds evenly. Compare the
raw signal, fitted roles, and residual using the same fold and day axis. See
[diagnostics.md](diagnostics.md) for what to inspect.

Align exogenous inputs to the same grid before constructing offset designs.
`make_offset_basis` shifts by samples, not timestamps. For fitted spline knots
or whitening, determine the final intersection of observation and all driver
availability masks first. Held-out signal values must not enter preprocessing;
see [exogenous-numerics.md](exogenous-numerics.md).
