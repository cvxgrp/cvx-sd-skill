# /// script
# requires-python = ">=3.13"
# dependencies = ["marimo", "signaldecomp", "numpy", "pandas", "matplotlib"]
# ///
"""Worked example (marimo): signal-decomposition degradation-style analysis.

DOMAIN NOTE (read this first)
----------------------------
This is a WORKED EXAMPLE FROM A SPECIFIC DOMAIN (photovoltaic performance),
strictly annotated as such. The signaldecomp library itself is domain-agnostic.

The raw public dataset (PVDAQ system 4) is SUB-DAILY power/irradiance/temp. The
PV *domain* pipeline that turns that into **daily normalized energy** (the y we
decompose) -- temperature correction, PVWatts normalization, daily aggregation
-- lives UPSTREAM in a separate prep script (prep_pvdaq4_daily_y.py) and is NOT
part of signaldecomp. That prep step is the "domain layer"; the skill sits
BELOW it. From the loaded y onward, everything here is the general substrate:
standardize -> build components -> solve -> report/plot/validate.

SCAFFOLD STATUS: the data-load cell currently generates a SYNTHETIC daily y so
the notebook structure and signaldecomp wiring can be validated without the
network/rdtools prep. Replace that cell with the cached real y (see the TODO)
once prep_pvdaq4_daily_y.py has produced it.

Run:  uv run --group examples marimo edit examples/pvdaq4_degradation.py
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import signaldecomp as sd

    return mo, np, pd, sd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Signal-decomposition degradation analysis (PVDAQ system 4)

    **Domain example, annotated.** `y` is *daily normalized energy* produced
    by a PV-domain pipeline upstream (see the module docstring / prep
    script). Everything below operates on `y` with the domain-agnostic
    `signaldecomp` substrate.

    The model decomposes `y` into a residual plus structural components:

    - a **seasonal** component (truncated Fourier over an ~annual period),
    - a **trend** component (linear / smooth / pwl / monotone),
    - the **residual**, under a selectable loss.

    Adjust the controls; the point solve is reactive (sub-second). The
    expensive analyses (CI, stability) are button-gated.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load data (`y` = daily normalized energy)

    **SCAFFOLD:** synthetic daily `y` for now. TODO: replace with the cached
    real series from `prep_pvdaq4_daily_y.py`.
    """)
    return


@app.cell
def _(np, pd):
    # TODO(real-y): replace this synthetic block with:
    #     y_series = pd.read_pickle("examples/pvdaq4_daily_y.pickle")
    #     (cached daily normalized energy produced by prep_pvdaq4_daily_y.py)
    # For now, a synthetic daily series ~4 years: trend + annual season + noise,
    # with a gap, so the annual Fourier and expanding-window have enough data.
    _rng = np.random.default_rng(0)
    _n = 4 * 365
    _idx = pd.date_range("2015-01-01", periods=_n, freq="D")
    _t = np.arange(_n)
    _trend = 1.0 - 0.00005 * _t                      # slow decline
    _season = 0.05 * np.sin(2 * np.pi * _t / 365.2425)
    _y = _trend + _season + 0.02 * _rng.standard_normal(_n)
    _y[600:630] = np.nan                             # a gap
    y_series = pd.Series(_y, index=_idx, name="daily_normalized_energy")
    print(f"y: {len(y_series)} daily points, "
          f"{y_series.index.min().date()} -> {y_series.index.max().date()}, "
          f"{int(y_series.isna().sum())} missing")
    return (y_series,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Standardize the time axis

    Put `y` on a regular daily grid (delta = 86400 s). This is the entry
    point to the general substrate.
    """)
    return


@app.cell
def _(sd, y_series):
    std = sd.standardize_time_axis(y_series, freq="D")
    y = std["y"]
    print(f"delta = {std['delta']} s, freq = {std['freq']}, T = {len(y)}")
    return std, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Model controls
    """)
    return


@app.cell
def _(mo):
    # Define all widgets unconditionally so downstream cells can read .value;
    # display is assembled conditionally in the next cell.
    trend_type = mo.ui.radio(
        options=["linear", "smooth", "pwl", "monotone"],
        value="smooth",
        label="Trend type",
    )
    loss_type = mo.ui.radio(
        options=["l2", "l1", "huber", "quantile"],
        value="l2",
        label="Loss",
    )
    num_harmonics = mo.ui.slider(
        start=1, stop=20, step=1, value=3, label="Num harmonics", show_value=True
    )
    lam_seasonal = mo.ui.slider(
        start=-4.0, stop=2.0, step=0.1, value=-1.0,
        label="log10 lambda_seasonal", show_value=True,
    )
    lam_trend = mo.ui.slider(
        start=-4.0, stop=4.0, step=0.1, value=1.0,
        label="log10 lambda_trend", show_value=True,
    )
    huber_m = mo.ui.slider(
        start=0.01, stop=2.0, step=0.01, value=0.2, label="Huber M", show_value=True
    )
    q_level = mo.ui.slider(
        start=0.05, stop=0.95, step=0.05, value=0.5, label="Quantile q", show_value=True
    )
    return (
        huber_m,
        lam_seasonal,
        lam_trend,
        loss_type,
        num_harmonics,
        q_level,
        trend_type,
    )


@app.cell
def _(
    huber_m,
    lam_seasonal,
    lam_trend,
    loss_type,
    mo,
    num_harmonics,
    q_level,
    trend_type,
):
    # Conditional display: lambda_trend only for smooth/pwl/monotone (linear has
    # no fit weight); q only for quantile; M only for huber.
    _rows = [
        mo.hstack([trend_type, loss_type], justify="start", gap="3rem"),
        mo.hstack([num_harmonics, lam_seasonal], justify="start", gap="2rem"),
    ]
    if trend_type.value in ("smooth", "pwl", "monotone"):
        _rows.append(lam_trend)
    else:
        _rows.append(mo.md("_lambda_trend not used for a linear trend._"))
    if loss_type.value == "huber":
        _rows.append(huber_m)
    if loss_type.value == "quantile":
        _rows.append(q_level)
    controls = mo.vstack(_rows)
    controls
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Solve (reactive)
    """)
    return


@app.cell
def _(
    huber_m,
    lam_seasonal,
    lam_trend,
    loss_type,
    num_harmonics,
    q_level,
    sd,
    trend_type,
    y,
):
    # Build the trend component from the radio choice. Note the per-type weight
    # semantics: linear_trend takes no fit weight; the others take `weight`.
    def _make_trend():
        w = 10.0 ** lam_trend.value
        if trend_type.value == "linear":
            return sd.linear_trend(role="trend")
        if trend_type.value == "smooth":
            return sd.smooth_trend(w, role="trend")
        if trend_type.value == "pwl":
            return sd.pwl_trend(w, role="trend")
        return sd.monotone_trend(weight=w, role="trend")

    def _make_loss():
        if loss_type.value == "l2":
            return sd.l2_loss()
        if loss_type.value == "l1":
            return sd.l1_loss()
        if loss_type.value == "huber":
            return sd.huber_loss(M=huber_m.value)
        return sd.quantile_loss(q=q_level.value)

    # Annual period in samples (daily data -> 365.2425).
    seasonal = sd.multiperiodic(
        365.2425, num_harmonics=num_harmonics.value,
        weight=10.0 ** lam_seasonal.value, role="seasonal",
    )
    built = sd.make_problem(y, components=[seasonal, _make_trend()],
                            residual_loss=_make_loss())
    out = sd.solve(built)
    print(f"status: {out['status']}")
    return (out,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Report
    """)
    return


@app.cell
def _(mo, out, sd, y):
    mo.md(sd.format_report(out, y=y, title="Decomposition"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Decomposition plot
    """)
    return


@app.cell
def _(out, sd, std, y):
    sd.plot_decomposition(out, y=y, index=std["index"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Confidence interval (button-gated)

    Moving-block residual bootstrap on the **final** model. Expensive
    (many solves) -- run explicitly. CI is meaningful only on the specified
    model, never inside a tuning loop.
    """)
    return


@app.cell
def _(mo):
    run_ci = mo.ui.run_button(label="run CI bootstrap")
    run_ci
    return (run_ci,)


@app.cell
def _(
    huber_m,
    lam_seasonal,
    lam_trend,
    loss_type,
    mo,
    num_harmonics,
    q_level,
    run_ci,
    sd,
    trend_type,
    y,
):
    mo.stop(not run_ci.value)

    # Rebuild the same model as a build_fn(sig) for the bootstrap.
    def _build(sig):
        w = 10.0 ** lam_trend.value
        if trend_type.value == "linear":
            trend = sd.linear_trend(role="trend")
        elif trend_type.value == "smooth":
            trend = sd.smooth_trend(w, role="trend")
        elif trend_type.value == "pwl":
            trend = sd.pwl_trend(w, role="trend")
        else:
            trend = sd.monotone_trend(weight=w, role="trend")
        if loss_type.value == "l2":
            loss = sd.l2_loss()
        elif loss_type.value == "l1":
            loss = sd.l1_loss()
        elif loss_type.value == "huber":
            loss = sd.huber_loss(M=huber_m.value)
        else:
            loss = sd.quantile_loss(q=q_level.value)
        seas = sd.multiperiodic(
            365.2425, num_harmonics=num_harmonics.value,
            weight=10.0 ** lam_seasonal.value, role="seasonal",
        )
        return sd.make_problem(sig, components=[seas, trend], residual_loss=loss)

    # Extract the linear slope if present; otherwise the mean level as a stand-in
    # scalar (domain-specific rate math would live here in a real analysis).
    def _extract(o):
        vals = o["values"]
        if "trend_b" in vals:
            return float(vals["trend_b"])
        return float(vals["trend"].mean())

    ci = sd.bootstrap_ci(y, _build, _extract, block_size=365,
                         n_resamples=200, random_state=0)
    print("CI (block=365, 200 resamples):", ci)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Fit stability (button-gated)

    How the estimated components move as the record grows. Component-curve
    snapshots are the default (works for any trend type).
    """)
    return


@app.cell
def _(mo):
    run_stab = mo.ui.run_button(label="run stability")
    run_stab
    return (run_stab,)


@app.cell
def _(
    huber_m,
    lam_seasonal,
    lam_trend,
    loss_type,
    mo,
    num_harmonics,
    q_level,
    run_stab,
    sd,
    trend_type,
    y,
):
    mo.stop(not run_stab.value)

    def _build(sig):
        w = 10.0 ** lam_trend.value
        if trend_type.value == "linear":
            trend = sd.linear_trend(role="trend")
        elif trend_type.value == "smooth":
            trend = sd.smooth_trend(w, role="trend")
        elif trend_type.value == "pwl":
            trend = sd.pwl_trend(w, role="trend")
        else:
            trend = sd.monotone_trend(weight=w, role="trend")
        if loss_type.value == "l2":
            loss = sd.l2_loss()
        elif loss_type.value == "l1":
            loss = sd.l1_loss()
        elif loss_type.value == "huber":
            loss = sd.huber_loss(M=huber_m.value)
        else:
            loss = sd.quantile_loss(q=q_level.value)
        seas = sd.multiperiodic(
            365.2425, num_harmonics=num_harmonics.value,
            weight=10.0 ** lam_seasonal.value, role="seasonal",
        )
        return sd.make_problem(sig, components=[seas, trend], residual_loss=loss)

    stability = sd.expanding_window_stability(y, _build, min_window=365, step=180)
    stab_fig = sd.plot_stability(stability, role="trend")
    stab_fig
    return


if __name__ == "__main__":
    app.run()
