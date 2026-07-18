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
PV *domain* pipeline that turns that into daily normalized energy (the y we
decompose) lives UPSTREAM in a separate prep script (prep_pvdaq4_daily_y.py) and
is NOT part of signaldecomp. Likewise, turning a solved trend curve into a
percent-per-year DEGRADATION RATE is domain math, kept in a sibling module
(pv_domain.py). Everything imported from `signaldecomp` is the general
substrate; everything imported from `pv_domain` is the PV domain layer.

Run:  uv run --group examples marimo edit examples/pvdaq4_degradation.py
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt

    import signaldecomp as sd
    import pv_domain as pv  # PV DOMAIN LAYER (sibling module), not signaldecomp

    return mo, pd, plt, pv, sd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Signal-decomposition degradation analysis (PVDAQ system 4)

    **Domain example, annotated.** `y` is *daily normalized energy* from a
    PV-domain pipeline (prep script). The decomposition and all substrate
    operations below are domain-agnostic `signaldecomp`; the %/yr
    degradation rate is PV domain math from `pv_domain`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load data (`y` = daily normalized energy)

    Cached artifact from `prep_pvdaq4_daily_y.py` (gitignored). Regenerate:
    `uv run examples/prep_pvdaq4_daily_y.py`.
    """)
    return


@app.cell
def _(mo, pd):
    y_series = pd.read_pickle("examples/pvdaq4_daily_y.pickle")
    mo.md(
        f"**y:** {len(y_series)} daily points, "
        f"{y_series.index.min().date()} \u2192 {y_series.index.max().date()}, "
        f"{int(y_series.isna().sum())} missing"
    )
    return (y_series,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Standardize the time axis
    """)
    return


@app.cell
def _(mo, sd, y_series):
    std = sd.standardize_time_axis(y_series, freq="D")
    y = std["y"]
    mo.md(
        f"**standardized:** delta = {std['delta']:.0f} s, "
        f"freq = {std['freq']}, T = {len(y)}"
    )
    return std, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Model controls
    """)
    return


@app.cell
def _(mo):
    # Widgets defined unconditionally so downstream cells can read .value;
    # display is assembled conditionally in the next cell.
    trend_type = mo.ui.radio(
        options=["linear", "monotone", "smooth", "pwl"],
        value="linear",
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
    log_transform = mo.ui.switch(label="Log transform (multiplicative)", value=False)
    return (
        huber_m,
        lam_seasonal,
        lam_trend,
        log_transform,
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
    log_transform,
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
    _rows.append(log_transform)
    controls = mo.vstack(_rows)
    controls
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Prepare input (optional log transform)

    With log transform on, the decomposition is solved on `log(y)`
    (multiplicative model); everything downstream operates on `y_model`.
    """)
    return


@app.cell
def _(log_transform, mo, sd, y):
    y_model = sd.prepare_input(y, log_transform=log_transform.value)
    mo.md(
        f"**input:** {'log(y) (multiplicative)' if log_transform.value else 'y (additive)'}"
    )
    return (y_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Solve (reactive)
    """)
    return


@app.cell
def _(
    huber_m,
    lam_seasonal,
    lam_trend,
    loss_type,
    mo,
    num_harmonics,
    q_level,
    sd,
    trend_type,
    y_model,
):
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

    seasonal = sd.multiperiodic(
        365.2425, num_harmonics=num_harmonics.value,
        weight=10.0 ** lam_seasonal.value, role="seasonal",
    )
    built = sd.make_problem(y_model, components=[seasonal, _make_trend()],
                            residual_loss=_make_loss())
    out = sd.solve(built)
    mo.md(f"**solve status:** {out['status']}")
    return (out,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Reports (substrate + domain)
    """)
    return


@app.cell
def _(mo, out, sd, y_model):
    mo.md(sd.format_report(out, y=y_model, title="Decomposition (substrate)"))
    return


@app.cell
def _(log_transform, loss_type, mo, num_harmonics, out, pv, std, trend_type):
    samples_per_year = pv.samples_per_year_from_delta(std["delta"])
    rate = pv.overall_degradation_rate(
        out, samples_per_year, log_space=log_transform.value
    )
    _meta = {
        "trend_type": trend_type.value,
        "loss": loss_type.value,
        "num_harmonics": num_harmonics.value,
        "log_transform": log_transform.value,
    }
    mo.md(pv.format_degradation_report(rate, _meta))
    return (samples_per_year,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Decomposition plot

    When log transform is on, components are shown in **log space** (the
    model is additive there). Use `signaldecomp.recover_components` for an
    original-domain (multiplicative) view.
    """)
    return


@app.cell
def _(log_transform, out, plt, sd, std, y):
    # Uniform for both modes: recover_frame back-transforms to the original
    # domain (undoing the log transform when on), and residual_ref sets the
    # residual panel baseline (0 additive, 1 multiplicative factor).
    sd.plot_decomposition(
        df=sd.recover_frame(
            out, log_transform=log_transform.value, index=std["index"], y=y
        ),
        residual_ref=1.0 if log_transform.value else 0.0,
    )
    _fig = plt.gcf()
    _fig.get_axes()[0]\
        .set_ylim(0.6, 1)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Confidence interval (button-gated)

    Moving-block residual bootstrap of the **overall degradation rate** on
    the final model. Expensive (many solves); CI is meaningful only on the
    specified model, never inside a tuning loop.
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
    log_transform,
    loss_type,
    mo,
    num_harmonics,
    pv,
    q_level,
    run_ci,
    samples_per_year,
    sd,
    trend_type,
    y_model,
):
    mo.stop(not run_ci.value)

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

    # The CI measures the DOMAIN quantity: overall %/yr degradation rate.
    def _extract(o):
        return pv.overall_degradation_rate(
            o, samples_per_year, log_space=log_transform.value
        )
    _n_samples = 400
    with mo.status.spinner(
        subtitle=f"Bootstrapping CI ({_n_samples} resamples, block=365) ..."
    ):
        ci = sd.bootstrap_ci(y_model, _build, _extract, block_size=365,
                             n_resamples=_n_samples, random_state=0)
    _lines = ["**Bootstrap CI** of overall degradation rate (%/yr):", ""]
    for _k, _v in ci.items():
        _lines.append(f"- [{_v[0]:+.3f}, {_v[1]:+.3f}] %/yr")
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Fit stability (button-gated)

    How the trend moves as the record grows. Component-curve snapshots are
    the default (works for any trend type).
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
    y_model,
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

    with mo.status.spinner(subtitle="Solving expanding windows ...") as _spinner:
        stability = sd.expanding_window_stability(
            y_model, _build, min_window=365, step=180
        )
        _spinner.update(subtitle="Plotting stability ...")
        stab_fig = sd.plot_stability(stability, role="trend")
    stab_fig
    return


if __name__ == "__main__":
    app.run()
