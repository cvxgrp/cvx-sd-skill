# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.14",
#     "matplotlib>=3.11",
#     "numpy>=2.5",
#     "pandas>=3.0",
#     "signaldecomp",
# ]
# ///
"""Marimo notebook: compare trend classes for weekly gasoline prices.

Run from the repository root:

    uv run python -m marimo edit examples/gasoline_price_trends.py

The notebook uses the bundled EIA/FRED GASREGW source series.
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import signaldecomp as sd

    return Path, mo, np, pd, plt, sd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Which trend class best describes weekly gasoline prices?

    This notebook compares four convex claims about the latent price trend:

    - **linear:** one slope over the entire record;
    - **smooth:** curvature is mean-square-small;
    - **piecewise linear (PWL):** slope changes are sparse;
    - **piecewise constant (PWC):** level changes are sparse.

    Every model also contains the same annual Fourier component and is fit to
    **log price**, so its components have multiplicative interpretations on the
    dollar-per-gallon scale.

    The comparison deliberately keeps two questions separate:

    1. Can the model reconstruct a masked calendar year?
    2. Does its full-data structural result support the trend claim?

    Lowest training residual is not the selection rule.
    """)
    return


@app.cell
def _(Path, np, pd):
    data_path = Path("examples/gasoline_prices.csv")
    price_frame = (
        pd.read_csv(data_path, parse_dates=["observation_date"])
        .set_index("observation_date")
        .sort_index()
    )
    price = price_frame["GASREGW"].astype(float)
    log_price = np.log(price.to_numpy())
    observed = ~np.isnan(log_price)
    return data_path, log_price, observed, price, price_frame


@app.cell
def _(data_path, mo, observed, price_frame):
    mo.md(f"""
    **Input:** `{data_path}`<br>
    **Record:** {price_frame.index.min().date()}–{price_frame.index.max().date()}<br>
    **Grid:** {len(price_frame):,} weekly samples; {int(observed.sum()):,}
    observed; {int((~observed).sum())} missing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison controls

    Four harmonics and seasonal weight 0.15 are the defaults selected in the
    preceding seasonal analysis. Each trend class gets its **own** weight grid:
    the numerical scales of global L2² curvature and normalized local L1
    penalties are not interchangeable.

    For a weighted trend, selection uses the largest weight whose holdout error
    is within `δ` of that class's minimum—the practical "street fighting"
    rule. Linear has no trend weight.
    """)
    return


@app.cell
def _(mo):
    harmonic_control = mo.ui.dropdown(
        options=[1, 2, 4, 6, 8],
        value=4,
        label="Annual harmonics",
    )
    seasonal_weight_control = mo.ui.slider(
        start=-3.0,
        stop=0.5,
        step=0.05,
        value=-0.825,
        label="log10 seasonal weight",
        show_value=True,
    )
    tolerance_control = mo.ui.slider(
        start=0.0,
        stop=0.05,
        step=0.005,
        value=0.01,
        label="holdout δ",
        show_value=True,
    )
    mo.hstack(
        [harmonic_control, seasonal_weight_control, tolerance_control],
        justify="start",
        gap="2rem",
    )
    return harmonic_control, seasonal_weight_control, tolerance_control


@app.cell
def _(sd):
    annual_period = sd.period_samples(sd.SECONDS_PER_YEAR, sd.SECONDS_PER_WEEK)
    trend_weight_grids = {
        "smooth": [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0],
        "pwl": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
        "pwc": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
    }
    return annual_period, trend_weight_grids


@app.cell
def _(annual_period, harmonic_control, np, sd, seasonal_weight_control):
    seasonal_weight = 10.0 ** seasonal_weight_control.value

    def make_trend(kind, weight=None):
        if kind == "linear":
            return sd.linear_trend(role="trend")
        if kind == "smooth":
            return sd.smooth_trend(weight, order=2, role="trend")
        if kind == "pwl":
            return sd.pwl_trend(weight, role="trend")
        if kind == "pwc":
            return sd.pwc_trend(weight, role="trend")
        raise ValueError(f"unknown trend class: {kind}")

    def build_model(signal, kind, weight=None):
        return sd.make_problem(
            np.asarray(signal, dtype=float),
            components=[
                make_trend(kind, weight),
                sd.multiperiodic(
                    annual_period,
                    num_harmonics=harmonic_control.value,
                    weight=seasonal_weight,
                    role="seasonal",
                ),
            ],
        )

    return (build_model,)


@app.cell
def _(price_frame):
    holdout_start = price_frame.index.searchsorted("2018-01-01")
    holdout_stop = price_frame.index.searchsorted("2019-01-01")
    holdout_slice = slice(holdout_start, holdout_stop)
    return (holdout_slice,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Held-out reconstruction paths

    The calendar year 2018 is replaced by missing values during these fits.
    Scores measure reconstruction of its known log prices. Final models later
    in the notebook are rebuilt using **all** observed weeks.
    """)
    return


@app.cell
def _(
    build_model,
    holdout_slice,
    log_price,
    np,
    pd,
    sd,
    tolerance_control,
    trend_weight_grids,
):
    holdout_rows = []
    selected_weights = {"linear": None}

    linear_selection = sd.holdout_select(
        log_price,
        {"linear": lambda signal: build_model(signal, "linear")},
        holdout_slice=holdout_slice,
    )
    holdout_rows.append(
        {
            "trend_class": "linear",
            "weight": np.nan,
            "holdout_log_rmse": linear_selection["scores"]["linear"],
            "selected": True,
        }
    )

    for _kind, _weights in trend_weight_grids.items():
        _candidates = {
            f"{_weight:g}": (
                lambda signal, kind=_kind, weight=_weight: build_model(
                    signal, kind, weight
                )
            )
            for _weight in _weights
        }
        _selection = sd.holdout_select(
            log_price,
            _candidates,
            holdout_slice=holdout_slice,
        )
        _scores = {
            float(_weight): float(_score)
            for _weight, _score in _selection["scores"].items()
        }
        _minimum = min(_scores.values())
        _eligible = [
            _weight
            for _weight, _score in _scores.items()
            if _score <= _minimum * (1 + tolerance_control.value)
        ]
        _selected_weight = max(_eligible)
        selected_weights[_kind] = _selected_weight
        for _weight, _score in _scores.items():
            holdout_rows.append(
                {
                    "trend_class": _kind,
                    "weight": _weight,
                    "holdout_log_rmse": _score,
                    "selected": _weight == _selected_weight,
                }
            )

    holdout_results = pd.DataFrame(holdout_rows)
    return holdout_results, selected_weights


@app.cell
def _(holdout_results, plt):
    _weighted = holdout_results[holdout_results["trend_class"] != "linear"]
    _linear_score = holdout_results.loc[
        holdout_results["trend_class"] == "linear", "holdout_log_rmse"
    ].iloc[0]
    _fig, _axis = plt.subplots(figsize=(9, 4.8))
    for _kind, _group in _weighted.groupby("trend_class", sort=False):
        _axis.plot(
            _group["weight"],
            _group["holdout_log_rmse"],
            marker="o",
            label=_kind,
        )
        _chosen = _group[_group["selected"]]
        _axis.scatter(
            _chosen["weight"],
            _chosen["holdout_log_rmse"],
            s=110,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            zorder=4,
        )
    _axis.axhline(
        _linear_score,
        color="0.35",
        ls="--",
        lw=1,
        label=f"linear ({_linear_score:.3f})",
    )
    _axis.set_xscale("log")
    _axis.set_xlabel("trend regularization weight")
    _axis.set_ylabel("2018 holdout log RMSE")
    _axis.set_title("Holdout paths (rings mark selected weights)")
    _axis.legend(frameon=False, ncol=2)
    _axis.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Full-data fits and structural diagnostics

    The selected specifications are now refit over the complete record.

    Structural counts use explicit effect-size thresholds:

    - a PWL knot must change the weekly log-price slope by at least 0.1
      percentage point;
    - a PWC jump must change the fitted price level by at least 1%.

    These thresholds make the null result reachable. They are diagnostics, not
    claims that every detected knot or jump is economically meaningful.
    """)
    return


@app.cell
def _(build_model, log_price, sd, selected_weights):
    fitted_models = {
        _kind: sd.solve(build_model(log_price, _kind, _weight))
        for _kind, _weight in selected_weights.items()
    }
    return (fitted_models,)


@app.cell
def _(
    fitted_models,
    holdout_results,
    np,
    observed,
    pd,
    price_frame,
    selected_weights,
):
    summary_rows = []
    pwl_knot_threshold = np.log1p(0.001)
    pwc_jump_threshold = np.log1p(0.01)
    _smooth_curvature_relative_threshold = 0.01

    for _kind, _out in fitted_models.items():
        _trend = np.asarray(_out["values"]["trend"])
        _seasonal = np.asarray(_out["values"]["seasonal"])
        _residual = np.asarray(_out["values"]["residual"])
        _weekly_effect = (
            pd.DataFrame(
                {
                    "iso_week": price_frame.index.isocalendar().week.astype(int),
                    "effect": 100 * np.expm1(_seasonal),
                }
            )
            .groupby("iso_week")["effect"]
            .mean()
        )
        _selected = holdout_results[
            (holdout_results["trend_class"] == _kind)
            & holdout_results["selected"]
        ].iloc[0]
        _complexity = np.nan
        _complexity_label = "—"
        if _kind == "smooth":
            _curvature = np.diff(_trend, n=2)
            _curvature_cutoff = (
                _smooth_curvature_relative_threshold
                * np.max(np.abs(_curvature))
            )
            _material_curvature = _curvature[
                np.abs(_curvature) >= _curvature_cutoff
            ]
            _complexity = int(
                np.count_nonzero(
                    np.signbit(_material_curvature[1:])
                    != np.signbit(_material_curvature[:-1])
                )
            )
            _complexity_label = f"{_complexity} curvature reversals"
        elif _kind == "pwl":
            _complexity = int(
                (np.abs(np.diff(_trend, n=2)) >= pwl_knot_threshold).sum()
            )
            _complexity_label = f"{_complexity} knots"
        elif _kind == "pwc":
            _complexity = int(
                (np.abs(np.diff(_trend)) >= pwc_jump_threshold).sum()
            )
            _complexity_label = f"{_complexity} jumps"
        elif _kind == "linear":
            _complexity = 2
            _complexity_label = "2 coefficients"

        summary_rows.append(
            {
                "trend_class": _kind,
                "selected_weight": selected_weights[_kind],
                "holdout_log_rmse": _selected["holdout_log_rmse"],
                "full_data_log_rmse": np.sqrt(
                    np.mean(_residual[observed] ** 2)
                ),
                "annual_amplitude_pct": 100
                * np.expm1(_seasonal.max() - _seasonal.min()),
                "peak_iso_week": int(_weekly_effect.idxmax()),
                "trough_iso_week": int(_weekly_effect.idxmin()),
                "structural_complexity": _complexity,
                "complexity_label": _complexity_label,
            }
        )
    model_summary = pd.DataFrame(summary_rows).sort_values("holdout_log_rmse")
    return model_summary, pwc_jump_threshold, pwl_knot_threshold


@app.cell
def _(model_summary):
    _display_summary = model_summary[
        [
            "trend_class",
            "selected_weight",
            "holdout_log_rmse",
            "full_data_log_rmse",
            "annual_amplitude_pct",
            "peak_iso_week",
            "trough_iso_week",
            "complexity_label",
        ]
    ].rename(
        columns={
            "trend_class": "Trend class",
            "selected_weight": "Selected weight",
            "holdout_log_rmse": "Holdout log RMSE",
            "full_data_log_rmse": "Full-data log RMSE",
            "annual_amplitude_pct": "Annual amplitude (%)",
            "peak_iso_week": "Peak ISO week",
            "trough_iso_week": "Trough ISO week",
            "complexity_label": "Complexity",
        }
    )
    _display_summary.style.format(
        {
            "Selected weight": lambda value: "—" if value is None else f"{value:g}",
            "Holdout log RMSE": "{:.4f}",
            "Full-data log RMSE": "{:.4f}",
            "Annual amplitude (%)": "{:.1f}%",
        }
    ).set_table_styles(
        [
            {
                "selector": "th.col_heading",
                "props": [
                    ("white-space", "normal"),
                    ("line-height", "1.2"),
                    ("min-width", "5rem"),
                    ("max-width", "8rem"),
                    ("text-align", "center"),
                ],
            }
        ]
    )
    return


@app.cell(hide_code=True)
def plot_controls(mo):
    plot_trends_control = mo.ui.multiselect(
        options=["linear", "smooth", "pwl", "pwc"],
        value=["linear", "smooth", "pwl", "pwc"],
        label="Trends and residuals to plot",
    )
    reset_plot_weights_button = mo.ui.button(
        value=0,
        on_click=lambda count: count + 1,
        label="Reset",
        tooltip="Restore the weights selected by holdout validation",
    )
    plot_trends_control
    return plot_trends_control, reset_plot_weights_button


@app.cell(hide_code=True)
def plot_weight_inputs(
    mo,
    np,
    reset_plot_weights_button,
    selected_weights,
    trend_weight_grids,
):
    _reset_generation = reset_plot_weights_button.value
    smooth_plot_weight_control = mo.ui.slider(
        start=np.floor(np.log10(min(trend_weight_grids["smooth"]))),
        stop=8,
        step=0.05,
        value=np.round(np.log10(selected_weights["smooth"]), 2),
        label="log10 smooth weight",
        show_value=True,
    )
    pwl_plot_weight_control = mo.ui.slider(
        start=np.floor(np.log10(min(trend_weight_grids["pwl"]))),
        stop=5,
        step=0.05,
        value=np.round(np.log10(selected_weights["pwl"]), 2),
        label="log10 PWL weight",
        show_value=True,
    )
    pwc_plot_weight_control = mo.ui.slider(
        start=np.floor(np.log10(min(trend_weight_grids["pwc"]))),
        stop=np.ceil(np.log10(max(trend_weight_grids["pwc"]))),
        step=0.05,
        value=np.round(np.log10(selected_weights["pwc"]), 2),
        label="log10 PWC weight",
        show_value=True,
    )
    return (
        pwc_plot_weight_control,
        pwl_plot_weight_control,
        smooth_plot_weight_control,
    )


@app.cell(hide_code=True)
def plot_weight_controls(
    mo,
    plot_trends_control,
    pwc_plot_weight_control,
    pwl_plot_weight_control,
    reset_plot_weights_button,
    smooth_plot_weight_control,
):
    _visible_weight_controls = [
        _control
        for _kind, _control in [
            ("smooth", smooth_plot_weight_control),
            ("pwl", pwl_plot_weight_control),
            ("pwc", pwc_plot_weight_control),
        ]
        if _kind in plot_trends_control.value
    ]
    mo.hstack(
        [*_visible_weight_controls, reset_plot_weights_button],
        justify="start",
        gap="2rem",
    ) if _visible_weight_controls else mo.md(
        "*Linear has no trend regularization weight.*"
        if "linear" in plot_trends_control.value
        else ""
    )
    return


@app.cell(hide_code=True)
def plot_models(
    build_model,
    fitted_models,
    log_price,
    pwc_plot_weight_control,
    pwl_plot_weight_control,
    sd,
    smooth_plot_weight_control,
):
    plotted_models = {
        "linear": fitted_models["linear"],
        "smooth": sd.solve(
            build_model(
                log_price,
                "smooth",
                10.0 ** smooth_plot_weight_control.value,
            )
        ),
        "pwl": sd.solve(
            build_model(
                log_price,
                "pwl",
                10.0 ** pwl_plot_weight_control.value,
            )
        ),
        "pwc": sd.solve(
            build_model(
                log_price,
                "pwc",
                10.0 ** pwc_plot_weight_control.value,
            )
        ),
    }
    return (plotted_models,)


@app.cell
def _(np, plot_trends_control, plotted_models, plt, price, price_frame):
    _fig, _axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    _axes[0].plot(
        price_frame.index,
        price,
        color="0.65",
        lw=0.7,
        label="observed",
    )
    for _kind, _out in plotted_models.items():
        if _kind in plot_trends_control.value:
            _trend_price = np.exp(np.asarray(_out["values"]["trend"]))
            _axes[0].plot(price_frame.index, _trend_price, lw=1.5, label=_kind)
            _residual_pct = 100 * np.expm1(np.asarray(_out["values"]["residual"]))
            _axes[1].plot(
                price_frame.index,
                _residual_pct,
                lw=0.7,
                alpha=0.8,
                label=_kind,
            )
    _axes[0].set_ylabel("trend ($/gallon)")
    _axes[0].set_title("Full-data trend estimates")
    _axes[0].legend(frameon=False, ncol=5)
    _axes[1].axhline(0, color="black", lw=0.6)
    _axes[1].set_ylabel("residual (%)")
    _axes[1].set_title("Residuals expose the consequences of each trend claim")
    for _axis in _axes:
        _axis.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## L1 structural paths

    PWL and PWC weights affect both reconstruction and the number of detected
    structural changes. The useful choice is rarely "whichever minimizes
    residual error." A structural rule can instead require a plausible knot or
    jump count and a minimum effect size, then refit the chosen model on all
    data.

    PWL is the more plausible description here, but its raw L1 result can
    shrink slope changes and retain weak knots. An IRL1 polish would be a
    sensible escalation after choosing the initial PWL weight.
    """)
    return


@app.cell
def _(
    build_model,
    log_price,
    np,
    pd,
    pwc_jump_threshold,
    pwl_knot_threshold,
    sd,
    trend_weight_grids,
):
    structural_rows = []
    for _kind, _threshold, _difference_order in (
        ("pwl", pwl_knot_threshold, 2),
        ("pwc", pwc_jump_threshold, 1),
    ):
        for _weight in trend_weight_grids[_kind]:
            _out = sd.solve(build_model(log_price, _kind, _weight))
            _trend = np.asarray(_out["values"]["trend"])
            _residual = np.asarray(_out["values"]["residual"])
            structural_rows.append(
                {
                    "trend_class": _kind,
                    "weight": _weight,
                    "change_count": int(
                        (
                            np.abs(np.diff(_trend, n=_difference_order))
                            >= _threshold
                        ).sum()
                    ),
                    "full_data_log_rmse": np.sqrt(np.nanmean(_residual**2)),
                }
            )
    structural_paths = pd.DataFrame(structural_rows)
    return (structural_paths,)


@app.cell
def _(plt, selected_weights, structural_paths):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for _kind, _group in structural_paths.groupby("trend_class", sort=False):
        _axes[0].plot(
            _group["weight"],
            _group["change_count"],
            marker="o",
            label=_kind,
        )
        _axes[1].plot(
            _group["weight"],
            _group["full_data_log_rmse"],
            marker="o",
            label=_kind,
        )
        for _axis in _axes:
            _axis.axvline(
                selected_weights[_kind],
                color=_axis.lines[-1].get_color(),
                ls=":",
                lw=1,
            )
    _axes[0].set_ylabel("effect-size-qualified changes")
    _axes[0].set_title("Structural complexity path")
    _axes[1].set_ylabel("full-data log RMSE")
    _axes[1].set_title("Training fit path")
    for _axis in _axes:
        _axis.set_xscale("log")
        _axis.set_xlabel("trend regularization weight")
        _axis.legend(frameon=False)
        _axis.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo, model_summary):
    _best_holdout = model_summary.iloc[0]
    _pwl = model_summary[model_summary["trend_class"] == "pwl"].iloc[0]
    _smooth = model_summary[model_summary["trend_class"] == "smooth"].iloc[0]
    mo.md(
        f"""
        ## Reading the comparison

        - **{_best_holdout['trend_class'].upper()}** has the lowest 2018
          holdout error, but that result alone does not establish that gasoline
          prices truly move through discrete constant regimes.
        - **PWL** materially improves full-data residual RMS over the smooth
          trend and yields {_pwl['complexity_label']}; those breakpoints are
          candidates for structural review, not automatic events.
        - The **smooth** model is less structurally committal and gives the
          clearest baseline for seasonal amplitude and timing.
        - Seasonal amplitude changes from
          {_smooth['annual_amplitude_pct']:.1f}% under smooth to
          {_pwl['annual_amplitude_pct']:.1f}% under PWL. This sensitivity is
          exactly why trend class belongs in model exploration.

        A productive next step is to inspect PWL knot dates, impose an explicit
        admissible knot-count/effect-size rule, and only then apply IRL1
        polishing. Holdout error and structural plausibility answer different
        questions; both should remain visible.
        """
    )
    return


if __name__ == "__main__":
    app.run()
