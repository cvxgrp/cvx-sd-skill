# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.15",
#     "matplotlib>=3.10",
#     "numpy>=2.0",
#     "pandas>=3.0",
#     "scipy==1.18.0",
#     "signaldecomp @ file:///Users/bmeyers/github/agent-test/gasoline-analysis/.agents/skills/cvx-signal-decomposition",
# ]
# ///
"""Reference notebook for comparing convex gasoline-price trend models."""

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    import signaldecomp as sd
    from signaldecomp import basis as sd_basis

    return Path, mo, np, pd, plt, sd, sd_basis, sp, splu


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Which trend class best describes weekly gasoline prices?

    We compare four convex claims about the latent price trend:

    - **affine in model space:** linear price in level mode, but an **exponential price trend** in log mode;
    - **smooth:** curvature is mean-square-small;
    - **piecewise linear (PWL):** slope changes are sparse;
    - **piecewise constant (PWC):** level changes are sparse.

    Every model includes the same annual Fourier component. The **scale control is
    part of the model**:

    - log price treats changes as proportional and gives multiplicative
      components;
    - price level treats errors and changes in dollars per gallon.

    Holdout reconstruction, full-data structure, and trend–seasonal competition
    answer different questions and remain visible separately. Lowest training
    residual is not a model-selection rule.
    """)
    return


@app.cell
def _(Path, np, pd):
    data_path = Path(__file__).with_name("gasoline_prices.csv")
    price_frame = (
        pd.read_csv(data_path, parse_dates=["observation_date"])
        .set_index("observation_date")
        .sort_index()
    )
    price_series = price_frame["GASREGW"].astype(float)
    price_values = price_series.to_numpy(dtype=float)
    observed_mask = np.isfinite(price_values)
    return data_path, observed_mask, price_frame, price_series, price_values


@app.cell
def _(data_path, mo, observed_mask, price_frame):
    mo.md(
        f"""
        **Input:** `{data_path.name}` ·
        **Record:** {price_frame.index.min().date()}–{price_frame.index.max().date()} ·
        **Grid:** {len(price_frame):,} weekly samples,
        {int(observed_mask.sum()):,} observed,
        {int((~observed_mask).sum())} missing
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model and validation controls

    The default is log price because a fixed percentage move is more comparable
    across decades than a fixed dollar move. Flip the control to see what that
    assumption changes.

    Each weighted trend gets its own candidate grid. Within each family, the
    validation rule picks the **strongest** weight whose median error is no more
    than `δ × 100%` above that family's minimum median error. This conservative
    reconstruction rule supplies a starting point—not an automatic structural
    verdict for PWL or PWC.
    """)
    return


@app.cell
def _(mo):
    transform_control = mo.ui.radio(
        options=["Log price (recommended)", "Price level"],
        value="Log price (recommended)",
        label="Model scale",
    )
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
        label="near-minimum δ",
        show_value=True,
    )
    mo.vstack(
        [
            transform_control,
            mo.hstack(
                [harmonic_control, seasonal_weight_control, tolerance_control],
                justify="start",
                gap="2rem",
            ),
        ]
    )
    return (
        harmonic_control,
        seasonal_weight_control,
        tolerance_control,
        transform_control,
    )


@app.cell
def _(np, price_values, transform_control):
    use_log_scale = transform_control.value.startswith("Log")
    model_signal = np.log(price_values) if use_log_scale else price_values.copy()

    def to_price_scale(model_values):
        values = np.asarray(model_values, dtype=float)
        return np.exp(values) if use_log_scale else values

    scale_label = "log price" if use_log_scale else "price level"
    residual_label = "residual (%)" if use_log_scale else "residual ($/gallon)"
    affine_trend_label = "exponential" if use_log_scale else "linear"
    curvature_space_label = "log-price fit space" if use_log_scale else "price-level fit space"
    return (
        affine_trend_label,
        curvature_space_label,
        model_signal,
        residual_label,
        scale_label,
        to_price_scale,
        use_log_scale,
    )


@app.cell
def _(sd):
    annual_period_samples = sd.period_samples(
        sd.SECONDS_PER_YEAR, sd.SECONDS_PER_WEEK
    )
    trend_weight_grids = {
        "smooth": [1.0, 10.0, 100.0, 300.0, 3000.0],
        "pwl": [0.1, 0.3, 1.0, 3.0, 30.0, 300.0],
        "pwc": [0.1, 0.3, 1.0, 3.0, 30.0, 300.0],
    }
    return annual_period_samples, trend_weight_grids


@app.cell
def _(annual_period_samples, harmonic_control, sd, seasonal_weight_control):
    seasonal_weight = 10.0 ** seasonal_weight_control.value

    def make_trend(trend_class, weight=None):
        if trend_class == "linear":
            return sd.linear_trend(role="trend")
        if trend_class == "smooth":
            return sd.smooth_trend(weight, order=2, role="trend")
        if trend_class == "pwl":
            return sd.pwl_trend(weight, role="trend")
        if trend_class == "pwc":
            return sd.pwc_trend(weight, role="trend")
        raise ValueError(f"unknown trend class: {trend_class}")

    def build_model(signal, trend_class, weight=None):
        return sd.make_problem(
            signal,
            components=[
                make_trend(trend_class, weight),
                sd.multiperiodic(
                    annual_period_samples,
                    num_harmonics=harmonic_control.value,
                    weight=seasonal_weight,
                    role="seasonal",
                ),
            ],
            residual_loss="l2",
        )

    return build_model, seasonal_weight


@app.cell
def _(price_frame):
    validation_years = [2004, 2012, 2018]
    validation_blocks = {
        year: slice(
            price_frame.index.searchsorted(f"{year}-01-01"),
            price_frame.index.searchsorted(f"{year + 1}-01-01"),
        )
        for year in validation_years
    }
    return validation_blocks, validation_years


@app.cell(hide_code=True)
def _(mo, validation_years):
    mo.md(f"""
    ## Multi-era held-out reconstruction

    Calendar years **{", ".join(map(str, validation_years))}** are masked one at a
    time. Models are fit on the remaining observations and scored against the
    known held-out prices.

    Scores are always reported in **dollars per gallon**, even when the model is
    fit in log space, so the transform choices share a common outcome scale. Thin
    lines show individual years and thick lines show their median. With only three
    years, the median is robust to one difficult episode but can also conceal it;
    read the individual traces before treating a low median as stable.
    """)
    return


@app.cell
def _(
    build_model,
    model_signal,
    np,
    pd,
    price_values,
    sd,
    to_price_scale,
    tolerance_control,
    trend_weight_grids,
    validation_blocks,
):
    validation_rows = []
    candidate_weights = {"linear": [None], **trend_weight_grids}

    for validation_class, validation_weights in candidate_weights.items():
        for validation_weight in validation_weights:
            validation_scores = []
            for validation_year, validation_slice in validation_blocks.items():
                masked_signal = model_signal.copy()
                heldout_positions = np.arange(len(model_signal))[validation_slice]
                heldout_positions = heldout_positions[
                    np.isfinite(model_signal[heldout_positions])
                ]
                masked_signal[heldout_positions] = np.nan
                validation_fit = sd.solve(
                    build_model(
                        masked_signal,
                        validation_class,
                        validation_weight,
                    )
                )
                validation_reconstruction = (
                    validation_fit["values"]["trend"]
                    + validation_fit["values"]["seasonal"]
                )
                predicted_price = to_price_scale(validation_reconstruction)
                validation_rmse = np.sqrt(
                    np.mean(
                        (
                            predicted_price[heldout_positions]
                            - price_values[heldout_positions]
                        )
                        ** 2
                    )
                )
                validation_scores.append(float(validation_rmse))
                validation_rows.append(
                    {
                        "trend_class": validation_class,
                        "weight": validation_weight,
                        "year": validation_year,
                        "holdout_price_rmse": float(validation_rmse),
                    }
                )

    validation_detail = pd.DataFrame(validation_rows)
    validation_summary = (
        validation_detail.groupby(["trend_class", "weight"], dropna=False)[
            "holdout_price_rmse"
        ]
        .agg(["median", "std"])
        .reset_index()
        .rename(columns={"median": "median_rmse", "std": "rmse_sd"})
    )

    selected_weights = {"linear": None}
    for selection_class in trend_weight_grids:
        selection_group = validation_summary[
            validation_summary["trend_class"] == selection_class
        ]
        minimum_score = selection_group["median_rmse"].min()
        eligible_rows = selection_group[
            selection_group["median_rmse"]
            <= minimum_score * (1.0 + tolerance_control.value)
        ]
        selected_weights[selection_class] = float(eligible_rows["weight"].max())
    return selected_weights, validation_detail, validation_summary


@app.cell
def _(
    affine_trend_label,
    plt,
    pwc_weight_control,
    pwl_weight_control,
    selected_weights,
    smooth_weight_control,
    validation_detail,
    validation_summary,
):
    validation_figure, validation_axis = plt.subplots(
        figsize=(10.5, 5), constrained_layout=True
    )
    weighted_validation = validation_summary[
        validation_summary["trend_class"] != "linear"
    ]
    current_slider_weights = {
        "smooth": 10.0 ** smooth_weight_control.value,
        "pwl": 10.0 ** pwl_weight_control.value,
        "pwc": 10.0 ** pwc_weight_control.value,
    }
    for validation_plot_class, validation_plot_group in weighted_validation.groupby(
        "trend_class", sort=False
    ):
        validation_line, = validation_axis.plot(
            validation_plot_group["weight"],
            validation_plot_group["median_rmse"],
            marker="o",
            linewidth=2.0,
            zorder=3,
            label=validation_plot_class,
        )
        family_color = validation_line.get_color()
        family_year_detail = validation_detail[
            validation_detail["trend_class"] == validation_plot_class
        ]
        for heldout_year, heldout_year_group in family_year_detail.groupby(
            "year", sort=True
        ):
            validation_axis.plot(
                heldout_year_group["weight"],
                heldout_year_group["holdout_price_rmse"],
                color=family_color,
                linewidth=0.8,
                alpha=0.25,
                zorder=1,
            )
        selected_validation_row = validation_plot_group[
            validation_plot_group["weight"]
            == selected_weights[validation_plot_class]
        ]
        validation_axis.scatter(
            selected_validation_row["weight"],
            selected_validation_row["median_rmse"],
            s=120,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )
        validation_axis.axvline(
            current_slider_weights[validation_plot_class],
            color=family_color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.4,
            zorder=1,
        )
    linear_validation_score = validation_detail[
        validation_detail["trend_class"] == "linear"
    ]["holdout_price_rmse"].median()
    validation_axis.axhline(
        linear_validation_score,
        color="0.35",
        linestyle="--",
        linewidth=1,
        label=f"{affine_trend_label} ({linear_validation_score:.3f})",
    )
    validation_axis.set_xscale("log")
    validation_axis.set_xlabel("trend regularization weight")
    validation_axis.set_ylabel("median held-out RMSE ($/gallon)")
    validation_axis.set_title(
        "Multi-era validation paths (thin lines: years; thick lines: medians)"
    )
    validation_axis.legend(frameon=False, ncol=2)
    validation_axis.spines[["top", "right"]].set_visible(False)
    plt.ylim((0.01, 0.51))
    validation_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Full-data fits and scale-aware diagnostics

    Validation-selected specifications are rebuilt on all observations.
    Structural counts use explicit effect-size thresholds:

    | Scale | PWL slope-change threshold | PWC jump threshold |
    |---|---:|---:|
    | Log price | 0.1 percentage point/week | 1% of price |
    | Price level | $0.005/week | $0.05/gallon |

    These thresholds are sensitivity specifications, not claims that every
    counted change is economically meaningful. Counts describe the penalized fit;
    they do not imply that its knot slopes or regime levels are unbiased.
    """)
    return


@app.cell
def _(build_model, model_signal, sd, selected_weights):
    selected_models = {
        fit_class: sd.solve(build_model(model_signal, fit_class, fit_weight))
        for fit_class, fit_weight in selected_weights.items()
    }
    return (selected_models,)


@app.cell
def _(
    annual_period_samples,
    harmonic_control,
    model_signal,
    np,
    observed_mask,
    pd,
    price_frame,
    sd_basis,
    seasonal_weight,
    selected_models,
    selected_weights,
    sp,
    splu,
    to_price_scale,
    use_log_scale,
    validation_summary,
):
    pwl_effect_threshold = np.log1p(0.001) if use_log_scale else 0.005
    pwc_effect_threshold = np.log1p(0.01) if use_log_scale else 0.05
    def trend_and_total_edf(trend_class, weight, fitted_model):
        T = len(model_signal)
        time = np.arange(T, dtype=float)
        fitted_trend = np.asarray(fitted_model["values"]["trend"], dtype=float)

        if trend_class == "linear":
            trend_basis = np.column_stack([np.ones(T), time])
            trend_basis, _ = np.linalg.qr(trend_basis)
            Q = sp.csr_matrix(trend_basis)
            trend_penalty = sp.csr_matrix((2, 2))
        elif trend_class == "smooth":
            Q = sp.eye(T, format="csr")
            D2 = sp.diags(
                [np.ones(T - 2), -2 * np.ones(T - 2), np.ones(T - 2)],
                [0, 1, 2],
                shape=(T - 2, T),
                format="csr",
            )
            trend_penalty = float(weight) * (D2.T @ D2)
        elif trend_class == "pwl":
            second_difference = np.diff(fitted_trend, n=2)
            active_tolerance = 1e-5 * max(1.0, float(np.ptp(fitted_trend)))
            active_knots = np.flatnonzero(
                np.abs(second_difference) > active_tolerance
            ) + 1
            hinge_basis = np.maximum(
                0.0, time[:, None] - active_knots[None, :]
            )
            trend_basis = np.column_stack([np.ones(T), time, hinge_basis])
            trend_basis, _ = np.linalg.qr(trend_basis)
            Q = sp.csr_matrix(trend_basis)
            trend_penalty = sp.csr_matrix((Q.shape[1], Q.shape[1]))
        else:
            first_difference = np.diff(fitted_trend)
            active_tolerance = 1e-5 * max(1.0, float(np.ptp(fitted_trend)))
            active_jumps = np.flatnonzero(
                np.abs(first_difference) > active_tolerance
            ) + 1
            step_basis = (time[:, None] >= active_jumps[None, :]).astype(float)
            trend_basis = np.column_stack([np.ones(T), step_basis])
            trend_basis, _ = np.linalg.qr(trend_basis)
            Q = sp.csr_matrix(trend_basis)
            trend_penalty = sp.csr_matrix((Q.shape[1], Q.shape[1]))

        seasonal_basis = sd_basis.make_basis_matrix(
            harmonic_control.value,
            T,
            [annual_period_samples],
        )[:, 1:]
        seasonal_regularizer = sd_basis.make_regularization_matrix(
            harmonic_control.value,
            seasonal_weight,
            [annual_period_samples],
        ).tocsr()[:, 1:]
        B = sp.csr_matrix(seasonal_basis)
        Z = sp.hstack([Q, B], format="csr")
        Z_observed = Z[observed_mask, :]
        Q_observed = Q[observed_mask, :]

        seasonal_penalty = seasonal_regularizer.T @ seasonal_regularizer
        penalty = sp.block_diag(
            (trend_penalty, seasonal_penalty),
            format="csc",
        )
        normal_matrix = (
            (Z_observed.T @ Z_observed) / T + penalty
        ).tocsc()
        influence_coefficients = splu(normal_matrix).solve(
            Z_observed.T.toarray() / T
        )
        trend_edf = float(
            Q_observed.multiply(
                influence_coefficients[: Q.shape[1], :].T
            ).sum()
        )
        total_edf = float(
            Z_observed.multiply(influence_coefficients.T).sum()
        )
        return trend_edf, total_edf

    summary_rows = []

    for summary_class, summary_fit in selected_models.items():
        summary_trend = np.asarray(summary_fit["values"]["trend"])
        summary_seasonal = np.asarray(summary_fit["values"]["seasonal"])
        summary_residual = np.asarray(summary_fit["values"]["residual"])
        summary_reconstruction = summary_trend + summary_seasonal
        price_reconstruction = to_price_scale(summary_reconstruction)
        price_residual = (
            price_reconstruction[observed_mask]
            - price_frame["GASREGW"].to_numpy()[observed_mask]
        )
        seasonal_baseline = float(np.median(summary_trend))
        seasonal_price = to_price_scale(seasonal_baseline + summary_seasonal)
        seasonal_amplitude = (
            100 * np.expm1(summary_seasonal.max() - summary_seasonal.min())
            if use_log_scale
            else float(np.ptp(summary_seasonal))
        )
        weekly_seasonal = (
            pd.DataFrame(
                {
                    "iso_week": price_frame.index.isocalendar().week.astype(int),
                    "effect": seasonal_price - to_price_scale(seasonal_baseline),
                }
            )
            .groupby("iso_week")["effect"]
            .mean()
        )

        selected_validation = validation_summary[
            (validation_summary["trend_class"] == summary_class)
            & (
                validation_summary["weight"].isna()
                if summary_class == "linear"
                else validation_summary["weight"].eq(selected_weights[summary_class])
            )
        ].iloc[0]
        summary_rows.append(
            {
                "trend_class": summary_class,
                "selected_weight": selected_weights[summary_class],
                "holdout_price_rmse": selected_validation["median_rmse"],
                "holdout_sd": selected_validation["rmse_sd"],
                "full_data_price_rmse": np.sqrt(np.mean(price_residual**2)),
                "model_residual_ac1": np.corrcoef(
                    summary_residual[:-1][
                        observed_mask[:-1] & observed_mask[1:]
                    ],
                    summary_residual[1:][
                        observed_mask[:-1] & observed_mask[1:]
                    ],
                )[0, 1],
                "seasonal_amplitude": seasonal_amplitude,
                "peak_iso_week": int(weekly_seasonal.idxmax()),
                "trough_iso_week": int(weekly_seasonal.idxmin()),
            }
        )

    model_summary = pd.DataFrame(summary_rows).sort_values("holdout_price_rmse")
    return (
        model_summary,
        pwc_effect_threshold,
        pwl_effect_threshold,
        trend_and_total_edf,
    )


@app.cell(hide_code=True)
def _(curvature_space_label, mo):
    mo.md(r"""
    ## Compare the trend shapes

    Validation supplies starting weights. Move the sliders to judge structural
    stability; the table below the plot follows the slider fits while always
    showing all four families. The multiselect changes only the plotted lines.
    Use **Reset weights** to return to the median-validation starting values.

    **Resid. AC(1)** is the correlation between residuals in genuinely adjacent
    observed weeks. Values near zero indicate little remaining one-week
    persistence; large positive values indicate temporal structure the model has
    not absorbed.

    Both EDF columns are calculated in **"""
        + curvature_space_label
        + r"""** from the local influence of observed data on the jointly fitted
    decomposition:

    \[
    \operatorname{EDF}_{\mathrm{trend}}
    = \operatorname{tr}\!\left(
    \frac{\partial \hat\tau_{\mathrm{obs}}}
         {\partial y_{\mathrm{obs}}}
    \right), \qquad
    \operatorname{EDF}_{\mathrm{total}}
    = \operatorname{tr}\!\left(
    \frac{\partial (\hat\tau+\hat s)_{\mathrm{obs}}}
         {\partial y_{\mathrm{obs}}}
    \right).
    \]

    Trend EDF isolates trend flexibility after accounting for competition with
    seasonality; total EDF measures the complete structural reconstruction.
    Linear and smooth fits use their exact quadratic influence matrices. PWL and
    PWC use the exact local derivative conditional on the fitted active knot or
    jump set, with tiny solver-level differences excluded by a numerical
    tolerance.

    The slider upper limits reveal the difference penalties' nullspaces: smooth
    and PWL approach affine trends, while PWC approaches a constant.
    """)
    return


@app.cell
def _(affine_trend_label, mo):
    plotted_classes_control = mo.ui.multiselect(
        options={
            affine_trend_label: "linear",
            "smooth": "smooth",
            "pwl": "pwl",
            "pwc": "pwc",
        },
        value=[affine_trend_label, "smooth", "pwl", "pwc"],
        label="Trends to compare",
    )
    reset_plot_weights_button = mo.ui.button(
        value=0,
        on_click=lambda count: count + 1,
        label="Reset weights",
        tooltip="Restore the weights selected by median holdout validation",
    )
    plotted_classes_control
    return plotted_classes_control, reset_plot_weights_button


@app.cell(hide_code=True)
def plot_weight_controls(mo, np, reset_plot_weights_button, selected_weights):
    _reset_generation = reset_plot_weights_button.value
    smooth_weight_control = mo.ui.slider(
        start=-1.0,
        stop=8.0,
        step=0.05,
        value=float(np.round(np.log10(selected_weights["smooth"]), 2)),
        label="log10 smooth weight",
        show_value=True,
    )
    pwl_weight_control = mo.ui.slider(
        start=-2.0,
        stop=6.0,
        step=0.05,
        value=float(np.round(np.log10(selected_weights["pwl"]), 2)),
        label="log10 PWL weight",
        show_value=True,
    )
    pwc_weight_control = mo.ui.slider(
        start=-2.0,
        stop=4.0,
        step=0.05,
        value=float(np.round(np.log10(selected_weights["pwc"]), 2)),
        label="log10 PWC weight",
        show_value=True,
    )
    mo.hstack(
        [
            smooth_weight_control,
            pwl_weight_control,
            pwc_weight_control,
            reset_plot_weights_button,
        ],
        justify="start",
        gap="1.5rem",
    )
    return pwc_weight_control, pwl_weight_control, smooth_weight_control


@app.cell
def _(
    build_model,
    model_signal,
    pwc_weight_control,
    pwl_weight_control,
    sd,
    selected_models,
    smooth_weight_control,
):
    interactive_models = {
        "linear": selected_models["linear"],
        "smooth": sd.solve(
            build_model(model_signal, "smooth", 10.0 ** smooth_weight_control.value)
        ),
        "pwl": sd.solve(
            build_model(model_signal, "pwl", 10.0 ** pwl_weight_control.value)
        ),
        "pwc": sd.solve(
            build_model(model_signal, "pwc", 10.0 ** pwc_weight_control.value)
        ),
    }
    return (interactive_models,)


@app.cell
def _(
    affine_trend_label,
    interactive_models,
    np,
    plotted_classes_control,
    plt,
    price_frame,
    price_series,
    residual_label,
    to_price_scale,
    use_log_scale,
):
    trend_figure, trend_axes = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, constrained_layout=True
    )
    trend_axes[0].plot(
        price_frame.index,
        price_series,
        color="0.7",
        linewidth=0.7,
        label="observed",
    )
    for plotted_class, plotted_fit in interactive_models.items():
        plotted_label = (
            affine_trend_label if plotted_class == "linear" else plotted_class
        )
        if plotted_class in plotted_classes_control.value:
            plotted_trend = to_price_scale(plotted_fit["values"]["trend"])
            plotted_residual = np.asarray(plotted_fit["values"]["residual"])
            if use_log_scale:
                plotted_residual = 100 * np.expm1(plotted_residual)
            trend_axes[0].plot(
                price_frame.index,
                plotted_trend,
                linewidth=1.6,
                label=plotted_label,
            )
            trend_axes[1].plot(
                price_frame.index,
                plotted_residual,
                linewidth=0.7,
                alpha=0.8,
                label=plotted_label,
            )
    trend_axes[0].set_ylabel("trend ($/gallon)")
    trend_axes[0].set_title("Full-data trend estimates")
    trend_axes[0].legend(frameon=False, ncol=5)
    trend_axes[1].axhline(0, color="black", linewidth=0.6)
    trend_axes[1].set_ylabel(residual_label)
    trend_axes[1].set_title("Residuals expose the consequences of each trend claim")
    for trend_axis in trend_axes:
        trend_axis.spines[["top", "right"]].set_visible(False)
    trend_figure
    return


@app.cell
def _(
    affine_trend_label,
    curvature_space_label,
    interactive_models,
    np,
    observed_mask,
    pd,
    price_frame,
    pwc_weight_control,
    pwl_weight_control,
    smooth_weight_control,
    to_price_scale,
    trend_and_total_edf,
    use_log_scale,
):
    seasonal_unit = "%" if use_log_scale else "$/gallon"

    def build_interactive_summary():
        slider_weights = {
            "linear": None,
            "smooth": 10.0 ** smooth_weight_control.value,
            "pwl": 10.0 ** pwl_weight_control.value,
            "pwc": 10.0 ** pwc_weight_control.value,
        }
        rows = []
        observed_prices = price_frame["GASREGW"].to_numpy()[observed_mask]

        for trend_class, fitted_model in interactive_models.items():
            trend = np.asarray(fitted_model["values"]["trend"])
            seasonal = np.asarray(fitted_model["values"]["seasonal"])
            residual = np.asarray(fitted_model["values"]["residual"])
            reconstructed_price = to_price_scale(trend + seasonal)
            price_error = reconstructed_price[observed_mask] - observed_prices

            baseline = float(np.median(trend))
            seasonal_price = to_price_scale(baseline + seasonal)
            annual_amplitude = (
                100 * np.expm1(seasonal.max() - seasonal.min())
                if use_log_scale
                else float(np.ptp(seasonal))
            )
            weekly_effect = (
                pd.DataFrame(
                    {
                        "iso_week": price_frame.index.isocalendar().week.astype(int),
                        "effect": seasonal_price - to_price_scale(baseline),
                    }
                )
                .groupby("iso_week")["effect"]
                .mean()
            )

            trend_edf, total_edf = trend_and_total_edf(
                trend_class,
                slider_weights[trend_class],
                fitted_model,
            )

            rows.append(
                {
                    "Trend": (
                        affine_trend_label
                        if trend_class == "linear"
                        else trend_class
                    ),
                    "Weight": slider_weights[trend_class],
                    "RMSE ($/gal)": np.sqrt(np.mean(price_error**2)),
                    "Resid. AC(1)": np.corrcoef(
                        residual[:-1][
                            observed_mask[:-1] & observed_mask[1:]
                        ],
                        residual[1:][
                            observed_mask[:-1] & observed_mask[1:]
                        ],
                    )[0, 1],
                    f"Annual amp. ({seasonal_unit})": annual_amplitude,
                    "Peak wk": int(weekly_effect.idxmax()),
                    "Trough wk": int(weekly_effect.idxmin()),
                    "Trend EDF": trend_edf,
                    "Total EDF": total_edf,
                }
            )
        return pd.DataFrame(rows)

    display_summary = build_interactive_summary()
    _grouped_summary = display_summary[
        [
            "Trend",
            "Weight",
            "RMSE ($/gal)",
            "Resid. AC(1)",
            "Trend EDF",
            "Total EDF",
            f"Annual amp. ({seasonal_unit})",
            "Peak wk",
            "Trough wk",
        ]
    ].copy()
    _grouped_summary.columns = pd.MultiIndex.from_tuples(
        [
            ("", "Trend"),
            ("Fit & complexity", "Weight"),
            ("Fit & complexity", "RMSE ($/gal)"),
            ("Fit & complexity", "Resid. AC(1)"),
            ("Fit & complexity", "Trend EDF"),
            ("Fit & complexity", "Total EDF"),
            ("Seasonality", f"Annual amp. ({seasonal_unit})"),
            ("Seasonality", "Peak wk"),
            ("Seasonality", "Trough wk"),
        ]
    )
    _weight_column = ("Fit & complexity", "Weight")
    _rmse_column = ("Fit & complexity", "RMSE ($/gal)")
    _ac1_column = ("Fit & complexity", "Resid. AC(1)")
    _trend_edf_column = ("Fit & complexity", "Trend EDF")
    _total_edf_column = ("Fit & complexity", "Total EDF")
    _amplitude_column = ("Seasonality", f"Annual amp. ({seasonal_unit})")
    _trend_column = ("", "Trend")

    _grouped_summary.style.hide(axis="index").format(
        {
            _weight_column: (
                lambda value: "—" if pd.isna(value) else f"{value:.3g}"
            ),
            _rmse_column: "{:.4f}",
            _ac1_column: "{:.3f}",
            _trend_edf_column: "{:.1f}",
            _total_edf_column: "{:.1f}",
            _amplitude_column: "{:.2f}",
        },
        na_rep="—",
    ).set_caption(
        f"Slider-driven full-data summary · EDF measured in {curvature_space_label}"
    ).set_table_styles(
        [
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("text-align", "left"),
                    ("font-style", "italic"),
                    ("padding-bottom", "0.75rem"),
                ],
            },
            {
                "selector": "th.col_heading.level0",
                "props": [
                    ("font-weight", "700"),
                    ("text-align", "center"),
                    ("border-bottom", "1px solid var(--slate-4)"),
                    ("padding", "0.35rem 0.65rem"),
                ],
            },
            {
                "selector": "th.col_heading.level1",
                "props": [
                    ("white-space", "normal"),
                    ("line-height", "1.15"),
                    ("min-width", "5rem"),
                    ("text-align", "center"),
                    ("vertical-align", "bottom"),
                    ("padding", "0.45rem 0.7rem"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("white-space", "nowrap"),
                    ("padding", "0.45rem 0.7rem"),
                    ("text-align", "right"),
                ],
            },
        ]
    ).set_properties(
        subset=[_trend_column],
        **{"text-align": "left", "min-width": "7rem"},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## L1 structural paths

    PWL and PWC weights change both reconstruction and the number of detected
    structural changes. Full-data paths therefore complement holdout paths. A
    final event-oriented specification should impose a plausible maximum change
    count and a meaningful minimum effect size, admit a null result, and choose
    the least-regularized admissible model.

    PWL is particularly useful for gasoline prices because extended rises and
    falls can be summarized as approximately linear phases. Its sparse knots
    localize changes in slope, and knots where slope changes from positive to
    negative—or negative to positive—provide candidate peaks and troughs. This is
    more date-specific than finding a maximum on a broadly curving smooth trend.
    Not every knot is a turning point: peak/trough interpretation should also
    require a genuine slope-sign reversal, meaningful price prominence, and
    stability across nearby weights.

    Plain L1 is a support-discovery tool, not an unshrunk level estimator. It
    penalizes jump or slope-change magnitude, so accepted PWC regimes are pulled
    toward one another and accepted PWL slope changes can be attenuated. After a
    structurally credible weight is chosen:

    1. use **IRL1** when many weak changes remain or important changes appear
       overly shrunk; it reduces penalties on established large changes and
       increases them on small ones;
    2. treat IRL1 as a local-search heuristic—a sequence of convex fits—not a
       guarantee of the globally correct change points; and
    3. for final amplitudes, freeze the accepted knot or jump locations and refit
       the segment parameters jointly with seasonality **without the L1
       magnitude penalty**.

    That fixed-support refit gives unshrunk least-squares levels or slopes
    conditional on the chosen structure.
    """)
    return


@app.cell
def _(
    build_model,
    model_signal,
    np,
    observed_mask,
    pd,
    price_values,
    pwc_effect_threshold,
    pwl_effect_threshold,
    sd,
    to_price_scale,
    trend_weight_grids,
):
    structural_rows = []
    for structural_class, structural_threshold, difference_order in (
        ("pwl", pwl_effect_threshold, 2),
        ("pwc", pwc_effect_threshold, 1),
    ):
        for structural_weight in trend_weight_grids[structural_class]:
            structural_fit = sd.solve(
                build_model(model_signal, structural_class, structural_weight)
            )
            structural_trend = np.asarray(structural_fit["values"]["trend"])
            structural_reconstruction = (
                structural_trend + structural_fit["values"]["seasonal"]
            )
            structural_price = to_price_scale(structural_reconstruction)
            structural_rows.append(
                {
                    "trend_class": structural_class,
                    "weight": structural_weight,
                    "qualified_changes": int(
                        (
                            np.abs(np.diff(structural_trend, n=difference_order))
                            >= structural_threshold
                        ).sum()
                    ),
                    "full_data_price_rmse": np.sqrt(
                        np.mean(
                            (
                                structural_price[observed_mask]
                                - price_values[observed_mask]
                            )
                            ** 2
                        )
                    ),
                }
            )
    structural_paths = pd.DataFrame(structural_rows)
    return (structural_paths,)


@app.cell
def _(plt, selected_weights, structural_paths):
    structural_figure, structural_axes = plt.subplots(
        1, 2, figsize=(11.5, 4.3), constrained_layout=True
    )
    for path_class, path_group in structural_paths.groupby(
        "trend_class", sort=False
    ):
        structural_axes[0].plot(
            path_group["weight"],
            path_group["qualified_changes"],
            marker="o",
            label=path_class,
        )
        structural_axes[1].plot(
            path_group["weight"],
            path_group["full_data_price_rmse"],
            marker="o",
            label=path_class,
        )
        for path_axis in structural_axes:
            path_axis.axvline(
                selected_weights[path_class],
                color=path_axis.lines[-1].get_color(),
                linestyle=":",
                linewidth=1,
            )
    structural_axes[0].set_ylabel("effect-size-qualified changes")
    structural_axes[0].set_title("Structural complexity path")
    structural_axes[1].set_ylabel("full-data RMSE ($/gallon)")
    structural_axes[1].set_title("Training fit path")
    for styled_axis in structural_axes:
        styled_axis.set_xscale("log")
        styled_axis.set_xlabel("trend regularization weight")
        styled_axis.legend(frameon=False)
        styled_axis.spines[["top", "right"]].set_visible(False)
    structural_figure
    return


@app.cell(hide_code=True)
def _(affine_trend_label, mo, model_summary, scale_label):
    best_holdout_row = model_summary.iloc[0]
    smooth_summary_row = model_summary[
        model_summary["trend_class"] == "smooth"
    ].iloc[0]
    pwl_summary_row = model_summary[
        model_summary["trend_class"] == "pwl"
    ].iloc[0]
    mo.md(
        f"""
        ## Reading the comparison

        On **{scale_label}**, **{best_holdout_row['trend_class'].upper()}** has
        the lowest median held-out dollar RMSE. That ranks reconstruction under
        the current controls; it does not prove that gasoline prices follow that
        structural form. A low median can coexist with one poor holdout year.

        - **{affine_trend_label.capitalize()}** is a useful stiff baseline, but strong residual
          autocorrelation indicates that one model-space slope misses major price eras.
        - **Smooth** is the least structurally committal flexible baseline. Its
          selected fit provides the least structurally committal flexible baseline.
        - **PWL** uses sparse slope changes rather than distributed quadratic
          curvature. For this series, slope-sign reversals offer naturally
          interpretable candidate peaks and troughs. Its knots are candidates for
          review, not automatically identified market events.
        - **PWC** is easy to over-interpret. Require economically meaningful
          jumps, a plausible regime count, and stability across holdout years
          before calling its steps regimes. Its penalized regime levels are
          shrunk, not final unpenalized estimates.
        - Annual amplitude and timing vary across trend classes, demonstrating
          direct competition between trend and seasonality.

        ### Unresolved choices

        1. Compare log and level fits visually; the transform encodes whether
           proportional or absolute error matters.
        2. The data are nominal. CPI deflation could materially change the
           decades-long trend even after using logs.
        3. The 2008 and 2020 movements may justify a separate sparse shock
           component rather than forcing the trend to absorb them.
        4. Validation varies across market eras. Treat small median-score
           differences as flat, not as evidence for a precise universal weight.
        5. If PWL or PWC is selected for change-point interpretation, specify
           admissible effects and counts first, consider IRL1 for support
           stabilization, then perform a fixed-support unpenalized refit for final
           slopes or regime levels.
        6. The early missing block is structurally imputed, but uncertainty
           intervals belong only after the final operating model is selected.
        """
    )
    return


if __name__ == "__main__":
    app.run()
