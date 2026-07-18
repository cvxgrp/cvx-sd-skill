"""Wire decomposition outputs back to pandas (and, later, reports/plots).

The decomposition core works on bare numpy arrays (it does not carry a time
index, by design). This module closes the loop: re-wrap solved components onto
a pandas index so the user gets labeled, plottable time series back -- the
"translation-OUT" step from raw frame in to labeled components out.

Provides the pandas round-trip (:func:`components_to_frame`), stacked-panel and
stability plots (:func:`plot_decomposition`, :func:`plot_stability`), and a
domain-agnostic markdown summary (:func:`format_report`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def components_to_frame(out, index=None, y=None, mask=None):
    """Re-wrap solved decomposition components as a pandas DataFrame.

    One column per *full-length* component -- the residual and each structural
    role -- aligned on ``index``. Scalar or non-length-T aux quantities (trend
    slopes, exogenous coefficients, Fourier coefficient vectors) are excluded:
    they are not time series. A ``reconstruction`` column (the sum of the
    structural components, i.e. ``y - residual`` on observed entries) is always
    included; a ``y`` column is included when the observed signal is supplied.

    Because the decomposition defines every component on every grid point, the
    returned columns are dense -- including where the input had gaps, which the
    model has imputed. Pass ``mask`` to restore NaN at originally-unobserved
    entries instead of the imputed values.

    Parameters
    ----------
    out : dict
        A solved output from :func:`signaldecomp.solve` (needs ``"values"`` with
        a ``"residual"`` entry).
    index : pandas.Index, optional
        Index to align on (typically the ``"index"`` from
        :func:`signaldecomp.standardize_time_axis`). Defaults to a RangeIndex.
    y : numpy.ndarray, optional
        The observed signal; if given, added as a ``y`` column (NaNs preserved).
    mask : numpy.ndarray of bool, optional
        Observed-entry mask (True = observed). If given, component and
        reconstruction values at unobserved entries are set to NaN (holes
        preserved rather than imputed). Does not affect the ``y`` column.

    Returns
    -------
    pandas.DataFrame
        Columns: each full-length role, ``residual``, ``reconstruction``, and
        (if ``y`` given) ``y``. Column order: structural roles in solve order,
        then ``residual``, ``reconstruction``, ``y``.

    Raises
    ------
    ValueError
        If lengths are inconsistent, or ``out`` lacks a residual.
    """
    values = out["values"]
    if "residual" not in values:
        raise ValueError("out['values'] has no 'residual'; not a solved output.")
    residual = np.asarray(values["residual"], dtype=float)
    T = residual.shape[0]

    # Full-length structural components, in solve order (dict preserves order),
    # excluding the residual and any non-length-T aux.
    structural = {}
    for role, val in values.items():
        if role == "residual":
            continue
        arr = np.asarray(val, dtype=float) if np.ndim(val) else None
        if arr is not None and arr.shape == (T,):
            structural[role] = arr

    reconstruction = (
        np.sum(np.stack(list(structural.values()), axis=0), axis=0)
        if structural
        else np.zeros(T)
    )

    if index is None:
        index = pd.RangeIndex(T)
    elif len(index) != T:
        raise ValueError(f"index length {len(index)} != component length {T}.")

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (T,):
            raise ValueError(f"mask shape {mask.shape} != ({T},).")

    cols = {}
    for role, arr in structural.items():
        cols[role] = np.where(mask, arr, np.nan) if mask is not None else arr
    cols["residual"] = (
        np.where(mask, residual, np.nan) if mask is not None else residual
    )
    cols["reconstruction"] = (
        np.where(mask, reconstruction, np.nan) if mask is not None else reconstruction
    )
    if y is not None:
        y = np.asarray(y, dtype=float)
        if y.shape != (T,):
            raise ValueError(f"y shape {y.shape} != ({T},).")
        cols["y"] = y

    return pd.DataFrame(cols, index=index)


# Colorblind-friendly cycle for component panels (Tableau colorblind10 order),
# applied via a local rc_context so we never mutate global matplotlib state.
_CB_CYCLE = [
    "#006BA4", "#FF800E", "#ABABAB", "#595959", "#5F9ED1",
    "#C85200", "#898989", "#A2C8EC", "#FFBC79", "#CFCFCF",
]


def plot_decomposition(out, y=None, index=None, df=None, figsize=None):
    """Stacked-panel plot of a decomposition: signal+fit, each role, residual.

    Panels, top to bottom: the observed signal with the reconstruction overlaid;
    one panel per structural role; then the residual (drawn as a signed fill
    above/below zero). Generalizes the fixed seasonal/trend/residual layout of
    the source to the role-based, append-only model (any number of components).

    Parameters
    ----------
    out : dict
        A solved output from :func:`signaldecomp.solve`. Ignored if ``df`` is
        given.
    y : numpy.ndarray, optional
        Observed signal, overlaid on the top panel if given.
    index : pandas.Index, optional
        x-axis index (e.g. the time_axis ``"index"``); default RangeIndex.
    df : pandas.DataFrame, optional
        A prebuilt :func:`components_to_frame` result to plot directly, instead
        of ``out`` (alternate one-call path).
    figsize : tuple, optional
        Figure size; defaults to a height that scales with the panel count.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if df is None:
        df = components_to_frame(out, index=index, y=y)
    reserved = {"residual", "reconstruction", "y"}
    roles = [c for c in df.columns if c not in reserved]
    x = df.index

    n_panels = 2 + len(roles)  # signal+fit, each role, residual
    if figsize is None:
        figsize = (12, 2.2 * n_panels)

    with plt.rc_context({"axes.prop_cycle": plt.cycler(color=_CB_CYCLE)}):
        fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
        if n_panels == 1:
            axes = [axes]

        # Top: observed signal + reconstruction.
        ax = axes[0]
        if "y" in df.columns:
            ax.plot(x, df["y"], lw=0.8, label="observed", zorder=1)
        ax.plot(x, df["reconstruction"], lw=1.5, label="reconstruction", zorder=2)
        ax.set_ylabel("signal")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
        ax.set_title("Decomposition", fontsize=11, fontweight="bold")

        # One panel per structural role.
        for ax, role in zip(axes[1 : 1 + len(roles)], roles):
            ax.plot(x, df[role], lw=1.4)
            ax.set_ylabel(role)

        # Residual: signed fill above/below zero.
        ax = axes[-1]
        resid = df["residual"].to_numpy()
        ax.fill_between(x, resid, 0.0, where=(resid >= 0), alpha=0.5, lw=0)
        ax.fill_between(x, resid, 0.0, where=(resid < 0), alpha=0.5, lw=0)
        ax.axhline(0.0, color="black", lw=0.7)
        ax.set_ylabel("residual")

        for ax in axes:
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        fig.align_ylabels(axes)
        fig.tight_layout()
    return fig


def plot_stability(stability, role=None, figsize=None):
    """Multi-panel stability summary for one component role over expanding windows.

    Panels: (1) the role's component curve at each window length, colored by
    window length ("spaghetti"); (2) normalized RMS change between successive
    windows. If the stability run used a scalar ``extractor``, two more panels
    are added: (3) each scalar quantity's value vs window length with dashed
    convergence markers, and (4) its absolute between-window change.

    Parameters
    ----------
    stability : dict
        Output of :func:`signaldecomp.expanding_window_stability`.
    role : str, optional
        Which component role's snapshots to show. Defaults to the first role.
    figsize : tuple, optional
        Figure size; defaults to a height scaled to the panel count.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    windows = stability["windows"]
    F = windows.size
    snap_roles = list(stability["snapshots"].keys())
    if not snap_roles:
        raise ValueError("stability has no component snapshots to plot.")
    if role is None:
        role = snap_roles[0]
    elif role not in snap_roles:
        raise ValueError(f"role {role!r} not in snapshots {snap_roles}.")

    has_scalar = "history" in stability
    mid = (windows[:-1] + windows[1:]) / 2
    n_panels = 2 + (2 if has_scalar else 0)
    if figsize is None:
        figsize = (12, 2.6 * n_panels)

    with plt.rc_context({"axes.prop_cycle": plt.cycler(color=_CB_CYCLE)}):
        fig, axes = plt.subplots(n_panels, 1, figsize=figsize)
        colors = cm.plasma(np.linspace(0.15, 0.9, F))

        # Panel 1: snapshot spaghetti, colored by window length.
        ax = axes[0]
        snap = stability["snapshots"][role]
        for i in range(F):
            row = snap[i]
            valid = ~np.isnan(row)
            if valid.any():
                ax.plot(np.where(valid)[0], row[valid], color=colors[i],
                        lw=0.7, alpha=0.7)
        ax.set_ylabel(role)
        ax.set_title(f"{role}: fits by window length", fontsize=9)
        sm = plt.cm.ScalarMappable(
            cmap="plasma", norm=plt.Normalize(windows[0], windows[-1])
        )
        fig.colorbar(sm, ax=ax, label="window length (samples)", pad=0.01)

        # Panel 2: normalized RMSD between successive windows.
        ax = axes[1]
        ax.plot(mid, stability["rmsd"][role], lw=1.2)
        ax.fill_between(mid, stability["rmsd"][role], alpha=0.15)
        ax.set_ylabel("norm. RMSD")
        ax.set_xlabel("window length (samples)")
        ax.set_title(f"{role}: normalized change between successive windows",
                     fontsize=9)

        # Panels 3-4: scalar history + delta, if an extractor was used.
        if has_scalar:
            ax = axes[2]
            for k, arr in stability["history"].items():
                (line,) = ax.plot(windows, arr, lw=1.2, label=k)
                c_at = stability["converged_at"].get(k)
                if c_at is not None:
                    ax.axvline(c_at, color=line.get_color(), lw=0.8, ls="--",
                               alpha=0.6)
            ax.set_ylabel("value")
            ax.set_xlabel("window length (samples)")
            ax.set_title(
                f"scalar history (dashed = converged within "
                f"±{stability['tol']:g} of final)", fontsize=9)
            ax.legend(fontsize=8, framealpha=0.7)

            ax = axes[3]
            for k, arr in stability["delta"].items():
                ax.plot(mid, arr, lw=1.0, label=k, alpha=0.8)
            ax.set_ylabel("|Δ value|")
            ax.set_xlabel("window length (samples)")
            ax.set_title("absolute change between successive windows", fontsize=9)
            ax.legend(fontsize=8, framealpha=0.7)

        for ax in axes:
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
    return fig


def format_report(out, y=None, title="Signal decomposition"):
    """Domain-agnostic markdown summary of a solved decomposition.

    Reports solve status, each structural component's share of the
    reconstruction energy (a proxy for how much it contributes to the fit --
    useful when deciding which components merit holdout-tuning), residual
    statistics, and any scalar aux quantities. Deliberately free of domain
    terms (no rates, units, physical interpretation): those belong to the
    caller's domain layer.

    Parameters
    ----------
    out : dict
        A solved output from :func:`signaldecomp.solve`.
    y : numpy.ndarray, optional
        The observed signal; enables fit RMS/MAE on observed entries and mask
        coverage.
    title : str
        Heading for the report.

    Returns
    -------
    str
        A markdown-formatted report.
    """
    values = out["values"]
    if "residual" not in values:
        raise ValueError("out['values'] has no 'residual'; not a solved output.")
    residual = np.asarray(values["residual"], dtype=float)
    T = residual.shape[0]

    structural = {}
    scalar_aux = {}
    for role, val in values.items():
        if role == "residual":
            continue
        arr = np.asarray(val, dtype=float) if np.ndim(val) else None
        if arr is not None and arr.shape == (T,):
            structural[role] = arr
        elif np.ndim(val) == 0:
            scalar_aux[role] = float(val)

    reconstruction = (
        np.sum(np.stack(list(structural.values()), axis=0), axis=0)
        if structural
        else np.zeros(T)
    )

    lines = [f"# {title}", ""]
    lines.append(f"- **status:** {out.get('status', 'unknown')}")
    lines.append(f"- **length (T):** {T}")

    energies = {r: float(np.sum(a**2)) for r, a in structural.items()}
    total_energy = sum(energies.values())
    lines.append("")
    lines.append("## Components (share of reconstruction energy)")
    lines.append("")
    if total_energy > 0:
        for role in structural:
            share = 100.0 * energies[role] / total_energy
            lines.append(f"- **{role}:** {share:5.1f}%")
    else:
        lines.append("- (no structural components)")

    lines.append("")
    lines.append("## Residual")
    lines.append("")
    lines.append(f"- **residual RMS:** {np.sqrt(np.mean(residual**2)):.4g}")
    lines.append(f"- **residual MAE:** {np.mean(np.abs(residual)):.4g}")
    if y is not None:
        y = np.asarray(y, dtype=float)
        obs = ~np.isnan(y)
        n_obs = int(obs.sum())
        err = reconstruction[obs] - y[obs]
        lines.append(
            f"- **observed entries:** {n_obs} / {T} "
            f"({100.0 * n_obs / T:.1f}% coverage)"
        )
        lines.append(f"- **fit RMS (observed):** {np.sqrt(np.mean(err**2)):.4g}")
        lines.append(f"- **fit MAE (observed):** {np.mean(np.abs(err)):.4g}")

    if scalar_aux:
        lines.append("")
        lines.append("## Scalar quantities")
        lines.append("")
        for k in sorted(scalar_aux):
            lines.append(f"- **{k}:** {scalar_aux[k]:.6g}")

    return "\n".join(lines)
