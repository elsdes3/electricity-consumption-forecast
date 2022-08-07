#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.ticker import FuncFormatter
from scipy.stats import skew
from sklearn.metrics import r2_score


def customize_splines(ax: plt.axis) -> plt.axis:
    ax.spines["left"].set_edgecolor("black")
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_edgecolor("black")
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["top"].set_edgecolor("lightgrey")
    ax.spines["top"].set_linewidth(1)
    ax.spines["right"].set_edgecolor("lightgrey")
    ax.spines["right"].set_linewidth(1)
    return ax


def generate_plots(m, residual, scores_dict, forecast, primary_metric):
    # Plot residual timeseries
    _, ax = plt.subplots(figsize=(12, 4))
    residual.plot(ax=ax)
    ax.set_title(
        f"{primary_metric.upper()} = {scores_dict[primary_metric]:.2f} "
        f"(SMAPE = {scores_dict['smape(%)']:.2f})",
        loc="left",
        fontweight="bold",
    )
    ax.grid(which="both", axis="both", color="lightgrey")
    _ = customize_splines(ax)

    # Plot learned trend and seasonalities
    _ = m.plot_components(forecast)


def plot_histogram(
    ts,
    ptitle,
    tick_label_fontsize=12,
    ptitle_fontsize=12,
    dpi=75,
    x_thou_comma_sep=True,
    y_thou_comma_sep=True,
    fig_size=(12, 4),
):
    _, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ts.plot(kind="hist", ax=ax, color="blue", zorder=3, edgecolor="white")
    ax.grid(color="lightgrey", zorder=0)
    ax.set_ylabel(None)
    ax.set_title(
        ptitle, loc="left", fontweight="bold", fontsize=ptitle_fontsize
    )
    if x_thou_comma_sep:
        ax.get_xaxis().set_major_formatter(
            FuncFormatter(lambda x, p: format(int(x), ","))
        )
    if y_thou_comma_sep:
        ax.get_yaxis().set_major_formatter(
            FuncFormatter(lambda x, p: format(int(x), ","))
        )
    ax.xaxis.set_tick_params(labelsize=tick_label_fontsize)
    ax.yaxis.set_tick_params(labelsize=tick_label_fontsize)
    ax = customize_splines(ax)
    ax.axvline(x=ts.mean(), color="red", zorder=3, ls="--")
    for _, spine in ax.spines.items():
        spine.set_zorder(10)


def plot_multi_ts(
    tss_dict,
    ptitle,
    zero_line_color="",
    legend_loc=(0.575, 1.1),
    tick_label_fontsize=12,
    ptitle_fontsize=12,
    legend_fontsize=12,
    dpi=75,
    xtick_halign="center",
    xtick_angle=0,
    fig_size=(12, 4),
):
    _, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    for k, v in tss_dict.items():
        v[0].plot(ax=ax, color=v[1], ls=v[2], label=k, lw=v[3])
    if len(tss_dict) > 1:
        ax.legend(
            loc="upper left",
            frameon=False,
            bbox_to_anchor=legend_loc,
            # columnspacing=0.2,
            ncol=1,
            handletextpad=0.1,
            prop={"size": legend_fontsize},
        )
    ax.set_title(
        ptitle, loc="left", fontweight="bold", fontsize=ptitle_fontsize
    )
    ax.set_xlabel(None)
    ax.get_yaxis().set_major_formatter(
        FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax.xaxis.set_tick_params(labelsize=tick_label_fontsize)
    ax.yaxis.set_tick_params(labelsize=tick_label_fontsize)
    ax.grid(which="both", axis="both", color="lightgrey")
    if zero_line_color != "":
        ax.axhline(0, 0, len(v[0]), color=zero_line_color, ls="--")
    ax.minorticks_on()
    for label in ax.get_xticklabels():
        label.set_ha(xtick_halign)
        label.set_rotation(xtick_angle)
    _ = customize_splines(ax)


def plot_ts_acf(
    ts,
    ptitle,
    nlags,
    markersize=0.5,
    tick_label_fontsize=12,
    ptitle_fontsize=12,
    dpi=75,
    fig_size=(12, 4),
):
    _, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    sm.graphics.tsa.plot_acf(
        ts,
        lags=nlags,
        ax=ax,
        use_vlines=False,
        title=None,
        markersize=markersize,
    )
    ax.set_title(
        ptitle, loc="left", fontweight="bold", fontsize=ptitle_fontsize
    )
    ax.set_xlabel(None)
    ax.grid(color="lightgrey")
    ax.xaxis.set_tick_params(labelsize=tick_label_fontsize)
    ax.yaxis.set_tick_params(labelsize=tick_label_fontsize)

    ax.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x, p: format(int(x), ","))
    )

    for item in ax.collections:
        # change the color of the CI
        if type(item) == PolyCollection:
            item.set_facecolor("blue")
        # change the color of the vertical lines
        if type(item) == LineCollection:
            item.set_color("blue")

    # change the color of the markers/horizontal line
    for item in ax.lines:
        item.set_color("blue")

    _ = customize_splines(ax)


def plot_pairwise_scatterplot_grid(
    data,
    pair_col,
    xvar,
    yvar,
    pairs,
    tick_label_fontsize=14,
    legend_fontsize=14,
    fig_size=(16, 30),
):
    fig = plt.figure(figsize=fig_size)
    grid = plt.GridSpec(len(pairs), 2, hspace=0.075, wspace=0.1)
    for q, c_pairs in enumerate(pairs):
        for k, c in enumerate(c_pairs):
            ax = fig.add_subplot(grid[q, k])
            data[data[pair_col] == c].plot.scatter(
                x=xvar,
                y=yvar,
                c="white",
                edgecolor="blue",
                ax=ax,
                s=60,
                label=c,
            )
            ax.grid(color="lightgrey")
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.xaxis.set_tick_params(labelsize=tick_label_fontsize)
            ax.yaxis.set_tick_params(labelsize=tick_label_fontsize)
            leg = ax.legend(
                loc="upper right",
                handlelength=0,
                handletextpad=0,
                frameon=False,
                prop={"size": legend_fontsize, "weight": "bold"},
            )
            for item in leg.legendHandles:
                item.set_visible(False)
            _ = customize_splines(ax)


def plot_y_vs_x(
    data,
    xvar,
    yvar,
    xvar_axis_label,
    yvar_axis_label,
    ptitle,
    ax,
    axis_label_fontsize=12,
    tick_label_fontsize=12,
    ptitle_fontsize=12,
    plot_hline=False,
    diag_line_coords=[],
):
    data.plot(
        ax=ax,
        x=xvar,
        y=yvar,
        kind="scatter",
        edgecolor="blue",
        zorder=3,
        s=40,
        c="none",
    )
    ax.set_xlabel(xvar_axis_label, fontsize=axis_label_fontsize)
    ax.set_ylabel(yvar_axis_label, fontsize=axis_label_fontsize)
    ax.xaxis.set_tick_params(labelsize=tick_label_fontsize)
    ax.yaxis.set_tick_params(labelsize=tick_label_fontsize)
    ax.grid(color="lightgrey", zorder=0)
    if plot_hline:
        ax.axhline(y=0, lw=2, c="red", ls="--", zorder=3)
    if diag_line_coords:
        data = data.dropna()
        m, b = np.polyfit(data[xvar].to_numpy(), data[yvar].to_numpy(), 1)
        ax.plot(
            data[xvar].to_numpy(),
            m * data[xvar].to_numpy() + b,
            color="black",
            lw=1.5,
            zorder=3,
        )
        r2 = r2_score(data[xvar].to_numpy(), data[yvar].to_numpy())
        lows, highs = diag_line_coords[0], diag_line_coords[1]
        ax.axline(lows, highs, c="red", zorder=3, ls="--")
        ptitle += r" ($\mathregular{R^2}$" + f"={r2:.2f})"
    ax.set_title(
        ptitle, fontsize=ptitle_fontsize, fontweight="bold", loc="left"
    )
    _ = customize_splines(ax)


def plot_diagnostic_grid(
    future_forecast,
    title_scores,
    shade_alpha=0.5,
    hist_annot_xy=[0.8, 0.85],
    axis_label_fontsize=12,
    tick_label_fontsize=12,
    ptitle_fontsize=12,
    hspace=0.25,
    wspace=0.1,
    fig_size=(12, 20),
):
    fig = plt.figure(figsize=fig_size)
    grid = plt.GridSpec(4, 2, hspace=hspace, wspace=wspace)
    ax1 = fig.add_subplot(grid[3, :])
    ax2 = fig.add_subplot(grid[2, :])
    ax3 = fig.add_subplot(grid[0, :])
    ax4 = fig.add_subplot(grid[1, 0])
    ax5 = fig.add_subplot(grid[1, 1])

    ts = (
        future_forecast.set_index("ds")["y"]
        - future_forecast.set_index("ds")["yhat"]
    )
    summ_stats = (
        f"Skewness = {skew(ts.dropna()):+.2f}\nMedian = {ts.median():+.2f}"
    )
    df_pred = future_forecast.set_index("ds")[
        ["y", "yhat", "yhat_lower", "yhat_upper"]
    ]

    hist_annot_x, hist_annot_y = hist_annot_xy
    country = future_forecast["country"][0]

    ts.plot.hist(ax=ax1, color="blue", lw=1, edgecolor="white")
    ax1.set_axisbelow(True)
    ax1.set_ylabel(None)
    ptitle = "Residual frequency histogram"
    ax1.set_title(ptitle, fontweight="bold", loc="left")
    ax1.text(
        hist_annot_x,
        hist_annot_y,
        summ_stats,
        fontsize=14,
        transform=ax1.transAxes,
    )
    ax1.axvline(x=ts.median(), ls="--", color="red")
    ax1.grid(which="both", axis="both", color="lightgrey")
    _ = customize_splines(ax1)

    lows_highs = [
        [df_pred[["y", "yhat"]].min().min()] * 2,
        [df_pred[["y", "yhat"]].max().max()] * 2,
    ]
    plot_y_vs_x(
        data=df_pred[["y", "yhat"]],
        xvar_axis_label="Observed",
        yvar_axis_label="Predicted",
        ptitle="Predicted vs Observed Values",
        ax=ax5,
        xvar="y",
        yvar="yhat",
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        ptitle_fontsize=ptitle_fontsize,
        diag_line_coords=lows_highs,
    )

    df_pred["y"].plot(ax=ax3, color="blue", lw=1, label="true")
    df_pred["yhat"].plot(ax=ax3, color="red", lw=1, label="pred")
    ax3.fill_between(
        df_pred.index,
        df_pred["yhat_lower"].tolist(),
        df_pred["yhat_upper"].tolist(),
        color="teal",
        lw=0,
        alpha=shade_alpha,
    )
    ax3.set_xlabel(None)
    ptitle = (
        f"{country} - Prediction (red) vs Observation (blue) ({title_scores})"
    )
    ax3.set_title(
        ptitle, loc="left", fontweight="bold", fontsize=ptitle_fontsize
    )
    ax3.grid(which="both", axis="both", color="lightgrey")
    _ = customize_splines(ax3)

    _ = sm.qqplot(ts.dropna(how="any"), fit=True, line="45", ax=ax4)
    ax4.set_title("Normal Q-Q for residual", loc="left", fontweight="bold")
    ax4.set_xlabel(ax4.get_xlabel(), fontsize=tick_label_fontsize)
    ax4.set_ylabel(ax4.get_xlabel(), fontsize=tick_label_fontsize)
    ax4.grid(which="both", axis="both", color="lightgrey")
    _ = customize_splines(ax4)

    ts.plot(ax=ax2, color="blue", zorder=3)
    ax2.axhline(y=0, color="k", zorder=0)
    ax2.set_title("Residual", loc="left", fontweight="bold")
    ax2.grid(which="both", axis="both", color="lightgrey", zorder=0)
    ax2.set_xlabel(None)
    _ = customize_splines(ax2)
