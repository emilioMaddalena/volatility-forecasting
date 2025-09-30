from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .asset import Asset


def plot_series(
    asset: Asset,
    series_types: List[str] | str,
    fig_size: tuple = (8, 6),
    labels: List[str] | None = None,
) -> None:
    """Plot different series linked to an asset."""
    color_sequence = ["k", "r", "b", "g", "m", "c", "y"]
    if isinstance(series_types, str):
        series_types = [series_types]
    if labels is None:
        labels = series_types

    plt.figure(figsize=fig_size)
    for i, series_type in enumerate(series_types):
        series = asset.__getattribute__(series_type)
        plt.plot(series.index, series, color=color_sequence[i], label=labels[i])
        # X ticks will be the years and nothing else
        years = pd.date_range(start=series.index[0], end=series.index[-1], freq="YS").year
        years = np.unique(
            np.concatenate(([series.index[0].year], years, [series.index[-1].year + 1]))
        )
        ticks = [pd.Timestamp(year=year, month=1, day=1) for year in years]
    plt.title(f"{asset.asset_name} ({asset.tick_name})")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.xlim([series.index[0], series.index[-1]])
    plt.xticks(ticks=ticks, labels=[str(year) for year in years], rotation=45)
    plt.legend()
    plt.grid()


def plot_autocorr(
    asset: Asset,
    series_type: str = "pointwise_volatility",
    max_lag: int = 30,
    fig_size: tuple = (8, 6),
) -> None:
    """Plot the autocorrelation of an asset series."""
    vol = asset.__getattribute__(series_type)
    autocorr = [vol.autocorr(lag) for lag in range(1, max_lag)]
    plt.figure(figsize=fig_size)
    plt.bar(range(1, 30), autocorr, color="k", zorder=10)
    plt.title(f"{asset.asset_name} ({asset.tick_name[1:]})")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.grid()


def plot_histogram(
    asset: Asset,
    series_type: str = "pointwise_volatility",
    nbins: int = 30,
    xlims: List[float] = None,
) -> None:
    """Plot the autocorrelation of an asset series."""
    plt.figure(figsize=(8, 6))
    series = asset.__getattribute__(series_type)
    sns.histplot(
        series.dropna(),
        color="k",
        alpha=1.0,
        bins=nbins,
        stat="density",
        zorder=10,
        edgecolor="white",
        linewidth=2.5,
    )
    sns.kdeplot(series.dropna(), bw_adjust=1, color="r", alpha=1.0, zorder=10, linewidth=2.0)
    plt.title(f"{asset.asset_name} ({asset.tick_name[1:]}) Histogram")
    if xlims:
        plt.xlim(xlims)
    plt.xlabel("Value")
    plt.grid()
