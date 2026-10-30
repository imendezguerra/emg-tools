"""Plot EMG signals in time and frequency.

PSD axes are labelled in mV^2/Hz, which assumes the data is in mV.
"""

from copy import copy
from typing import Optional, Union

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy import signal

from emg_tools.freq import get_psd


def _min_positive(x: np.ndarray) -> float:
    """Smallest strictly positive value of x, used as default log-scale vmin"""
    positive = x[x > 0]
    return positive.min() if positive.size else np.finfo(float).tiny


def plot_ch(
    data: np.ndarray,
    timestamps: np.ndarray,
    delta: Optional[int] = 5,
    ax: Optional[plt.Axes] = None,
    palette_name: Optional[str] = "viridis",
    **kwarg: Optional[Union[str, int, float]]
    )-> plt.Axes:

    """Plot all channels stacked vertically in a single axes.

    Each channel is shifted up by `max(data) / delta` times its index. Every fifth
    channel is labelled on the y axis.

    Args:
        data (np.ndarray): Float data with shape (samples, chs).
        timestamps (np.ndarray): Time of each sample in seconds, with shape (samples,).
        delta (int, optional): Divisor of the data maximum that sets the spacing
            between channels. Larger values bring channels closer. Default is 5.
        ax (plt.Axes, optional): Axes to plot on. Default is None, which creates a
            new figure.
        palette_name (str, optional): Name of the seaborn palette used to colour
            the channels. Default is "viridis".
        **kwarg: Additional keyword arguments passed to `matplotlib.axes.Axes.plot`.

    Returns:
        plt.Axes: Axes where the data is plotted.
    """

    # Data size
    chs = data.shape[1]

    # Compute channel offset
    max_val = np.nanmax(data)
    offset = max_val/delta

    # Apply offset to signals
    data_plot = copy(data)
    data_plot += range(0,chs) * offset

    # Plot signals
    if ax is None:
        _, ax = plt.subplots()

    # Create color palette and assign it to the axis cycle
    color_palette = sns.color_palette(palette_name, chs).as_hex()
    ax.set_prop_cycle(color=color_palette)

    ax.plot(timestamps, data_plot, **kwarg)
    ax.margins(0) # padding for each limit of the axis
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channels')
    ax.grid(True, axis='x')
    ch_ticks = np.linspace(0, (chs -1) * offset, chs)
    sel_chs = np.arange(0, chs, 5)
    ax.set_yticks(ch_ticks[sel_chs], sel_chs)

    return ax


def plot_ch_map(
    values: np.ndarray,
    ch_map: np.ndarray,
    bad_ch: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    palette_name: Optional[str] = "viridis",
    cbar_label: Optional[str] = None,
    **kwarg: Optional[Union[str, int, float]]
) -> plt.Axes:

    """Plot one value per channel on the electrode grid, annotated with channel
    numbers.

    Args:
        values (np.ndarray): Value per channel with shape (channels,).
        ch_map (np.ndarray): Channel map with shape (rows, cols). Empty cells
            (-1) are left blank.
        bad_ch (np.ndarray, optional): Channels outlined in red. Default is None.
        ax (plt.Axes, optional): Axes to plot the map. Default is None.
        palette_name (str, optional): Name of the colormap. Default is "viridis".
        cbar_label (str, optional): Colorbar label. Default is None.
        **kwarg: Additional keyword arguments to be passed to the `heatmap`
            function of seaborn.

    Returns:
        plt.Axes: Axes where the map is plotted.
    """

    if ax is None:
        _, ax = plt.subplots()

    # Arrange values on the grid, leaving empty cells blank
    values_map = np.where(ch_map >= 0, values[ch_map], np.nan)

    sns.heatmap(
        values_map, annot=ch_map, fmt="d", annot_kws={"fontsize": 6},
        cmap=palette_name, square=True, xticklabels=False, yticklabels=False,
        cbar_kws={"label": cbar_label}, ax=ax, **kwarg
    )

    # Outline bad channels
    if bad_ch is not None:
        for row, col in np.argwhere(np.isin(ch_map, bad_ch)):
            ax.add_patch(Rectangle((col, row), 1, 1, fill=False, edgecolor="red", lw=1.5))

    return ax


def plot_psd(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    ax: Optional[plt.Axes] = None,
    palette_name: Optional[str] = "viridis",
    log_scale: Optional[bool] = False,
    **kwarg: Optional[Union[str, int, float]]
) -> plt.Axes:

    """Plot the periodogram PSD of each channel as one line per channel.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        fs (int, optional): Sampling frequency in Hz. Default is 2048.
        ax (plt.Axes, optional): Axes to plot on. Default is None, which creates a
            new figure.
        palette_name (str, optional): Name of the seaborn palette used to colour
            the channels. Default is "viridis".
        log_scale (bool, optional): If True, the PSD axis is logarithmic. Default
            is False.
        **kwarg: Additional keyword arguments passed to `matplotlib.axes.Axes.plot`.

    Returns:
        plt.Axes: Axes where the PSD is plotted.
    """

    # Initialise variables
    chs = data.shape[1]

    # Define axis if None
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10,5), layout='tight')

    # Create color palette and assign it to the axis cycle
    color_palette = sns.color_palette(palette_name, chs).as_hex()
    ax.set_prop_cycle(color=color_palette)

    # Plot PSD
    psd, xf = get_psd(data, fs)

    if log_scale:
        ax.semilogy(xf, psd, **kwarg)
        ax.set_ylabel('PSD (mV^2/Hz)')
    else:
        ax.plot(xf, psd, **kwarg)
        ax.set_ylabel('PSD (mV^2/Hz)')
    ax.set_xlabel('Frequency (Hz)')

    return ax

def plot_psd_map(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    ax: Optional[plt.Axes] = None,
    log_scale: Optional[bool] = False,
    eps: Optional[float] = None,
    palette_name: Optional[str] = "viridis",
    **kwarg: Optional[Union[str, int, float]]
) -> plt.Axes:

    """Plot the periodogram PSD as a heatmap of frequency against channel.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        fs (int, optional): Sampling frequency in Hz. Default is 2048.
        ax (plt.Axes, optional): Axes to plot on. Default is None, which creates a
            new figure.
        log_scale (bool, optional): If True, the colour scale is logarithmic. Default
            is False.
        eps (float, optional): Lower limit of the log colour scale, to avoid log(0).
            Only used if log_scale is True. Default is None, which uses the smallest
            positive PSD value.
        palette_name (str, optional): Name of the matplotlib colormap. Default is
            "viridis".
        **kwarg: Additional keyword arguments passed to
            `matplotlib.axes.Axes.pcolormesh`.

    Returns:
        plt.Axes: Axes where the PSD is plotted.
    """

    # Initialise variables
    samples, chs = data.shape

    # Plot PSD
    psd, xf = get_psd(data, fs)

    # Plot PSD
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10,5), layout='tight')
    if log_scale:
        if eps is None:
            eps = _min_positive(psd)
        norm=colors.LogNorm(vmin=eps, vmax=psd.max())
        cbar_label = 'PSD (mV^2/Hz)'
    else:
        norm=None
        cbar_label = 'PSD (mV^2/Hz)'
    im = ax.pcolormesh(
        np.arange(chs), xf, psd,
        shading='gouraud', cmap=palette_name, norm=norm,
        **kwarg
        )
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, rotation=90)
    ax.grid(False)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Channels')

    return ax


def plot_comp_spectrogram(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    ax: Optional[Union[plt.Axes, np.ndarray]] = None,
    log_scale: Optional[bool] = False,
    eps: Optional[float] = None,
    palette_name: Optional[str] = "viridis",
    **kwarg: Optional[Union[str, int, float]]
) -> np.ndarray:

    """Compute the spectrogram of each channel and plot it with `plot_spectrogram`.

    Args:
        data (np.ndarray): Input data with shape (samples, chs) or (samples,).
        fs (int, optional): Sampling frequency in Hz. Default is 2048.
        ax (plt.Axes or np.ndarray, optional): Axes to plot on, one per channel.
            Default is None, which creates one row of axes per channel.
        log_scale (bool, optional): If True, the colour scale is logarithmic. Default
            is False.
        eps (float, optional): Lower limit of the colour scale. Default is None, which
            uses the smallest positive value in log scale and the minimum otherwise.
        palette_name (str, optional): Name of the matplotlib colormap. Default is
            "viridis".
        **kwarg: Additional keyword arguments passed to
            `matplotlib.axes.Axes.pcolormesh`.

    Returns:
        np.ndarray: Array of axes where the spectrogram of each channel is plotted.
    """

    # Compute spectrogram (one sided) with shape (freqs, chs, times)
    f, t, sxx = signal.spectrogram(data, fs, axis=0)

    # Rearrange to (freqs, times, chs)
    if sxx.ndim == 3:
        sxx = np.moveaxis(sxx, 1, -1)

    return plot_spectrogram(
        f, t, sxx, ax=ax, palette_name=palette_name,
        log_scale=log_scale, eps=eps, **kwarg
    )

def plot_spectrogram(
    f: np.ndarray,
    t: np.ndarray,
    sxx: np.ndarray,
    ax: Optional[Union[plt.Axes, np.ndarray]] = None,
    palette_name: Optional[str] = "magma",
    log_scale: Optional[bool] = False,
    eps: Optional[float] = None,
    **kwarg: Optional[Union[str, int, float]]
) -> np.ndarray:

    """Plot a precomputed spectrogram, one axes per signal.

    All signals share the same colour scale.

    Args:
        f (np.ndarray): Frequencies in Hz, with shape (freqs,).
        t (np.ndarray): Times in seconds, with shape (times,).
        sxx (np.ndarray): Spectrogram with shape (freqs, times, signals), or
            (freqs, times) for a single signal.
        ax (plt.Axes or np.ndarray, optional): Axes to plot on, one per signal.
            Default is None, which creates one row of axes per signal.
        palette_name (str, optional): Name of the matplotlib colormap. Default is
            "magma".
        log_scale (bool, optional): If True, the colour scale is logarithmic. Default
            is False.
        eps (float, optional): Lower limit of the colour scale. Default is None, which
            uses the smallest positive value in log scale and the minimum otherwise.
        **kwarg: Additional keyword arguments passed to
            `matplotlib.axes.Axes.pcolormesh`.

    Returns:
        np.ndarray: Array of axes where the spectrogram of each signal is plotted.
    """

    # Normalise spectrogram if log scale
    if log_scale:
        if eps is None:
            eps = _min_positive(sxx)
        norm=colors.LogNorm(vmin=eps, vmax=sxx.max())
        cbar_label = 'PSD (mV^2/Hz)'
    else:
        if eps is None:
            eps = sxx.min()
        norm=colors.Normalize(vmin=eps, vmax=sxx.max())
        cbar_label = 'PSD (mV^2/Hz)'

    # Plot spectrogram per signal
    if len(sxx.shape) == 2:
        sxx = np.expand_dims(sxx, axis=-1)
    nsig = sxx.shape[-1]

    if ax is None:
        cols = 1
        rows = np.round(nsig/cols).astype(int)

        fig, ax = plt.subplots(
            rows, cols, figsize=(12, 6),
            sharex=True, sharey=True, layout='constrained', squeeze=False
        )
    ax = np.atleast_1d(ax).flatten()

    for i in range(nsig):
        # Plot spectrogram
        im = ax[i].pcolormesh(
            t, f, sxx[:,:,i],
            shading='gouraud', cmap=palette_name, norm=norm,
            **kwarg
        )
        cbar = ax[i].figure.colorbar(im, ax=ax[i])
        cbar.set_label(cbar_label, rotation=90)
        ax[i].set_ylabel('Frequency (Hz)')
        ax[i].set_xlabel('Time (s)')
        ax[i].set_title(f'Channel {i}')

    return ax
