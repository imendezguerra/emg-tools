import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import signal

from emg_tools import plots

pytestmark = pytest.mark.plot


def test_plot_ch(emg, timestamps):
    _, ax = plt.subplots()
    out = plots.plot_ch(emg, timestamps, ax=ax)
    assert out is ax
    assert len(ax.lines) == emg.shape[1]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["0", "5", "10", "15"]


@pytest.mark.parametrize("log_scale", [False, True])
def test_plot_psd(emg, log_scale):
    ax = plots.plot_psd(emg, log_scale=log_scale)
    assert len(ax.lines) == emg.shape[1]
    assert ax.get_yscale() == ("log" if log_scale else "linear")


@pytest.mark.parametrize("log_scale", [False, True])
def test_plot_psd_map_with_zero_channel(emg, log_scale):
    emg[:, 0] = 0  # log scale must not fail on zero power
    ax = plots.plot_psd_map(emg, log_scale=log_scale)
    ax.figure.canvas.draw()
    assert ax.get_xlabel() == "Channels"


@pytest.mark.parametrize("log_scale", [False, True])
def test_plot_comp_spectrogram_per_channel(emg, log_scale):
    ax = plots.plot_comp_spectrogram(emg[:, :3], log_scale=log_scale)
    ax[0].figure.canvas.draw()
    assert [a.get_title() for a in ax] == ["Channel 0", "Channel 1", "Channel 2"]


def test_plot_comp_spectrogram_1d(emg):
    assert len(plots.plot_comp_spectrogram(emg[:, 0])) == 1


def test_plot_spectrogram_single_signal(emg, fs):
    f, t, sxx = signal.spectrogram(emg[:, 0], fs)
    assert len(plots.plot_spectrogram(f, t, sxx)) == 1

    _, ax = plt.subplots()
    out = plots.plot_spectrogram(f, t, sxx, ax=ax, rasterized=True)
    assert out[0] is ax
    assert ax.collections[0].get_rasterized()


def test_plot_spectrogram_multiple_signals(emg, fs):
    f, t, sxx = signal.spectrogram(emg[:, :2], fs, axis=0)
    _, axs = plt.subplots(1, 2)
    out = plots.plot_spectrogram(f, t, np.moveaxis(sxx, 1, -1), ax=axs)
    assert list(out) == list(axs)


def test_plot_ch_map(ch_map):
    ch_map = ch_map.copy()
    ch_map[0, 0] = -1  # empty cell
    ax = plots.plot_ch_map(np.arange(ch_map.size, dtype=float), ch_map, bad_ch=[1, 2])
    assert np.ma.getmaskarray(ax.collections[0].get_array()).ravel()[0]  # empty cell masked
    assert len(ax.patches) == 2
