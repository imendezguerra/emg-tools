# Getting started

## Installation

--8<-- "README.md:install"

## Data conventions

- **EMG data** is an array of shape `(samples, channels)`, with time along `axis=0`.
- **Grid data** has shape `(rows, cols, samples)`. Convert between both formats with
  `tools.arrange_data_spatially` and `tools.flatten_data_spatially`.
- **The channel map** `ch_map` is an integer array of shape `(rows, cols)` holding the 0-based
  channel index of each electrode. Empty grid positions are marked with `-1`.
- **Bad channels** are a list of 0-based channel indices.
- **Timestamps** are the time in seconds of each sample, e.g. `np.arange(n_samples) / fs`.
- **The sampling frequency** `fs` is in Hz (default `2048`).
- **Units:** PSD axes are labelled in mV²/Hz, which assumes the data is in mV.

## Quick start

--8<-- "README.md:quickstart"

## A full workflow on an electrode grid

The example below walks through a typical high-density EMG analysis: filter, replace a bad
channel, check the spectrum, compute the RMS envelope and map it on the grid. The code on this
page runs every time the docs are built, so the figures always match the current code.

<!-- Hidden setup: renders each figure inline as a PNG. -->
```python exec="true" session="gs"
import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def show():
    """Print the current figure as an inline PNG and close all figures."""
    buf = io.BytesIO()
    plt.gcf().savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close("all")
    print(f'<img alt="figure" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">')
```

The data is a synthetic 4×4 grid recorded during a 5 s contraction that ramps up and down. It
contains three common artefacts: 50 Hz powerline interference, a slow baseline drift and a
dead channel. The random generator has a fixed seed, so the data, numbers and figures are the
same on every run:

```python exec="true" source="above" session="gs" html="true"
import numpy as np

from emg_tools import feats, freq, plots, prepro, tools

fs = 2048                                   # sampling frequency (Hz)
n_samples = 5 * fs                          # 5 s recording
timestamps = np.arange(n_samples) / fs
ch_map = np.arange(16).reshape(4, 4)        # 4x4 grid, channels 0..15 in row-major order

# Synthetic EMG: broadband activity shared across the grid plus some independent noise per
# channel, with an amplitude that ramps up and down (a contraction)
rng = np.random.default_rng(seed=0)         # fixed seed: reproducible data
envelope = np.sin(np.pi * timestamps / timestamps[-1])[:, None]       # (samples, 1)
shared = rng.standard_normal((n_samples, 1))
emg = envelope * (shared + 0.3 * rng.standard_normal((n_samples, ch_map.size)))  # (samples, chs)

# Artefacts: 50 Hz powerline interference, a slow baseline drift and a dead channel
emg += 0.5 * np.sin(2 * np.pi * 50 * timestamps)[:, None]
emg += 0.2 * np.sin(2 * np.pi * 0.5 * timestamps)[:, None]
emg[:, 5] = 0

plots.plot_ch(emg, timestamps, delta=0.5)   # channels stacked over time; 5 is flat
show()  # markdown-exec: hide
```

### Filter

Filter first: powerline interference and baseline drift can hide a bad channel or make a good
one look bad, so bad channels are easier to identify on filtered data. Remove the powerline
interference with a narrow bandstop filter, then keep the usual EMG band (20–500 Hz), which also
removes the drift:

```python exec="true" source="above" session="gs" html="true"
emg_filt = prepro.remove_powerline(emg, fs=fs, cutoff=50, width=1)
emg_filt = prepro.bandpass_filter(emg_filt, fs=fs, cutoff=[20, 500], order=4)

plots.plot_ch(emg_filt, timestamps, delta=0.5)   # channel 5 stands out as flat
show()  # markdown-exec: hide
```

All filters are Butterworth filters applied forward and backward (`filtfilt=True`), so they do
not shift the signal in time. Set `filtfilt=False` for causal (forward only) filtering.
`highpass_filter` and `lowpass_filter` are also available.

### Replace bad channels

`replace_bad_ch` replaces each bad channel with the mean of its good neighbours on the grid
(up to 8 surrounding electrodes). Channel 5 sits at row 1, column 1, so it becomes the mean of
channels 0, 1, 2, 4, 6, 8, 9 and 10:

```python exec="true" source="above" session="gs" html="true"
emg_clean = tools.replace_bad_ch(emg_filt, bad_ch=[5], ch_map=ch_map)
```

- Bad channels are replaced in order, so an already replaced channel counts as a good neighbour
  for the next one.
- A bad channel with no good neighbours is set to zero.
- Grid data `(rows, cols, samples)` is accepted too.

### Check the spectrum

`get_psd` returns the one-sided PSD of every channel and its frequency axis. Comparing it before
and after filtering confirms that the artefacts are gone:

```python exec="true" source="above" session="gs" html="true"
psd_raw, xf = freq.get_psd(emg, fs)
psd_filt, _ = freq.get_psd(emg_filt, fs)

i50 = np.argmin(np.abs(xf - 50))
psd_raw[i50, 0], psd_filt[i50, 0]           # (0.61, 0.0003): 50 Hz attenuated ~2000 times

# Welch's method gives a smoother estimate, at a lower frequency resolution
psd_welch, xf_welch = freq.get_psd(emg_clean, fs, welch=True, nperseg=512)
psd_welch.shape                             # (257, 16), one bin every 4 Hz

fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True, layout="constrained")
plots.plot_psd(emg, fs, log_scale=True, ax=axs[0])   # one PSD line per channel
plots.plot_psd(emg_filt, fs, log_scale=True, ax=axs[1])
axs[0].set_title("Before filtering")
axs[1].set_title("After filtering")
axs[0].set_ylim(1e-10, 1)                  # hide the near-zero DC bin
show()  # markdown-exec: hide
```

`plot_psd_map` shows the same PSD as a heatmap of frequency against channel, which makes a
channel with a different spectrum stand out:

```python exec="true" source="above" session="gs" html="true"
plots.plot_psd_map(emg_clean, fs, log_scale=True, eps=1e-10)   # eps: lower colour limit
show()  # markdown-exec: hide
```

`get_spectrum` returns the amplitude spectrum instead, in the same units as the data.

### RMS envelope

`compute_rms` computes the RMS in sliding windows and interpolates it back to every sample, so
the result has the same shape as the data:

```python exec="true" source="above" session="gs" html="true"
rms, t_win = feats.compute_rms(emg_clean, timestamps, win_len_s=0.25, win_step_s=0.125, fs=fs)
rms.shape                                   # (10240, 16)
t_win[:3]                                   # ~[0.125, 0.25, 0.375]: window centres (s)
rms[[fs // 2, int(2.5 * fs), int(4.5 * fs)], 0]   # array([0.21, 0.68, 0.22]): follows the ramp

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(timestamps, rms)
ax.set(xlabel="Time (s)", ylabel="RMS", title="RMS envelope, one line per channel")
show()  # markdown-exec: hide
```

### Map the RMS on the grid

`arrange_data_spatially` turns `(samples, channels)` data into `(rows, cols, samples)`, and
`flatten_data_spatially` turns it back:

```python exec="true" source="above" session="gs" html="true"
rms_grid = tools.arrange_data_spatially(rms, ch_map)   # (4, 4, 10240)
tools.flatten_data_spatially(rms_grid, ch_map)         # back to (10240, 16)
```

`plot_ch_map` draws one value per channel on the grid, labelled with the channel numbers. At the
peak of the contraction every channel is between 0.66 and 0.71 (note the narrow colour scale),
including channel 5, outlined in red, which was replaced by the mean of its neighbours:

```python exec="true" source="above" session="gs" html="true"
peak = np.argmax(rms.mean(axis=1))
plots.plot_ch_map(rms[peak], ch_map, bad_ch=[5], cbar_label="RMS")
show()  # markdown-exec: hide
```

### Spectrogram

`plot_comp_spectrogram` computes and plots the spectrogram of each channel, one row of axes per
channel. Here, the first two channels:

```python exec="true" source="above" session="gs" html="true"
plots.plot_comp_spectrogram(emg_clean[:, :2], fs, log_scale=True)
show()  # markdown-exec: hide
```

### Plotting tips

- Every plotting function returns its axes and accepts an existing `ax`, so plots can be
  combined in one figure, as in the PSD comparison above.
- `delta` sets the spacing between channels in `plot_ch`: smaller values spread them further
  apart.
- `palette_name` sets the colours (a seaborn palette for line plots, a matplotlib colormap for
  heatmaps, grid maps and spectrograms).
- `plot_spectrogram` plots a spectrogram you have already computed, e.g. with
  `scipy.signal.spectrogram`.
