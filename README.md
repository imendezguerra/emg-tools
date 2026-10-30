# EMG Toolbox

[![PyPI](https://img.shields.io/pypi/v/emg-tools)](https://pypi.org/project/emg-tools/)
[![Python versions](https://img.shields.io/pypi/pyversions/emg-tools)](https://pypi.org/project/emg-tools/)
[![CI](https://github.com/imendezguerra/emg-tools/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/imendezguerra/emg-tools/actions/workflows/ci.yml)
[![Docs](https://github.com/imendezguerra/emg-tools/actions/workflows/docs.yml/badge.svg)](https://imendezguerra.github.io/emg-tools/)
[![DOI](https://zenodo.org/badge/<repo-id>.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## Overview
<!-- --8<-- [start:overview] -->
EMG Toolbox is a Python package to preprocess, analyse and plot electromyography (EMG) signals, including high-density recordings from electrode grids.

The package is composed of the following modules:

- `tools`: rearrange channels between `(samples, channels)` and grid `(rows, cols, samples)` formats, and replace bad channels with the mean of their neighbours.
- `prepro`: Butterworth bandpass, highpass, lowpass and powerline (bandstop) filters.
- `feats`: amplitude features such as the moving-window root mean square (RMS).
- `freq`: amplitude spectrum and power spectral density (periodogram or Welch).
- `plots`: plot channels over time, PSDs, PSD maps across channels, and spectrograms.
<!-- --8<-- [end:overview] -->

## Table of Contents
- [Installation](#installation)
- [Quick start](#quick-start)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Installation
<!-- --8<-- [start:install] -->
Install the latest release from PyPI (Python 3.10 or newer):

```sh
pip install emg-tools
```

The package is imported as `emg_tools`.
<!-- --8<-- [end:install] -->

To work on the code itself, see the development setup in [CONTRIBUTING.md](https://github.com/imendezguerra/emg-tools/blob/main/CONTRIBUTING.md#development-setup).

## Quick start
<!-- --8<-- [start:quickstart] -->
EMG data is an array of shape `(samples, channels)`. The example below filters a synthetic 16-channel recording and computes its RMS envelope:

```python
import numpy as np

from emg_tools import feats, prepro

fs = 2048                                   # sampling frequency (Hz)
timestamps = np.arange(5 * fs) / fs         # 5 s recording
rng = np.random.default_rng(0)
emg = rng.standard_normal((timestamps.size, 16))                # (samples, channels)
emg += 0.5 * np.sin(2 * np.pi * 50 * timestamps)[:, None]       # powerline noise

emg_filt = prepro.remove_powerline(emg, fs=fs, cutoff=50)
emg_filt = prepro.bandpass_filter(emg_filt, fs=fs, cutoff=[20, 500])

rms, t_win = feats.compute_rms(emg_filt, timestamps, win_len_s=0.25, win_step_s=0.125, fs=fs)
rms.shape                                   # (10240, 16): one RMS value per sample and channel
```
<!-- --8<-- [end:quickstart] -->

## Documentation
The full documentation, with a worked example and the API reference, is at
[imendezguerra.github.io/emg-tools](https://imendezguerra.github.io/emg-tools/).

## Contributing
Contributions are welcome! [CONTRIBUTING.md](https://github.com/imendezguerra/emg-tools/blob/main/CONTRIBUTING.md) explains how to set up a development environment, run the tests, preview the docs, and what the automated checks (pre-commit and CI) do.

## License
This project is licensed under the [MIT License](https://github.com/imendezguerra/emg-tools/blob/main/LICENSE).

## Citation

If you use this code in your research, please cite this repository:

```sh
@software{Mendez_Guerra_EMG_Toolbox,
author = {Mendez Guerra, Irene},
title = {{EMG Toolbox}},
url = {https://github.com/imendezguerra/emg-tools},
version = {2.0.0},
doi = {10.5281/zenodo.23820576}
}
```
## Contact

For any questions or inquiries, please contact us at:
```sh
Irene Mendez Guerra
irene.mendez17@imperial.ac.uk
```
